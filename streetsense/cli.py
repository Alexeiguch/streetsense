"""Command line entry point.

  python -m streetsense run   --source samples/street.webm --out outputs/annotated.mp4 --alerts outputs/alerts.jsonl
  python -m streetsense run   --source 0 --speak            # webcam, spoken alerts
  python -m streetsense ask   --image frame.png --question "Can I cross now?"   # Muse Spark, needs MODEL_API_KEY
  python -m streetsense sam   --image frame.png --out docs/benchmark/sam.png
  python -m streetsense serve --port 8000                    # phone-camera web demo
  python -m streetsense guide --source clip.webm --at 8 --question "How do I get to the crosswalk?"  # Muse Wonder, opt in
  python -m streetsense glasses-sim --source clip.webm --resolution medium --fps 15  # Ray-Ban stream profile, offline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


class VideoClock:
    """For files, alerts are timed in *video* seconds, so rate limits match what a walker would hear."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def cmd_run(a: argparse.Namespace) -> int:
    import cv2

    from .config import Config
    from .pipeline import Pipeline
    from .speech import Speaker

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"cannot open source {a.source}", file=sys.stderr)
        return 2
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    is_file = not isinstance(src, int)
    if is_file and a.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)

    cfg = Config()
    cfg.detector.imgsz = a.imgsz
    cfg.detector.weights = a.weights
    cfg.detector.backend = a.backend
    if getattr(a, "conf", None) is not None:
        cfg.detector.conf = a.conf
    cfg.surface.backend = getattr(a, "surface_backend", None) or a.backend
    cfg.vlm.enabled = not a.no_vlm
    cfg.open_vocab.enabled = bool(getattr(a, "open_vocab", False))
    if getattr(a, "open_vocab_weights", None):
        cfg.open_vocab.weights = a.open_vocab_weights
    cfg.depth.enabled = bool(getattr(a, "depth", False))
    if getattr(a, "depth_backend", None):
        cfg.depth.backend = a.depth_backend
    if getattr(a, "depth_every", None):
        cfg.depth.every_n_frames = int(a.depth_every)
    cfg.sam.enabled = bool(getattr(a, "sam", False))
    clock = VideoClock() if is_file else time.monotonic
    pipe = Pipeline(cfg, use_surface=not a.no_surface, clock=clock)
    speaker = Speaker(enabled=a.speak)
    surf_b = pipe.surface_seg.backend if pipe.surface_seg else "off"
    depth_b = pipe.depth.backend if pipe.depth else "off"
    print(f"[streetsense] source={a.source} fps_in={fps_in:.1f} detect={pipe.detector.backend} "
          f"surface={surf_b} depth={depth_b} "
          f"meta_api={'on' if pipe.vlm.enabled else 'off (no MODEL_API_KEY)'} sam={'on' if pipe.sam else 'off'} speech={speaker.backend}")

    writer = None
    alerts_f = open(a.alerts, "w") if a.alerts else None
    frames_dir = Path(a.save_frames_dir) if a.save_frames_dir else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)
    stride = max(1, a.stride)
    n = 0
    raw_idx = 0
    t_start = time.perf_counter()
    proc_time = 0.0
    last_shape = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_idx += 1
        if (raw_idx - 1) % stride:
            continue
        if a.resize and frame.shape[1] > a.resize:
            s = a.resize / frame.shape[1]
            frame = cv2.resize(frame, (a.resize, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        last_shape = frame.shape
        if is_file:
            clock.t = a.start + (raw_idx - 1) / fps_in
        t0 = time.perf_counter()
        out = pipe.process(frame, annotate=bool(a.out or frames_dir))
        proc_time += time.perf_counter() - t0
        if out.spoken:
            delivered = speaker.say(out.spoken)
            pipe.alerts.acknowledge_delivery(out.spoken, bool(delivered))
            rec = {"t_video_s": round(out.t, 2) if is_file else None, "frame": raw_idx - 1,
                   "priority": out.spoken.priority.name, "key": out.spoken.key, "text": out.spoken.text}
            if alerts_f:
                alerts_f.write(json.dumps(rec) + "\n")
                alerts_f.flush()
            if not a.speak:
                print(f"  t={out.t:6.2f}s  [{rec['priority']:8}] {rec['text']}")
        if a.out:
            if writer is None:
                h, w = out.annotated.shape[:2]
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in / stride, (w, h))
            writer.write(out.annotated)
        if frames_dir and a.save_every and n % a.save_every == 0:
            cv2.imwrite(str(frames_dir / f"frame_{raw_idx - 1:05d}.png"), out.annotated)
        n += 1
        if a.max_frames and n >= a.max_frames:
            break
    cap.release()
    if writer:
        writer.release()
    if alerts_f:
        alerts_f.close()
    wall = time.perf_counter() - t_start
    stats = {
        "frames_processed": n,
        "pipeline_fps": round(n / proc_time, 2) if proc_time else None,
        "wall_fps_incl_decode_and_encode": round(n / wall, 2) if wall else None,
        "mean_stage_ms": {k: round(v / max(1, n), 1) for k, v in pipe.timing_totals.items()},
        "input_resolution": list(last_shape[1::-1]) if last_shape else None,
        "yolo_imgsz": cfg.detector.imgsz,
        "surface_every_n_frames": cfg.surface.every_n_frames if pipe.surface_seg else None,
    }
    print(json.dumps(stats, indent=2))
    if a.stats:
        Path(a.stats).write_text(json.dumps(stats, indent=2))
    return 0


def cmd_ask(a: argparse.Namespace) -> int:
    import cv2

    from .config import VisionLLMConfig
    from .meta_api import MetaVisionAssistant

    img = cv2.imread(a.image)
    if img is None:
        print(f"cannot read {a.image}", file=sys.stderr)
        return 2
    vlm = MetaVisionAssistant(VisionLLMConfig(model=a.model))
    if not vlm.enabled:
        print("Meta Model API disabled: export MODEL_API_KEY=... (claim the hackathon credits first)", file=sys.stderr)
        return 3
    t0 = time.perf_counter()
    res = vlm.ask_sync(img, a.question)
    from .meta_usage import get_ledger
    out = {"latency_s": round(time.perf_counter() - t0, 2), **res, "meta_usage": get_ledger().summary()}
    print(json.dumps(out, indent=2))
    return 0



def cmd_sam(a: argparse.Namespace) -> int:
    """On-demand SAM 3.1 text prompts; writes an overlay PNG when --out is set."""
    import cv2

    from .meta_api import MetaSamSegmenter, overlay_sam_segments
    from .meta_usage import get_ledger

    img = cv2.imread(a.image)
    if img is None:
        print(f"cannot read {a.image}", file=sys.stderr)
        return 2
    sam = MetaSamSegmenter(model=a.model, max_image_side=a.max_side)
    if not sam.enabled:
        print("Meta SAM disabled: export MODEL_API_KEY=...", file=sys.stderr)
        return 3
    prompts = [p.strip() for p in a.prompts.split(",") if p.strip()]
    segs = [sam.segment(img, p, decode_masks=not a.boxes_only) for p in prompts]
    payload = {
        "segments": [
            {"prompt": s.prompt, "n_boxes": len(s.boxes), "n_masks": len(s.masks),
             "boxes": s.boxes, "latency_s": s.latency_s, "cost_usd": s.cost_usd}
            for s in segs
        ],
        "meta_usage": get_ledger().summary(),
    }
    if a.out:
        ov = overlay_sam_segments(img, segs)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.out, ov)
        payload["overlay"] = a.out
    print(json.dumps(payload, indent=2))
    return 0


def _local_config(a: argparse.Namespace):
    from .config import Config

    cfg = Config()
    cfg.detector.weights = getattr(a, "weights", cfg.detector.weights)
    cfg.detector.backend = getattr(a, "backend", "openvino")
    cfg.surface.backend = getattr(a, "backend", "openvino")
    cfg.vlm.interval_s = float("inf")  # no background Muse calls; only explicit ones
    return cfg


def _sample_frames(source: str, start: float, seconds: float, fps: float):
    """Yield (video_t, frame) from a file at `fps` in video time, or one still image."""
    import cv2

    img = cv2.imread(source) if Path(source).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else None
    if img is not None:
        yield 0.0, img
        return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    step = max(1, int(round(fps_in / max(0.5, fps))))
    i = 0
    t0 = start
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = t0 + i / fps_in
        if t - start > seconds:
            break
        if i % step == 0:
            yield t, frame
        i += 1
    cap.release()


def cmd_guide(a: argparse.Namespace) -> int:
    """Opt-in Muse Wonder guide: warm the local log on a clip, then ask one planning question."""
    from .config import WonderConfig
    from .glasses import GlassesStreamConfig, adapt_frame
    from .meta_usage import get_ledger
    from .pipeline import Pipeline
    from .wonder_agent import GuideTools, WonderGuide

    cfg = _local_config(a)
    gcfg = GlassesStreamConfig(resolution=a.glasses) if a.glasses else None
    if gcfg is not None:
        cfg.camera.hfov_deg = gcfg.hfov_deg
    clock = VideoClock()
    pipe = Pipeline(cfg, use_surface=not a.no_surface, clock=clock)
    start = max(0.0, a.at - a.warmup)
    frame = out = None
    for t, f in _sample_frames(a.source, start, a.warmup, a.sample_fps):
        clock.t = t
        if gcfg is not None:
            f = adapt_frame(f, gcfg)
        out = pipe.process(f, annotate=False)
        frame = f
    if frame is None:
        print("no frames read", file=sys.stderr)
        return 2
    wcfg = WonderConfig(enabled=True, reasoning_effort=a.effort, max_turns=a.max_turns)
    guide = WonderGuide(wcfg)
    if not guide.available:
        print("Muse Wonder disabled: export MODEL_API_KEY=...", file=sys.stderr)
        return 3
    maps = None
    if a.maps:
        from .config import MapsConfig
        from .maps import build_session

        maps = build_session(MapsConfig(enabled=True, provider=a.maps_provider, google=a.google_maps))
        lat, lon = (float(x) for x in a.gps.split(","))
        maps.set_position(lat, lon, heading=a.heading, accuracy_m=10.0)
    res = guide.run(a.question, frame, GuideTools.from_pipeline(pipe, frame, out, clock=clock, maps=maps))
    if maps is not None and maps.nav is not None:
        res_route = maps.nav.status()
    else:
        res_route = None
    payload = {"source": a.source, "at_s": a.at, "route": res_route, "warmup_frames": len(pipe.observations),
               "glasses": gcfg.to_dict() if gcfg else None, **res.to_dict(),
               "meta_usage": get_ledger().summary()}
    print(json.dumps(payload, indent=2, default=str))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a") as f:
            f.write(json.dumps({"question": a.question, **payload}, default=str) + "\n")
    return 0


def cmd_glasses_sim(a: argparse.Namespace) -> int:
    """Run the local pipeline on a clip reshaped to a Ray-Ban toolkit stream. No Meta calls."""
    import statistics

    from .glasses import GlassesStreamConfig, adapt_frame, glasses_cue
    from .pipeline import Pipeline

    gcfg = GlassesStreamConfig(resolution=a.resolution, frame_rate=a.fps, jpeg_quality=a.jpeg_quality)
    cfg = _local_config(a)
    cfg.vlm.enabled = False
    cfg.camera.hfov_deg = gcfg.hfov_deg
    clock = VideoClock()
    pipe = Pipeline(cfg, use_surface=not a.no_surface, clock=clock)
    lat, cues = [], []
    for t, f in _sample_frames(a.source, a.start, a.seconds, gcfg.frame_rate):
        clock.t = t
        g = adapt_frame(f, gcfg)
        t0 = time.perf_counter()
        out = pipe.process(g, annotate=False)
        lat.append((time.perf_counter() - t0) * 1000)
        cue = glasses_cue(out.spoken)
        if cue:
            cues.append({"t": round(t, 2), **cue})
    if not lat:
        print("no frames read", file=sys.stderr)
        return 2
    lat_sorted = sorted(lat)
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))]
    budget_ms = 1000.0 / gcfg.frame_rate
    stats = {
        "source": a.source,
        "profile": gcfg.to_dict(),
        "frames": len(lat),
        "process_ms_p50": round(statistics.median(lat), 1),
        "process_ms_p95": round(p95, 1),
        "frame_budget_ms": round(budget_ms, 1),
        "keeps_up_p50": statistics.median(lat) <= budget_ms,
        "cues": cues,
    }
    print(json.dumps(stats, indent=2))
    if a.stats:
        Path(a.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(a.stats).write_text(json.dumps(stats, indent=2))
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .web import create_app

    app = create_app(no_surface=a.no_surface, backend=a.backend,
                    surface_backend=getattr(a, "surface_backend", None))
    app.run(host=a.host, port=a.port, threaded=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="streetsense")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process a video file or webcam")
    r.add_argument("--source", required=True, help="video path or webcam index")
    r.add_argument("--out", help="annotated output video (.mp4)")
    r.add_argument("--alerts", help="JSONL log of spoken alerts")
    r.add_argument("--stats", help="write timing stats JSON here")
    r.add_argument("--save-frames-dir")
    r.add_argument("--save-every", type=int, default=0)
    r.add_argument("--max-frames", type=int, default=0)
    r.add_argument("--start", type=float, default=0.0, help="start offset in seconds (files)")
    r.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    r.add_argument("--resize", type=int, default=1280, help="downscale wider frames to this width")
    r.add_argument("--imgsz", type=int, default=640)
    r.add_argument("--weights", default="models/yolo11s.pt",
                    help="detector weights (default yolo11s; try yolo11n/yolo11m/rtdetr-l/rfdetr-nano/yolo26n)")
    r.add_argument("--conf", type=float, default=None,
                    help="global detector confidence floor (default 0.35; person uses 0.28)")
    r.add_argument("--backend", default="openvino", choices=["openvino", "torch"],
                    help="default inference backend for detector + surface (openvino|torch)")
    r.add_argument("--surface-backend", default=None, choices=["openvino", "torch"],
                    help="override surface segmenter backend (default: same as --backend)")
    r.add_argument("--open-vocab", action="store_true",
                    help="enable YOLO-World hazard loop every N frames (bollard/scooter/...)")
    r.add_argument("--open-vocab-weights", default=None,
                    help="open-vocab weights path (default models/yolov8s-worldv2.pt)")
    r.add_argument("--depth", action="store_true",
                    help="enable Depth Anything V2 Small (Apache-2.0) every N frames")
    r.add_argument("--depth-backend", default=None, choices=["openvino", "torch"],
                    help="depth backend (default: openvino)")
    r.add_argument("--depth-every", type=int, default=None,
                    help="run depth every N processed frames (default 6)")
    r.add_argument("--no-surface", action="store_true")
    r.add_argument("--sam", action="store_true",
                    help="enable SAM 3.1 crosswalk boost (slow; needs MODEL_API_KEY)")
    r.add_argument("--no-vlm", action="store_true")
    r.add_argument("--speak", action="store_true")
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("ask", help="one-shot Muse Spark question about an image")
    q.add_argument("--image", required=True)
    q.add_argument("--question", default="Is it safe to cross the street right now?")
    q.add_argument("--model", default="muse-spark-1.3")
    q.set_defaults(fn=cmd_ask)
    m = sub.add_parser("sam", help="on-demand SAM 3.1 text-prompted segmentation")
    m.add_argument("--image", required=True)
    m.add_argument("--prompts", default="crosswalk,curb,tactile paving,sidewalk")
    m.add_argument("--model", default="sam-3.1")
    m.add_argument("--max-side", type=int, default=640)
    m.add_argument("--out", help="write overlay PNG here")
    m.add_argument("--boxes-only", action="store_true", help="skip mask decode")
    m.set_defaults(fn=cmd_sam)
    s = sub.add_parser("serve", help="web demo: phone camera -> alerts spoken in the browser")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-surface", action="store_true")
    s.add_argument("--backend", default="openvino", choices=["openvino", "torch"])
    s.add_argument("--surface-backend", default=None, choices=["openvino", "torch"])
    s.set_defaults(fn=cmd_serve)
    g = sub.add_parser("guide", help="opt-in Muse Wonder guide agent over local facts (needs MODEL_API_KEY)")
    g.add_argument("--source", required=True, help="video file or still image")
    g.add_argument("--question", required=True)
    g.add_argument("--at", type=float, default=8.0, help="video second the question is asked at")
    g.add_argument("--warmup", type=float, default=6.0, help="seconds of local processing before the question")
    g.add_argument("--sample-fps", type=float, default=5.0)
    g.add_argument("--effort", default="low", choices=["minimal", "low", "medium", "high", "max"])
    g.add_argument("--max-turns", type=int, default=4)
    g.add_argument("--glasses", default=None, choices=["low", "medium", "high"],
                   help="reshape frames to a Ray-Ban toolkit stream first")
    g.add_argument("--weights", default="models/yolo11s.pt")
    g.add_argument("--backend", default="openvino", choices=["openvino", "torch"])
    g.add_argument("--no-surface", action="store_true")
    g.add_argument("--out", help="append the JSON result to this JSONL file")
    g.add_argument("--maps", action="store_true", help="give Wonder the map tools (plan_route, route_status, find_nearby)")
    g.add_argument("--maps-provider", default="fixture", choices=["fixture", "osm", "record"])
    g.add_argument("--gps", default="38.9031,-77.0387", help="walker position lat,lon (default Farragut Square, DC)")
    g.add_argument("--heading", type=float, default=180.0)
    g.add_argument("--google-maps", action="store_true",
                   help="add google_maps_search via Maps Experience agent tools (needs GOOGLE_MAPS_API_KEY)")
    g.set_defaults(fn=cmd_guide)
    gs = sub.add_parser("glasses-sim", help="run the local pipeline on a clip shaped like a Ray-Ban stream")
    gs.add_argument("--source", required=True)
    gs.add_argument("--resolution", default="medium", choices=["low", "medium", "high"])
    gs.add_argument("--fps", type=int, default=15, choices=[2, 7, 15, 24, 30])
    gs.add_argument("--jpeg-quality", type=int, default=60)
    gs.add_argument("--start", type=float, default=0.0)
    gs.add_argument("--seconds", type=float, default=10.0)
    gs.add_argument("--weights", default="models/yolo11s.pt")
    gs.add_argument("--backend", default="openvino", choices=["openvino", "torch"])
    gs.add_argument("--no-surface", action="store_true")
    gs.add_argument("--stats", help="write stats JSON here")
    gs.set_defaults(fn=cmd_glasses_sim)
    from .maps.demo import add_parser as add_walk_sim

    add_walk_sim(sub)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
