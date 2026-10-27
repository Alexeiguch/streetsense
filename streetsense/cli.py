"""Command line entry point.

  python -m streetsense run   --source samples/street.webm --out outputs/annotated.mp4 --alerts outputs/alerts.jsonl
  python -m streetsense run   --source 0 --speak            # webcam, spoken alerts
  python -m streetsense ask   --image frame.png --question "Can I cross now?"   # Muse Spark, needs MODEL_API_KEY
  python -m streetsense serve --port 8000                    # phone-camera web demo
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
    cfg.vlm.enabled = not a.no_vlm
    clock = VideoClock() if is_file else time.monotonic
    pipe = Pipeline(cfg, use_surface=not a.no_surface, clock=clock)
    speaker = Speaker(enabled=a.speak)
    print(f"[streetsense] source={a.source} fps_in={fps_in:.1f} surface={'on' if pipe.surface_seg else 'off'} "
          f"meta_api={'on' if pipe.vlm.enabled else 'off (no MODEL_API_KEY)'} speech={speaker.backend}")

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
            speaker.say(out.spoken.text)
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
    print(json.dumps({"latency_s": round(time.perf_counter() - t0, 2), **res}, indent=2))
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .web import create_app

    app = create_app(no_surface=a.no_surface)
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
    r.add_argument("--weights", default="models/yolo11n.pt",
                    help="Ultralytics weights path (default yolo11n; try models/yolo26n.pt)")
    r.add_argument("--no-surface", action="store_true")
    r.add_argument("--no-vlm", action="store_true")
    r.add_argument("--speak", action="store_true")
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("ask", help="one-shot Muse Spark question about an image")
    q.add_argument("--image", required=True)
    q.add_argument("--question", default="Is it safe to cross the street right now?")
    q.add_argument("--model", default="muse-spark-1.3")
    q.set_defaults(fn=cmd_ask)
    s = sub.add_parser("serve", help="web demo: phone camera -> alerts spoken in the browser")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-surface", action="store_true")
    s.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
