#!/usr/bin/env python3
"""Benchmark SegFormer surface: PyTorch vs OpenVINO (+ optional overlay), and full pipeline."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streetsense.config import Config
from streetsense.surface import (
    DEFAULT_OV_DIR,
    SurfaceSegmenter,
    ensure_openvino_segformer,
    summarise_labels,
)

CLIPS = [
    ("tyumen", "samples/tyumen_crossing.webm", None, None),
    ("dc", "samples/dc_ped_light.ogv", None, None),
    ("cdmx", "samples/cdmx_crossing_720p.webm", 20.0, 1200),
]


def pct(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize_ms(xs):
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean_ms": round(statistics.mean(xs), 2),
        "p50_ms": round(pct(xs, 50), 2),
        "p95_ms": round(pct(xs, 95), 2),
        "std_ms": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0,
    }


def load_frames(path, resize=1280, start_s=None, max_frames=None, stride=1):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    frames = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            if resize and fr.shape[1] > resize:
                s = resize / fr.shape[1]
                fr = cv2.resize(fr, (resize, int(fr.shape[0] * s)), interpolation=cv2.INTER_AREA)
            frames.append(fr)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames, fps


def miou(a, b):
    classes = np.unique(np.concatenate([a.ravel(), b.ravel()]))
    ious = []
    for c in classes:
        inter = np.logical_and(a == c, b == c).sum()
        union = np.logical_or(a == c, b == c).sum()
        if union:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 1.0


def time_segmenter(backend, frames, warmup=3):
    seg = SurfaceSegmenter(backend=backend)
    assert seg.backend == backend, f"expected {backend} got {seg.backend}"
    for fr in frames[:warmup]:
        seg(fr)
    lats = []
    results = []
    for fr in frames[warmup:]:
        t0 = time.perf_counter()
        r = seg(fr)
        lats.append((time.perf_counter() - t0) * 1000)
        results.append(r)
    return seg, summarize_ms(lats), results


def run_pipeline(det_backend, surf_backend, clip_path, start_s=None, max_frames=None,
                 annotate=True, max_proc=200):
    from streetsense.pipeline import Pipeline

    class Clock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

    cap = cv2.VideoCapture(clip_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    cfg = Config()
    cfg.detector.backend = det_backend
    cfg.surface.backend = surf_backend
    cfg.vlm.enabled = False
    clock = Clock()
    pipe = Pipeline(cfg, use_surface=True, clock=clock)
    warm = n = 0
    proc = 0.0
    raw = 0
    surface_call_ms = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw += 1
        if frame.shape[1] > 1280:
            s = 1280 / frame.shape[1]
            frame = cv2.resize(frame, (1280, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        clock.t = (start_s or 0.0) + (raw - 1) / fps_in
        t0 = time.perf_counter()
        out = pipe.process(frame, annotate=annotate)
        dt = time.perf_counter() - t0
        if warm < 5:
            warm += 1
            continue
        proc += dt
        n += 1
        # surface stage is 0 when reused; collect only when it ran
        if out.timings_ms.get("surface", 0) > 1.0:
            surface_call_ms.append(out.timings_ms["surface"])
        if max_proc and n >= max_proc:
            break
    cap.release()
    every = cfg.surface.every_n_frames
    return {
        "frames": n,
        "pipeline_fps": round(n / proc, 2) if proc else None,
        "mean_stage_ms": {k: round(v / max(1, n + warm), 1) for k, v in pipe.timing_totals.items()},
        "surface_when_runs": summarize_ms(surface_call_ms),
        "surface_amortised_ms": round(
            (sum(surface_call_ms) / max(1, n)) if surface_call_ms else 0.0, 2
        ),
        "detect_backend": pipe.detector.backend,
        "surface_backend": pipe.surface_seg.backend if pipe.surface_seg else None,
        "every_n": every,
    }


def run_alerts(det_backend, surf_backend, out_jsonl: Path):
    from streetsense.pipeline import Pipeline

    class Clock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

    cap = cv2.VideoCapture("samples/dc_ped_light.ogv")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cfg = Config()
    cfg.detector.backend = det_backend
    cfg.surface.backend = surf_backend
    cfg.vlm.enabled = False
    clock = Clock()
    pipe = Pipeline(cfg, use_surface=True, clock=clock)
    alerts = []
    raw = 0
    underfoot_hist = []
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw += 1
            if frame.shape[1] > 1280:
                s = 1280 / frame.shape[1]
                frame = cv2.resize(frame, (1280, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
            clock.t = (raw - 1) / fps_in
            out = pipe.process(frame, annotate=False)
            if out.surface:
                underfoot_hist.append(out.surface.underfoot)
            if out.spoken:
                rec = {
                    "t_video_s": round(out.t, 2),
                    "key": out.spoken.key,
                    "text": out.spoken.text,
                    "priority": out.spoken.priority.name,
                }
                alerts.append(rec)
                f.write(json.dumps(rec) + "\n")
    cap.release()
    from collections import Counter

    return alerts, dict(Counter(underfoot_hist))


def web_frame_latency(backend, n=40, warmup=5):
    from streetsense.web import create_app

    app = create_app(no_surface=False, backend=backend)
    client = app.test_client()
    cap = cv2.VideoCapture("samples/tyumen_crossing.webm")
    ok, fr = cap.read()
    cap.release()
    if fr.shape[1] > 640:
        s = 640 / fr.shape[1]
        fr = cv2.resize(fr, (640, int(fr.shape[0] * s)))
    ok, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    body = buf.tobytes()
    times = []
    server_ms = []
    for i in range(n + warmup):
        t0 = time.perf_counter()
        r = client.post("/frame", data=body, content_type="application/octet-stream")
        dt = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200
        j = r.get_json()
        if i >= warmup:
            times.append(dt)
            server_ms.append(j["server_ms"])
    return {"client": summarize_ms(times), "server_ms": summarize_ms(server_ms), "backend": backend}


def main():
    out = Path("outputs/benchmark_surface")
    out.mkdir(parents=True, exist_ok=True)
    results = {"export": {}, "segmenter": {}, "agreement": {}, "pipeline": {}, "alerts": {}, "web": {}}

    # export timing (scratch)
    import shutil

    scratch = out / "ov_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    t0 = time.perf_counter()
    ensure_openvino_segformer(cache_dir=scratch)
    results["export"]["first_run_convert_s"] = round(time.perf_counter() - t0, 2)
    results["export"]["default_cache_present"] = (DEFAULT_OV_DIR / "segformer.xml").is_file()
    print("export", results["export"])
    shutil.rmtree(scratch, ignore_errors=True)

    # segmenter timing + agreement on sampled frames from all clips
    sample_frames = []
    for name, path, start, max_f in CLIPS:
        frs, _ = load_frames(path, start_s=start, max_frames=40, stride=5)
        sample_frames.extend(frs[:12])
    print(f"segmenter sample frames: {len(sample_frames)}")

    for backend in ("torch", "openvino"):
        print(f"  segmenter {backend} ...")
        seg, lat, res = time_segmenter(backend, sample_frames, warmup=2)
        results["segmenter"][backend] = {
            "latency": lat,
            "underfoot": [r.underfoot for r in res],
            "backend": seg.backend,
        }
        print("   ", lat)

    # agreement torch vs OV on same frames (re-run both on identical list without warm cut mismatch)
    torch_seg = SurfaceSegmenter(backend="torch")
    ov_seg = SurfaceSegmenter(backend="openvino")
    agrees, mious, under_match = [], [], 0
    for fr in sample_frames:
        a = torch_seg(fr)
        b = ov_seg(fr)
        agrees.append(float((a.labels == b.labels).mean()))
        mious.append(miou(a.labels, b.labels))
        under_match += int(a.underfoot == b.underfoot)
    results["agreement"] = {
        "n_frames": len(sample_frames),
        "pixel_agreement_mean": round(float(np.mean(agrees)), 4),
        "pixel_agreement_min": round(float(np.min(agrees)), 4),
        "miou_mean": round(float(np.mean(mious)), 4),
        "underfoot_match_frac": round(under_match / len(sample_frames), 4),
    }
    print("agreement", results["agreement"])

    # full pipeline: before = torch surface + ov detect (previous default after detector switch was ov detect + torch surface)
    # after = ov detect + ov surface (new default)
    print("== pipeline ==")
    combos = [
        ("torch_surface", "openvino", "torch"),
        ("openvino_surface", "openvino", "openvino"),
    ]
    for label, det_b, surf_b in combos:
        results["pipeline"][label] = {}
        for clip_name, path, start, max_f in CLIPS:
            for annotate in (True, False):
                key = f"{clip_name}_overlay_{'on' if annotate else 'off'}"
                print(f"  {label} {key}")
                r = run_pipeline(det_b, surf_b, path, start_s=start, max_frames=max_f,
                                 annotate=annotate, max_proc=200)
                results["pipeline"][label][key] = r
                print("   ", r["pipeline_fps"], r["mean_stage_ms"], "surf_run", r["surface_when_runs"])

    print("== DC alerts ==")
    for label, det_b, surf_b in combos:
        alerts, under = run_alerts(det_b, surf_b, out / f"dc_alerts_{label}.jsonl")
        results["alerts"][label] = {
            "count": len(alerts),
            "events": alerts,
            "underfoot": under,
            "has_walk_10_8": any(abs(a["t_video_s"] - 10.8) < 0.4 and "walk" in a["text"].lower() for a in alerts),
            "has_red_52_8": any(
                abs(a["t_video_s"] - 52.8) < 0.5 and "red" in a["text"].lower() and "pedestrian" in a["text"].lower()
                for a in alerts
            ),
            "road_or_sidewalk_alerts": [a for a in alerts if "road" in a["text"].lower() or "sidewalk" in a["text"].lower() or "underfoot" in a["key"]],
        }
        print(label, results["alerts"][label]["count"],
              "walk", results["alerts"][label]["has_walk_10_8"],
              "red", results["alerts"][label]["has_red_52_8"],
              "underfoot", under)

    print("== web ==")
    for backend in ("torch", "openvino"):
        # backend sets both detect + surface
        results["web"][backend] = web_frame_latency(backend)
        print(backend, results["web"][backend])

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("Wrote", out / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
