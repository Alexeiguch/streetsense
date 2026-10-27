#!/usr/bin/env python3
"""Compare PyTorch vs OpenVINO backends for StreetSense YOLO11n on the sample clips."""

from __future__ import annotations

import json
import resource
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streetsense.config import Config, DetectorConfig
from streetsense.detector import ApproachTracker, YoloDetector, ensure_openvino_model, openvino_dir_for

CLIPS = [
    ("tyumen", "samples/tyumen_crossing.webm", None, None),
    ("dc", "samples/dc_ped_light.ogv", None, None),
    ("cdmx", "samples/cdmx_crossing_720p.webm", 20.0, 1200),
]


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


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


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter)


def match(fa, fb, thr=0.5):
    used = set()
    matched = 0
    for da in fa:
        best_j, best = None, 0.0
        for j, db in enumerate(fb):
            if j in used or da["label"] != db["label"]:
                continue
            v = iou(da["xyxy"], db["xyxy"])
            if v > best:
                best, best_j = v, j
        if best_j is not None and best >= thr:
            used.add(best_j)
            matched += 1
    return matched, len(fa), len(used), len(fb)


def load_frames(path, resize=1280, start_s=None, max_frames=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if resize and fr.shape[1] > resize:
            s = resize / fr.shape[1]
            fr = cv2.resize(fr, (resize, int(fr.shape[0] * s)), interpolation=cv2.INTER_AREA)
        frames.append(fr)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def run_detector(backend, frames, warmup=10):
    cfg = DetectorConfig(backend=backend, weights="models/yolo11n.pt")
    det = YoloDetector(cfg)
    assert det.backend == backend, f"expected {backend}, got {det.backend}"
    approach = ApproachTracker()
    w = min(warmup, len(frames))
    for i in range(w):
        det(frames[i])
    lats = []
    dets_per = []
    approaching_frames = 0
    ids_present = 0
    for i, fr in enumerate(frames[w:]):
        t0 = time.perf_counter()
        dets = det(fr)
        lats.append((time.perf_counter() - t0) * 1000)
        t = (w + i) / 30.0
        ap = approach.update(t, dets)
        if ap:
            approaching_frames += 1
        if any(d.track_id is not None for d in dets):
            ids_present += 1
        dets_per.append([
            {"label": d.label, "xyxy": [d.box.x1, d.box.y1, d.box.x2, d.box.y2], "tid": d.track_id}
            for d in dets
        ])
    return {
        "backend": det.backend,
        "model_path": det.model_path,
        "latency": summarize_ms(lats),
        "frames_with_track_ids": ids_present,
        "frames_measured": len(lats),
        "approaching_frames": approaching_frames,
        "dets": dets_per,
        "class_counts": dict(Counter(d["label"] for fr in dets_per for d in fr)),
        "peak_rss_mb": round(rss_mb(), 1),
    }


def run_pipeline(backend, clip_path, start_s=None, max_frames=None, annotate=True, max_proc=250):
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
    cfg.detector.backend = backend
    cfg.vlm.enabled = False
    clock = Clock()
    pipe = Pipeline(cfg, use_surface=True, clock=clock)
    assert pipe.detector.backend == backend
    warm = n = 0
    proc = 0.0
    raw = 0
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
        pipe.process(frame, annotate=annotate)
        dt = time.perf_counter() - t0
        if warm < 5:
            warm += 1
            continue
        proc += dt
        n += 1
        if max_frames and (n + warm) >= max_frames:
            break
        if max_proc and n >= max_proc:
            break
    cap.release()
    return {
        "frames": n,
        "pipeline_fps": round(n / proc, 2) if proc else None,
        "mean_stage_ms": {k: round(v / max(1, n + warm), 1) for k, v in pipe.timing_totals.items()},
        "backend": pipe.detector.backend,
    }


def run_alerts(backend, out_jsonl: Path):
    from streetsense.pipeline import Pipeline

    class Clock:
        def __init__(self):
            self.t = 0.0
        def __call__(self):
            return self.t

    cap = cv2.VideoCapture("samples/dc_ped_light.ogv")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cfg = Config()
    cfg.detector.backend = backend
    cfg.vlm.enabled = False
    clock = Clock()
    pipe = Pipeline(cfg, use_surface=True, clock=clock)
    alerts = []
    raw = 0
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
            if out.spoken:
                rec = {"t_video_s": round(out.t, 2), "key": out.spoken.key, "text": out.spoken.text,
                       "priority": out.spoken.priority.name}
                alerts.append(rec)
                f.write(json.dumps(rec) + "\n")
    cap.release()
    return alerts


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
        assert r.status_code == 200, r.data
        j = r.get_json()
        if i >= warmup:
            times.append(dt)
            server_ms.append(j["server_ms"])
    return {
        "backend": backend,
        "client_roundtrip": summarize_ms(times),
        "server_ms": summarize_ms(server_ms),
    }


def main():
    out = Path("outputs/benchmark_backends")
    out.mkdir(parents=True, exist_ok=True)
    results = {"export": {}, "clips": {}, "pipeline": {}, "alerts": {}, "web": {}}

    # First-run export timing (delete and re-export a temp copy to measure, without nuking production dir)
    pt = Path("models/yolo11n.pt")
    ov = openvino_dir_for(pt)
    results["export"]["already_present"] = ov.is_dir()
    if ov.is_dir():
        # time a re-export into a scratch dir by temporarily moving
        import shutil
        scratch_pt = out / "yolo11n_export_scratch.pt"
        shutil.copy2(pt, scratch_pt)
        scratch_ov = openvino_dir_for(scratch_pt)
        if scratch_ov.exists():
            shutil.rmtree(scratch_ov)
        t0 = time.perf_counter()
        ensure_openvino_model(scratch_pt, imgsz=640)
        results["export"]["first_run_export_s"] = round(time.perf_counter() - t0, 2)
        print("export", results["export"])
        shutil.rmtree(scratch_ov, ignore_errors=True)
        scratch_pt.unlink(missing_ok=True)
    else:
        t0 = time.perf_counter()
        ensure_openvino_model(pt, imgsz=640)
        results["export"]["first_run_export_s"] = round(time.perf_counter() - t0, 2)

    for clip_name, path, start, max_f in CLIPS:
        print(f"== {clip_name} ==")
        frames, fps = load_frames(path, start_s=start, max_frames=max_f)
        # Cap detector-only at 400 frames for wall clock (still plenty for stats)
        if len(frames) > 400:
            frames = frames[:400]
        print(f"  frames={len(frames)}")
        clip = {"n": len(frames), "fps_in": fps, "backends": {}}
        by_b = {}
        for backend in ("torch", "openvino"):
            print(f"  detector {backend} ...")
            r = run_detector(backend, frames)
            by_b[backend] = r
            clip["backends"][backend] = {
                "latency": r["latency"],
                "class_counts": r["class_counts"],
                "frames_with_track_ids": r["frames_with_track_ids"],
                "approaching_frames": r["approaching_frames"],
                "model_path": r["model_path"],
                "peak_rss_mb": r["peak_rss_mb"],
            }
            print(f"    {r['latency']} approaching_frames={r['approaching_frames']} ids={r['frames_with_track_ids']}")
        # agreement
        ma = na = mb = nb = 0
        for fa, fb in zip(by_b["torch"]["dets"], by_b["openvino"]["dets"]):
            a, na_, b, nb_ = match(fa, fb)
            ma += a; na += na_; mb += b; nb += nb_
        clip["agreement"] = {
            "iou_thr": 0.5,
            "torch_matched_frac": round(ma / na, 4) if na else None,
            "openvino_matched_frac": round(mb / nb, 4) if nb else None,
            "torch_dets": na,
            "openvino_dets": nb,
            "matched_pairs": ma,
        }
        print("  agreement", clip["agreement"])
        # drop heavy dets
        for b in clip["backends"].values():
            pass
        results["clips"][clip_name] = clip
        (out / "results_partial.json").write_text(json.dumps(results, indent=2))

    print("== pipeline FPS ==")
    for backend in ("torch", "openvino"):
        results["pipeline"][backend] = {}
        for clip_name, path, start, max_f in CLIPS:
            for annotate in (True, False):
                key = f"{clip_name}_overlay_{'on' if annotate else 'off'}"
                print(f"  {backend} {key}")
                results["pipeline"][backend][key] = run_pipeline(
                    backend, path, start_s=start, max_frames=max_f, annotate=annotate, max_proc=200
                )
                print("   ", results["pipeline"][backend][key])

    print("== DC alerts ==")
    for backend in ("torch", "openvino"):
        alerts = run_alerts(backend, out / f"dc_alerts_{backend}.jsonl")
        results["alerts"][backend] = {
            "count": len(alerts),
            "events": alerts,
            "has_walk_10_8": any(abs(a["t_video_s"] - 10.8) < 0.4 and "walk" in a["text"].lower() for a in alerts),
            "has_red_52_8": any(abs(a["t_video_s"] - 52.8) < 0.5 and "red" in a["text"].lower() and "pedestrian" in a["text"].lower() for a in alerts),
        }
        print(f"  {backend}: {len(alerts)} alerts walk@10.8={results['alerts'][backend]['has_walk_10_8']} "
              f"red@52.8={results['alerts'][backend]['has_red_52_8']}")

    print("== web /frame ==")
    for backend in ("torch", "openvino"):
        results["web"][backend] = web_frame_latency(backend)
        print(f"  {backend}", results["web"][backend])

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("Wrote", out / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
