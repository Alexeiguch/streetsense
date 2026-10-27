#!/usr/bin/env python3
"""Benchmark YOLO11n vs YOLO26n for StreetSense (CPU, same clips, same settings).

Writes JSON + markdown-friendly summaries under outputs/benchmark/.
Never prints env secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLIPS = [
    ("tyumen", "samples/tyumen_crossing.webm", None, None),  # name, path, start_s, max_frames
    ("dc", "samples/dc_ped_light.ogv", None, None),
    ("cdmx", "samples/cdmx_crossing_720p.webm", 20.0, 1200),  # match README window
]

CARE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "traffic light", "stop sign", "bench", "dog", "chair",
}

# Frames to extract for hand labelling (clip, video_time_s) — diverse moments
HAND_FRAMES = [
    ("tyumen", 2.0), ("tyumen", 8.0), ("tyumen", 15.0), ("tyumen", 22.0),
    ("dc", 5.0), ("dc", 10.8), ("dc", 20.0), ("dc", 35.0), ("dc", 52.8), ("dc", 60.0),
    ("cdmx", 25.0), ("cdmx", 35.0), ("cdmx", 45.0), ("cdmx", 55.0), ("cdmx", 65.0),
]


def rss_mb() -> float:
    # ru_maxrss is KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize_ms(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean_ms": round(statistics.mean(xs), 2),
        "p50_ms": round(pct(xs, 50), 2),
        "p95_ms": round(pct(xs, 95), 2),
        "std_ms": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0,
        "min_ms": round(min(xs), 2),
        "max_ms": round(max(xs), 2),
    }


def load_frames(path: str, resize: int = 1280, start_s: float | None = None, max_frames: int | None = None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if resize and frame.shape[1] > resize:
            s = resize / frame.shape[1]
            frame = cv2.resize(frame, (resize, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_dets(dets_a, dets_b, iou_thr=0.5):
    """Greedy class-aware matching. Returns (#matched_a, #a, #matched_b, #b)."""
    used_b = set()
    matched_a = 0
    for da in dets_a:
        best_j, best_iou = None, 0.0
        for j, db in enumerate(dets_b):
            if j in used_b or da["label"] != db["label"]:
                continue
            i = iou_xyxy(da["xyxy"], db["xyxy"])
            if i > best_iou:
                best_iou, best_j = i, j
        if best_j is not None and best_iou >= iou_thr:
            used_b.add(best_j)
            matched_a += 1
    matched_b = len(used_b)
    return matched_a, len(dets_a), matched_b, len(dets_b)


def run_detector_only(weights: str, frames: list, imgsz=640, conf=0.35, iou=0.5,
                      warmup=10, track=True, repeats=2) -> dict:
    from ultralytics import YOLO

    t_load0 = time.perf_counter()
    model = YOLO(weights)
    load_s = time.perf_counter() - t_load0
    # warmup
    w = min(warmup, len(frames))
    for i in range(w):
        if track:
            model.track(frames[i], persist=True, tracker="bytetrack.yaml",
                        imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False)
        else:
            model.predict(frames[i], imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False)

    all_lat = []
    per_repeat = []
    for r in range(repeats):
        # fresh tracker state
        if track:
            model = YOLO(weights)
            for i in range(w):
                model.track(frames[i], persist=True, tracker="bytetrack.yaml",
                            imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False)
        lat = []
        for fr in frames[w:]:
            t0 = time.perf_counter()
            if track:
                model.track(fr, persist=True, tracker="bytetrack.yaml",
                            imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False)
            else:
                model.predict(fr, imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False)
            lat.append((time.perf_counter() - t0) * 1000)
        all_lat.extend(lat)
        per_repeat.append(summarize_ms(lat))
    size_mb = Path(weights).stat().st_size / (1024 * 1024)
    return {
        "weights": weights,
        "load_s": round(load_s, 3),
        "weights_mb": round(size_mb, 2),
        "peak_rss_mb": round(rss_mb(), 1),
        "warmup_frames": w,
        "measured_frames_per_repeat": len(frames) - w,
        "repeats": repeats,
        "latency": summarize_ms(all_lat),
        "per_repeat": per_repeat,
        "track": track,
    }


def collect_detections(weights: str, frames: list, imgsz=640, conf=0.35, iou=0.5, track=True):
    from ultralytics import YOLO
    from streetsense.detector import KEEP

    model = YOLO(weights)
    names = model.names
    per_frame = []
    class_counts = Counter()
    for i, fr in enumerate(frames):
        if track:
            res = model.track(fr, persist=True, tracker="bytetrack.yaml",
                              imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False)[0]
        else:
            res = model.predict(fr, imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for b, c, p in zip(xyxy, cls, confs):
                label = names[int(c)]
                if label not in KEEP:
                    continue
                d = {"label": label, "xyxy": [float(x) for x in b], "conf": float(p)}
                dets.append(d)
                class_counts[label] += 1
        per_frame.append(dets)
    return per_frame, dict(class_counts)


def run_pipeline_fps(weights: str, clip_path: str, start_s=None, max_frames=None,
                     resize=1280, imgsz=640, annotate=True, use_surface=True, max_proc=None):
    """Full StreetSense pipeline FPS (no VLM)."""
    from streetsense.config import Config
    from streetsense.pipeline import Pipeline

    class VideoClock:
        def __init__(self):
            self.t = 0.0
        def __call__(self):
            return self.t

    cap = cv2.VideoCapture(clip_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    cfg = Config()
    cfg.detector.weights = weights
    cfg.detector.imgsz = imgsz
    cfg.vlm.enabled = False
    clock = VideoClock()
    pipe = Pipeline(cfg, use_surface=use_surface, clock=clock)
    # warmup 5 frames
    warm = 0
    n = 0
    proc = 0.0
    raw_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_idx += 1
        if resize and frame.shape[1] > resize:
            s = resize / frame.shape[1]
            frame = cv2.resize(frame, (resize, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        clock.t = (start_s or 0.0) + (raw_idx - 1) / fps_in
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
        "annotate": annotate,
        "surface": use_surface,
        "peak_rss_mb": round(rss_mb(), 1),
    }


def run_alerts(weights: str, clip_path: str, out_jsonl: Path, start_s=None, max_frames=None,
               resize=1280, imgsz=640):
    from streetsense.config import Config
    from streetsense.pipeline import Pipeline

    class VideoClock:
        def __init__(self):
            self.t = 0.0
        def __call__(self):
            return self.t

    cap = cv2.VideoCapture(clip_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    cfg = Config()
    cfg.detector.weights = weights
    cfg.detector.imgsz = imgsz
    cfg.vlm.enabled = False
    clock = VideoClock()
    pipe = Pipeline(cfg, use_surface=True, clock=clock)
    alerts = []
    raw_idx = 0
    n = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_idx += 1
            if resize and frame.shape[1] > resize:
                s = resize / frame.shape[1]
                frame = cv2.resize(frame, (resize, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
            clock.t = (start_s or 0.0) + (raw_idx - 1) / fps_in
            out = pipe.process(frame, annotate=False)
            if out.spoken:
                rec = {
                    "t_video_s": round(out.t, 2),
                    "frame": raw_idx - 1,
                    "priority": out.spoken.priority.name,
                    "key": out.spoken.key,
                    "text": out.spoken.text,
                }
                alerts.append(rec)
                f.write(json.dumps(rec) + "\n")
            n += 1
            if max_frames and n >= max_frames:
                break
    cap.release()
    return alerts


def try_export_bench(weights: str, frames: list, fmt: str, imgsz=640, warmup=5) -> dict | None:
    """Export to ONNX or OpenVINO and time predict on a subset of frames."""
    from ultralytics import YOLO
    try:
        model = YOLO(weights)
        t0 = time.perf_counter()
        exported = model.export(format=fmt, imgsz=imgsz, device="cpu", simplify=True)
        export_s = time.perf_counter() - t0
        exp_path = Path(exported) if not isinstance(exported, Path) else exported
        emodel = YOLO(str(exp_path))
        subset = frames[: min(80, len(frames))]
        for fr in subset[:warmup]:
            emodel.predict(fr, imgsz=imgsz, device="cpu", verbose=False)
        lat = []
        for fr in subset[warmup:]:
            t0 = time.perf_counter()
            emodel.predict(fr, imgsz=imgsz, device="cpu", verbose=False)
            lat.append((time.perf_counter() - t0) * 1000)
        size_mb = exp_path.stat().st_size / (1024 * 1024) if exp_path.is_file() else None
        if exp_path.is_dir():
            size_mb = sum(p.stat().st_size for p in exp_path.rglob("*") if p.is_file()) / (1024 * 1024)
        return {
            "format": fmt,
            "path": str(exp_path),
            "export_s": round(export_s, 2),
            "size_mb": round(size_mb, 2) if size_mb is not None else None,
            "latency": summarize_ms(lat),
            "note": "predict only (no ByteTrack); subset of frames",
        }
    except Exception as e:
        return {"format": fmt, "error": f"{type(e).__name__}: {e}"}


def coco128_val(weights: str) -> dict:
    from ultralytics import YOLO
    model = YOLO(weights)
    # coco128 auto-downloads; small subset, not full COCO
    t0 = time.perf_counter()
    metrics = model.val(data="coco128.yaml", imgsz=640, device="cpu", plots=False, verbose=False)
    elapsed = time.perf_counter() - t0
    box = metrics.box
    return {
        "dataset": "coco128 (Ultralytics auto-download; 128-image COCO subset, NOT full COCO val)",
        "elapsed_s": round(elapsed, 1),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
    }


def save_comparison_png(frame, dets_a, dets_b, path: Path, title_a="YOLO11n", title_b="YOLO26n"):
    h, w = frame.shape[:2]
    left = frame.copy()
    right = frame.copy()
    colors = {
        "person": (0, 200, 0), "car": (0, 140, 255), "bus": (0, 100, 200),
        "truck": (0, 80, 180), "bicycle": (200, 200, 0), "motorcycle": (180, 160, 0),
        "traffic light": (0, 0, 255), "bench": (180, 100, 100),
    }

    def draw(img, dets, title):
        for d in dets:
            x1, y1, x2, y2 = map(int, d["xyxy"])
            col = colors.get(d["label"], (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f"{d['label']} {d['conf']:.2f}", (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    left = draw(left, dets_a, title_a)
    right = draw(right, dets_b, title_b)
    canvas = np.concatenate([left, right], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def extract_hand_frames(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    clip_map = {n: (p, s) for n, p, s, _ in CLIPS}
    for name, t_s in HAND_FRAMES:
        path, start = clip_map[name]
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # absolute video time
        cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        if frame.shape[1] > 1280:
            s = 1280 / frame.shape[1]
            frame = cv2.resize(frame, (1280, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        fname = f"{name}_t{t_s:05.1f}.jpg"
        fp = out_dir / fname
        cv2.imwrite(str(fp), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        meta.append({"clip": name, "t_s": t_s, "path": str(fp), "shape": list(frame.shape)})
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/benchmark")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--skip-coco", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument("--detector-max-frames", type=int, default=0,
                    help="limit frames per clip for detector-only (0=all)")
    ap.add_argument("--pipeline-max-frames", type=int, default=200,
                    help="limit frames for full-pipeline FPS (default 200 for speed)")
    ap.add_argument("--only", choices=["all", "speed", "acc", "alerts", "frames", "export", "coco"], default="all")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    weights = {
        "yolo11n": str(ROOT / "models" / "yolo11n.pt"),
        "yolo26n": str(ROOT / "models" / "yolo26n.pt"),
    }
    results = {"machine": {"cpu_count": os.cpu_count(), "note": "box CPU, no GPU"}, "weights": {}, "clips": {}}

    for name, w in weights.items():
        results["weights"][name] = {
            "path": w,
            "size_mb": round(Path(w).stat().st_size / (1024 * 1024), 2),
        }

    # --- extract hand frames early ---
    if args.only in ("all", "frames", "acc"):
        print("== extract hand frames ==")
        results["hand_frames"] = extract_hand_frames(out / "hand_frames")
        print(f"  wrote {len(results['hand_frames'])} frames")

    # --- detector-only speed + accuracy per clip ---
    if args.only in ("all", "speed", "acc"):
        for clip_name, clip_path, start_s, max_f in CLIPS:
            print(f"== load clip {clip_name} ==")
            frames, fps = load_frames(clip_path, start_s=start_s, max_frames=max_f)
            if args.detector_max_frames and len(frames) > args.detector_max_frames:
                frames = frames[: args.detector_max_frames]
            print(f"  {len(frames)} frames @ ~{fps:.1f} fps in")
            clip_res = {"n_frames": len(frames), "fps_in": fps, "start_s": start_s, "models": {}}
            dets_by_model = {}
            for mname, w in weights.items():
                print(f"  -- detector-only {mname} --")
                if args.only in ("all", "speed"):
                    speed = run_detector_only(w, frames, repeats=args.repeats)
                    print(f"     latency mean={speed['latency']['mean_ms']}ms p50={speed['latency']['p50_ms']} p95={speed['latency']['p95_ms']}")
                else:
                    speed = None
                print(f"  -- collect dets {mname} --")
                per_frame, counts = collect_detections(w, frames)
                dets_by_model[mname] = per_frame
                clip_res["models"][mname] = {
                    "speed": speed,
                    "class_counts": counts,
                    "total_dets": sum(counts.values()),
                }
            # agreement
            print("  -- agreement --")
            m_a, n_a, m_b, n_b = 0, 0, 0, 0
            for fa, fb in zip(dets_by_model["yolo11n"], dets_by_model["yolo26n"]):
                ma, na, mb, nb = match_dets(fa, fb)
                m_a += ma; n_a += na; m_b += mb; n_b += nb
            clip_res["agreement"] = {
                "iou_thr": 0.5,
                "yolo11n_matched_frac": round(m_a / n_a, 4) if n_a else None,
                "yolo26n_matched_frac": round(m_b / n_b, 4) if n_b else None,
                "yolo11n_dets": n_a,
                "yolo26n_dets": n_b,
                "matched_pairs": m_a,
            }
            print(f"     11n matched by 26n: {clip_res['agreement']['yolo11n_matched_frac']} "
                  f"| 26n matched by 11n: {clip_res['agreement']['yolo26n_matched_frac']}")

            # comparison PNGs at a mid frame
            mid = len(frames) // 2
            save_comparison_png(
                frames[mid],
                dets_by_model["yolo11n"][mid],
                dets_by_model["yolo26n"][mid],
                out / f"compare_{clip_name}_mid.png",
            )
            # also at ~1/4
            q = max(0, len(frames) // 4)
            save_comparison_png(
                frames[q],
                dets_by_model["yolo11n"][q],
                dets_by_model["yolo26n"][q],
                out / f"compare_{clip_name}_q1.png",
            )
            clip_res["compare_pngs"] = [
                str(out / f"compare_{clip_name}_mid.png"),
                str(out / f"compare_{clip_name}_q1.png"),
            ]
            # save dets for hand-frame evaluation later
            hand_dets = {}
            for hf in results.get("hand_frames", []):
                if hf["clip"] != clip_name:
                    continue
                # find nearest frame index by time
                # frames[0] starts at start_s
                t0 = start_s or 0.0
                idx = int(round((hf["t_s"] - t0) * fps))
                idx = max(0, min(len(frames) - 1, idx))
                hand_dets[hf["path"]] = {
                    "approx_idx": idx,
                    "yolo11n": dets_by_model["yolo11n"][idx],
                    "yolo26n": dets_by_model["yolo26n"][idx],
                }
            clip_res["hand_frame_dets"] = hand_dets
            results["clips"][clip_name] = clip_res
            # free memory
            del frames, dets_by_model
            (out / "results_partial.json").write_text(json.dumps(results, indent=2))

    # --- full pipeline FPS ---
    if args.only in ("all", "speed") and not args.skip_pipeline:
        print("== full pipeline FPS ==")
        results["pipeline"] = {}
        for mname, w in weights.items():
            results["pipeline"][mname] = {}
            for clip_name, clip_path, start_s, max_f in CLIPS:
                lim = args.pipeline_max_frames
                if max_f:
                    lim = min(lim, max_f) if lim else max_f
                for annotate in (True, False):
                    key = f"{clip_name}_overlay_{'on' if annotate else 'off'}"
                    print(f"  {mname} {key} ...")
                    r = run_pipeline_fps(w, clip_path, start_s=start_s, max_frames=max_f,
                                         annotate=annotate, use_surface=True, max_proc=lim)
                    results["pipeline"][mname][key] = r
                    print(f"    fps={r['pipeline_fps']} stages={r['mean_stage_ms']}")
            (out / "results_partial.json").write_text(json.dumps(results, indent=2))

    # --- alerts on DC clip ---
    if args.only in ("all", "alerts"):
        print("== DC alerts ==")
        results["alerts"] = {}
        for mname, w in weights.items():
            path = out / f"dc_alerts_{mname}.jsonl"
            print(f"  {mname} ...")
            alerts = run_alerts(w, "samples/dc_ped_light.ogv", path)
            results["alerts"][mname] = {
                "count": len(alerts),
                "path": str(path),
                "events": alerts,
                "light_related": [a for a in alerts if "light" in a["key"] or "light" in a["text"].lower()
                                  or "walk" in a["text"].lower() or "countdown" in a["text"].lower()],
            }
            print(f"    {len(alerts)} alerts; light-related={len(results['alerts'][mname]['light_related'])}")
        (out / "results_partial.json").write_text(json.dumps(results, indent=2))

    # --- coco128 ---
    if args.only in ("all", "coco") and not args.skip_coco:
        print("== coco128 val ==")
        results["coco128"] = {}
        for mname, w in weights.items():
            print(f"  {mname} ...")
            results["coco128"][mname] = coco128_val(w)
            print(f"    {results['coco128'][mname]}")
        (out / "results_partial.json").write_text(json.dumps(results, indent=2))

    # --- ONNX / OpenVINO ---
    if args.only in ("all", "export") and not args.skip_export:
        print("== export bench (tyumen subset) ==")
        frames, _ = load_frames("samples/tyumen_crossing.webm", max_frames=100)
        results["export"] = {}
        for mname, w in weights.items():
            results["export"][mname] = {}
            for fmt in ("onnx", "openvino"):
                print(f"  {mname} -> {fmt} ...")
                results["export"][mname][fmt] = try_export_bench(w, frames, fmt)
                print(f"    {results['export'][mname][fmt]}")
        del frames

    results["peak_rss_mb_end"] = round(rss_mb(), 1)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
