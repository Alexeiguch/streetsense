"""Run from the repository root: python -m experiments.reconstruction --help."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import cv2
import numpy as np

from .geometry import (align_relative_depth, backproject, correspondence_metrics,
                       feature_matches, quality_gate)


def extract_frames(source: Path, times: list[float], out: Path):
    cap = cv2.VideoCapture(str(source))
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened() or fps <= 0:
        raise ValueError(f"Cannot open video: {source}")
    records, paths = [], []
    try:
        for i, t in enumerate(times):
            if not np.isfinite(t) or t < 0 or t * fps >= count:
                raise ValueError(f"Time {t} is outside video duration {count/fps:.3f}s")
            frame_index = min(round(t * fps), int(count)-1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {frame_index}")
            path = out / f"frame-{i:02d}.png"
            if not cv2.imwrite(str(path), frame):
                raise OSError(f"Cannot write {path}")
            records.append({"requested_s": t, "frame_index": frame_index,
                            "decoded_s": frame_index / fps,
                            "pixel_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
            paths.append(path)
    finally:
        cap.release()
    return paths, records


def evaluate(data, points=None):
    metrics = []
    masks = data["mask"]
    for a in range(len(data["rgb"]) - 1):
        b = a + 1
        uv_a, uv_b = feature_matches(data["rgb"][a], data["rgb"][b])
        row = correspondence_metrics(data["points"] if points is None else points,
                                     data["intrinsics"], data["cameras"], masks, a, b, uv_a, uv_b)
        row.update({"views": [a, b]})
        metrics.append(row)
    return metrics


def depth_baseline(data, backend="openvino"):
    # Reuse the production relative-depth reader without modifying its calibration or runtime.
    from streetsense.depth import DepthEstimator
    cold_start = time.perf_counter()
    estimator = DepthEstimator(backend=backend)
    load_s = time.perf_counter() - cold_start
    depths, scales, timings = [], [], []
    for rgb, reference, valid in zip(data["rgb"], data["depth"], data["mask"]):
        start = time.perf_counter()
        rel = estimator.infer(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        timings.append(time.perf_counter() - start)
        z, scale = align_relative_depth(rel, reference, valid)
        depths.append(z); scales.append(scale)
    points = np.stack([backproject(z, k, c) for z, k, c in zip(depths, data["intrinsics"], data["cameras"])])
    return points, {"method": "DepthAnythingV2 Small, per-view median scale aligned to MapAnything",
                    "backend": estimator.backend, "device": "cpu",
                    "pose_source": "MapAnything (shared, not independently estimated)",
                    "model_load_s": load_s,
                    "checkpoint": estimator.model_id,
                    "checkpoint_revision": getattr(getattr(estimator._torch_model, "config", None), "_commit_hash", None),
                    "cold_total_s": time.perf_counter() - cold_start,
                    "per_view_s": timings, "alignment_scales": scales,
                    "metrics": evaluate(data, points)}


def write_ply(path, data, max_points=40000):
    points, colors = data["points"][data["mask"]], data["rgb"][data["mask"]]
    ids = np.linspace(0, len(points)-1, min(len(points), max_points), dtype=int)
    with path.open("w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(ids)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points[ids], colors[ids]):
            f.write(" ".join([*(f"{v:.5f}" for v in p), *(str(v) for v in c)]) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Offline 3D experiment; never feeds navigation or crossing decisions.")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--times", default="0.2,0.6,1.0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--baseline-backend", choices=["openvino", "torch"], default="openvino")
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=1,
                        help="Inference repeats with a resident model; final prediction is exported")
    parser.add_argument("--reuse", action="store_true", help="Re-evaluate existing reconstruction.npz without loading Meta model")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        data = dict(np.load(args.out / "reconstruction.npz", allow_pickle=False))
        report = json.loads((args.out / "report.json").read_text())
    else:
        if args.source is None:
            parser.error("--source is required unless --reuse")
        times = [float(t) for t in args.times.split(",")]
        if not 2 <= len(times) <= 8 or times != sorted(times) or len(set(times)) != len(times):
            parser.error("Provide 2–8 distinct increasing frame times")
        if not 196 <= args.size <= 518 or args.size % 14:
            parser.error("--size must be a multiple of 14 between 196 and 518")
        paths, records = extract_frames(args.source, times, args.out)
        from .backend import reconstruct
        print(f"Reconstructing {len(paths)} views", flush=True)
        data, perf = reconstruct(paths, args.device, args.size, args.repeats)
        np.savez_compressed(args.out / "reconstruction.npz", **data)
        report = {"schema_version": 1, "source": str(args.source),
                  "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  "frames": records, "performance": perf,
                  "platform": platform.platform(), "opencv": cv2.__version__,
                  "scale": "model-predicted metres, not measured or validated",
                  "moving_objects_masked": False, "ground_truth_available": False}
    report["multiview_metrics"] = evaluate(data)
    report["gate"] = quality_gate(report["multiview_metrics"], [r["pixel_sha256"] for r in report["frames"]])
    report["valid_pixels"] = [int(m.sum()) for m in data["mask"]]
    # Preserve successful reconstruction and provenance if an optional baseline fails.
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.baseline:
        print("Checking existing relative-depth baseline", flush=True)
        baseline_points, report["depth_baseline"] = depth_baseline(data, args.baseline_backend)
        np.savez_compressed(args.out / "baseline.npz", points=baseline_points)
    # macOS reports bytes, Linux reports KiB. Unified GPU allocations are separate above.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    if args.reuse:
        report["reevaluation_process_peak_rss_bytes"] = rss
    else:
        report["process_peak_rss_bytes"] = rss
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_ply(args.out / "cloud.ply", data)
    from .viewer import write_viewer
    write_viewer(args.out / "demo.html", data, report)
    print(json.dumps({"output": str(args.out), "gate": report["gate"],
                      "inference_s": report["performance"]["inference_s"],
                      "metrics": report["multiview_metrics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
