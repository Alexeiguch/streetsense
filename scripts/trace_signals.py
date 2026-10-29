#!/usr/bin/env python3
"""Replay a clip into a per-frame local signal JSONL trace. No Meta calls.

Example: .venv/bin/python scripts/trace_signals.py samples/dc_ped_light.ogv \
    --out outputs/dc_trace.jsonl --crop-dir outputs/dc_crops
Only --crop-dir saves images, and only signal crops at local decision changes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streetsense.config import Config
from streetsense.pipeline import Pipeline
from streetsense.signal_trace import summarise_signals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crop-dir", type=Path)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--backend", choices=("openvino", "torch"), default="openvino")
    ap.add_argument("--reset-on-veto", action="store_true", help="test stricter walk reacquisition")
    args = ap.parse_args()
    if (not math.isfinite(args.fps) or args.fps <= 0 or not math.isfinite(args.start) or args.start < 0
            or (args.end is not None and (not math.isfinite(args.end) or args.end <= args.start))):
        ap.error("require fps > 0 and 0 <= start < end")
    # Explicitly disable every remote perception path, including env-controlled SAM.
    os.environ["STREETSENSE_SAM_FALLBACK"] = "0"
    if args.reset_on_veto:
        os.environ["STREETSENSE_SIGNAL_RESET_ON_VETO"] = "1"
    cfg = Config()
    cfg.vlm.enabled = cfg.sam.enabled = cfg.depth.enabled = cfg.open_vocab.enabled = False
    cfg.detector.backend = args.backend
    if args.backend == "torch":
        import torch
        torch.set_num_threads(1)
    clock = type("Clock", (), {"t": 0.0, "__call__": lambda self: self.t})()
    cap = cv2.VideoCapture(str(args.clip))
    if not cap.isOpened():
        ap.error(f"cannot open clip: {args.clip}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or args.fps > source_fps:
        cap.release()
        ap.error("trace fps must not exceed the clip's frame rate")
    pipe = Pipeline(cfg, use_surface=False, clock=clock)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.crop_dir:
        args.crop_dir.mkdir(parents=True, exist_ok=True)
    previous = None
    n = changes = 0
    next_t = args.start
    try:
        with args.out.open("w", buffering=1) as stream:
            source_i = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                t = source_i / source_fps
                source_i += 1
                if args.end is not None and t > args.end + 1e-6:
                    break
                if t < next_t - 1e-6:
                    continue
                next_t += 1 / args.fps
                if img.shape[1] != 1280:
                    scale = 1280 / img.shape[1]
                    img = cv2.resize(img, (1280, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                clock.t = t
                out = pipe.process(img, annotate=False, t=t)
                record = summarise_signals(out, img.shape[1], img.shape[0])
                record["transition"] = previous is not None and record["local"] != previous
                if record["transition"]:
                    changes += 1
                    if args.crop_dir:
                        for i, head in enumerate(record["heads"]):
                            x1, y1, x2, y2 = (int(v) for v in head["box"])
                            crop = img[max(0, y1):y2, max(0, x1):x2]
                            if crop.size:
                                name = f"{out.index:05d}_{t:.2f}_{i}_{head['state']}.png"
                                path = args.crop_dir / name
                                if not cv2.imwrite(str(path), crop):
                                    raise OSError(f"cannot save signal crop: {path}")
                                head["crop"] = str(path)
                stream.write(json.dumps(record) + "\n")
                previous = record["local"]
                n += 1
    finally:
        cap.release()
    if not n:
        ap.error("no frames sampled; check the requested interval")
    print(json.dumps({"frames": n, "transitions": changes, "trace": str(args.out), "meta_calls": 0}))


if __name__ == "__main__":
    main()
