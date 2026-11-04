#!/usr/bin/env python3
"""Same-host perception benchmark. No live API calls; use before/after labels.

python scripts/benchmark_review.py --label before --frames 60 --repeats 2
Compare matching clips, resolution, hardware, dependencies and model hashes.
Continuity is a box-IoU proxy, not ground-truth IDF1 or detection accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streetsense.config import Config
from streetsense.geometry import iou
from streetsense.pipeline import Pipeline


def distribution(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean_ms": round(statistics.mean(values), 4),
            "p50_ms": round(float(np.percentile(values, 50)), 4),
            "p95_ms": round(float(np.percentile(values, 95)), 4)}


def load(path, start, count):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames = []
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != 1280:
            scale = 1280 / frame.shape[1]
            frame = cv2.resize(frame, (1280, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if len(frames) < count:
        raise ValueError(f"{path}: need {count} frames; got {len(frames)}")
    return frames, fps


def measure(frames, fps, start, backend, track_conf=None):
    now = [start]
    cfg = Config()
    cfg.vlm.enabled = False
    cfg.detector.backend = cfg.surface.backend = backend
    cfg.detector.track_conf = track_conf
    pipe = Pipeline(cfg, clock=lambda: now[0])
    if pipe.surface_seg is None:
        raise RuntimeError('SegFormer failed to initialize; refusing a misleading pipeline benchmark')
    if pipe.detector.backend != backend or pipe.surface_seg.backend != backend:
        raise RuntimeError('Requested inference backend was not used')
    lats, stages = [], defaultdict(list)
    counts, states, alerts = Counter(), Counter(), []
    previous, breaks, pairs = [], 0, 0
    for index, frame in enumerate(frames):
        now[0] = start + index / fps
        t0 = time.perf_counter()
        out = pipe.process(frame, annotate=False)
        elapsed = (time.perf_counter() - t0) * 1000
        if index < 5:  # exclude model initialization and early tracker confirmation
            continue
        lats.append(elapsed)
        for k, v in out.timings_ms.items():
            stages[k].append(v)
        for d in out.detections:
            counts[d.label] += 1
            if d.label == 'traffic light':
                states[d.attrs.get('light_state', 'unknown')] += 1
            candidates = [p for p in previous if p.label == d.label and iou(p.box, d.box) >= .5]
            if candidates and d.track_id is not None:
                p = max(candidates, key=lambda p: iou(p.box, d.box))
                if p.track_id is not None:
                    pairs += 1
                    breaks += p.track_id != d.track_id
        previous = out.detections
        if out.spoken:
            alerts.append({'t': round(out.t, 3), 'text': out.spoken.text})
    return {'pipeline': distribution(lats), 'fps': round(1000 / statistics.mean(lats), 3),
            'stages': {k: distribution(v) for k, v in stages.items()},
            'detection_counts': dict(counts), 'light_states': dict(states), 'alerts': alerts,
            'continuity_proxy': {'matched_pairs': pairs, 'id_changes': breaks},
            'backends': {'detector': pipe.detector.backend, 'surface': pipe.surface_seg.backend}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--backend', choices=['openvino', 'torch'], default='openvino')
    ap.add_argument('--track-conf', type=float, default=None, help='Optional low-score association experiment, e.g. .10')
    args = ap.parse_args()
    os.environ['STREETSENSE_SAM_FALLBACK'] = '0'
    rows = []
    clips = [('dc_red', 'dc_ped_light.ogv', 1.0), ('dc_walk', 'dc_ped_light.ogv', 10.0),
             ('tyumen', 'tyumen_crossing.webm', 8.0)]
    for name, filename, start in clips:
        frames, fps = load(ROOT / 'samples' / filename, start, args.frames + 5)
        for repeat in range(args.repeats):
            row = {'clip': name, 'repeat': repeat, 'source_fps': fps,
                   **measure(frames, fps, start, args.backend, args.track_conf)}
            rows.append(row)
            print(json.dumps({'clip': name, 'repeat': repeat, 'fps': row['fps'],
                              'pipeline': row['pipeline']}), flush=True)
    versions = {}
    for name in ['ultralytics', 'torch', 'openvino', 'numpy', 'transformers']:
        versions[name] = importlib.metadata.version(name)
    result = {'label': args.label, 'host': platform.platform(), 'python': platform.python_version(),
              'versions': versions, 'resolution_width': 1280, 'annotate': False,
              'track_conf': args.track_conf,
              'model_sha256': hashlib.sha256((ROOT / 'models/yolo11s.pt').read_bytes()).hexdigest(),
              'api_calls': 0, 'rows': rows}
    target = ROOT / 'outputs/review' / f'{args.label}-performance.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(target, flush=True)


if __name__ == '__main__':
    main()
