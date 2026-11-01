#!/usr/bin/env python3
"""Serial wall-clock timing for baseline / one focused tile / six tiles, no Meta.

Run with other inference jobs stopped. This is a small host latency sample,
not a phone/glasses benchmark or an accuracy evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import eval_local_reader as ev
from experiment_light_detection import local_config
from streetsense.pipeline import Pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', type=int, default=30, help='measured frames per clip and method')
    ap.add_argument('--warmup', type=int, default=5)
    args = ap.parse_args()
    if args.frames < 1 or args.warmup < 1:
        ap.error('--frames and --warmup must be positive')
    os.environ.update({'STREETSENSE_SAM_FALLBACK': '0', 'STREETSENSE_LIGHT_TILES_EVERY': '0',
                       'STREETSENSE_LIGHT_ZOOM': '0.5,0.3', 'STREETSENSE_SIGNAL_CNN_VERSION': 'v1',
                       'STREETSENSE_SIGNAL_CNN': '1', 'STREETSENSE_SIGNAL_RESET_ON_VETO': '0'})
    # Same five cached source frames for each method, cycling for timing only.
    selected = {'dc': 13.8, 'pex6415590': 8.5, 'night_ped': 1.6}
    inputs = {}
    for clip, t in selected.items():
        row = next(r for r in ev.load_rows() if r['clip'] == clip and float(r['time_s']) == t)
        inputs[clip] = [ev.frame_at(row, round(t-.25*k, 2)) for k in range(4, -1, -1)]
    report = {'host': {'system': platform.system(), 'machine': platform.machine(), 'cpus': os.cpu_count()},
              'warmup_frames': args.warmup, 'measured_frames_per_clip': args.frames, 'methods': {}}
    for method in ('baseline', 'focused', 'tiles2x3'):
        os.environ.update({'STREETSENSE_LIGHT_TILES': '2x3' if method == 'tiles2x3' else '0',
                           'STREETSENSE_LIGHT_TILES_MODE': 'always',
                           'STREETSENSE_LIGHT_FOCUS': '1' if method == 'focused' else '0'})
        pipe = Pipeline(local_config(), use_surface=False)
        if pipe.detector.backend != 'openvino':
            raise RuntimeError('refusing a silent backend change')
        clips = {}
        for clip, frames in inputs.items():
            times, detection = [], []
            for i in range(args.warmup + args.frames):
                img = frames[i % len(frames)]
                start = time.perf_counter()
                out = pipe.process(img, annotate=False, t=i*.25)
                elapsed = (time.perf_counter()-start)*1000
                if i >= args.warmup:
                    times.append(elapsed)
                    detection.append(sum(v for k, v in out.timings_ms.items() if k.startswith('detect')))
            clips[clip] = {'shape': list(frames[0].shape), 'frames': len(times),
                           'pipeline_median_ms': float(np.median(times)),
                           'pipeline_p95_ms': float(np.percentile(times, 95)),
                           'detector_median_ms': float(np.median(detection)), 'pipeline_ms': times}
            print(method, clip, round(clips[clip]['pipeline_median_ms'], 1), flush=True)
        report['methods'][method] = {'configuration': ev.eval_configuration(), 'clips': clips}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
