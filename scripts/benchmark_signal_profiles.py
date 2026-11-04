#!/usr/bin/env python3
"""CPU cost of automatic vs GB figure profile at 1280px, without surface/remote models."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from benchmark_review import distribution, load
from streetsense.config import Config
from streetsense.detector import YoloDetector
from streetsense.pipeline import Pipeline
from eval_local_reader import eval_configuration
from streetsense import signal_cnn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--resume', action='store_true', help='reuse completed rows in the output file')
    ap.add_argument('--clock', choices=['video', 'live'], default='live',
                    help='live uses monotonic time for the same refinement cadence as the web camera')
    args = ap.parse_args()
    os.environ['STREETSENSE_SAM_FALLBACK'] = '0'
    os.environ['STREETSENSE_LIGHT_TILES'] = '0'
    os.environ['STREETSENSE_LIGHT_TILES_EVERY'] = '0'
    os.environ['STREETSENSE_SIGNAL_CNN_VERSION'] = 'v1'
    if signal_cnn.get() is None:
        raise RuntimeError('CNN v1 is unavailable; refusing an incomplete reader benchmark')
    model = YoloDetector(Config().detector)
    rows = json.loads(args.out.read_text())['rows'] if args.resume and args.out.is_file() else []
    if any(r['pipeline']['n'] != args.frames for r in rows):
        ap.error('resumed rows must use the same frame count')
    if any(r.get('clock', 'video') != args.clock for r in rows):
        ap.error('resumed rows must use the same clock mode')
    completed = {(r['clip'], r['country'], r['repeat']) for r in rows}
    for filename, start in [('ext/pexels_30404309_eu_ped.mp4', 2.),
                            ('ext/pexels_5665107_eu_ped.mp4', 2.),
                            ('heldout/pexels_6415590.mp4', 4.5)]:
        if all((filename, country, repeat) in completed for country in ['auto', 'GB'] for repeat in range(2)):
            continue
        frames, fps = load(ROOT / 'samples' / filename, start, args.frames + 10)
        for repeat, countries in enumerate([('auto', 'GB'), ('GB', 'auto')]):
            for country in countries:
                if (filename, country, repeat) in completed:
                    continue
                cfg = Config()
                cfg.reader.country = country
                cfg.vlm.enabled = cfg.sam.enabled = False
                now = [start]
                model.reset_tracking()
                with patch('streetsense.pipeline.build_detector', lambda cfg: model):
                    clock = (lambda: now[0]) if args.clock == 'video' else time.monotonic
                    pipe = Pipeline(cfg, use_surface=False, clock=clock)
                latency, refine, lights = [], [], []
                for index, frame in enumerate(frames):
                    now[0] = start + index / fps
                    t0 = time.perf_counter()
                    out = pipe.process(frame, annotate=False)
                    elapsed = (time.perf_counter() - t0) * 1000
                    if index < 10:
                        continue
                    latency.append(elapsed)
                    refine.append(out.timings_ms.get('signal_refine', 0))
                    lights.append(out.timings_ms['lights'])
                row = {'clip': filename, 'start_s': start, 'country': country, 'repeat': repeat,
                       'clock': args.clock, 'source_fps': fps,
                       'pipeline': distribution(latency), 'refine': distribution(refine),
                       'lights': distribution(lights), 'fps': 1000 / statistics.mean(latency),
                       'refine_error': pipe.signal_refiner.error}
                rows.append(row)
                print(json.dumps(row), flush=True)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps({'width': 1280, 'surface': False, 'api_calls': 0,
                                               'configuration': eval_configuration(no_tiles=True),
                                               'backend': model.backend, 'rows': rows}, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'width': 1280, 'surface': False, 'api_calls': 0,
                                   'configuration': eval_configuration(no_tiles=True),
                                   'backend': model.backend, 'rows': rows}, indent=1))


if __name__ == '__main__':
    main()
