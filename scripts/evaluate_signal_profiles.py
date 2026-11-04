#!/usr/bin/env python3
"""Paired local reader check on fixed detector boxes; no model API calls.

python scripts/evaluate_signal_profiles.py --out outputs/reader-uk-europe/profiles.json
Optionally reuse a previously saved --capture (row, frames with t/shape/heads, local).
The automatic replay must reproduce every captured decision before judging profiles.
Missing samples are explicit, never counted as correct. This is a development set,
not a new independent test set or live navigation validation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import eval_local_reader as ev
from streetsense.config import Config
from streetsense.detector import YoloDetector
from streetsense.pipeline import Pipeline
from streetsense.signal_trace import summarise_signals
from streetsense.types import Box, Detection

# Layout assignment by clip, independent of frame labels. Pexels and Basel are
# figure-style samples; no country/location is inferred from their filenames.
FIGURE_CLIPS = {'eu_ped_a', 'eu_ped_b', 'eu_ped_side', 'most11', 'most02', 'most03', 'most04',
                'tirana', 'tomaszow', 'pex6415590', 'uk_green', 'uk_red', 'basel', 'ffm'}


class FixedDetector:
    def __init__(self, model):
        self.model = model
        self.current = []

    def __call__(self, frame):
        return deepcopy(self.current)

    def reset_tracking(self):
        pass

    def focused_lights(self, frame, lights, conf):
        return self.model.focused_lights(frame, lights, conf)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', type=Path)
    ap.add_argument('--allow-missing', action='store_true', help='record missing samples explicitly; default is to fail')
    ap.add_argument('--dense', action='store_true', help='also check available European non-walk intervals at 4 FPS')
    ap.add_argument('--dense-only', action='store_true', help='run only the European non-walk scan')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    os.environ['STREETSENSE_SAM_FALLBACK'] = '0'
    os.environ['STREETSENSE_LIGHT_TILES'] = '0'
    os.environ['STREETSENSE_LIGHT_TILES_EVERY'] = '0'
    os.environ['STREETSENSE_SIGNAL_CNN_VERSION'] = 'v1'
    os.environ['STREETSENSE_READER_YELLOW_PROMOTE'] = '0'
    if ev.signal_cnn.get() is None:
        raise RuntimeError('CNN v1 is unavailable; refusing an incomplete reader evaluation')
    source = ({'config': ev.eval_configuration(no_tiles=True), 'rows': [], 'missing': []} if args.dense_only else
              json.loads(args.capture.read_text()) if args.capture else capture_baseline(args.allow_missing, args.out))
    model = YoloDetector(Config().detector)
    detector = FixedDetector(model)
    methods = {}
    for name, refine in [('baseline', False), ('profile_only', False), ('profile_refine', True)]:
        cfg = Config()
        cfg.vlm.enabled = cfg.sam.enabled = False
        cfg.reader.country = 'auto'
        cfg.reader.refine_heads = refine
        with patch('streetsense.pipeline.build_detector', lambda cfg: detector):
            methods[name] = Pipeline(cfg, use_surface=False)
    rows, summary = [], {}
    for sequence in source['rows']:
        r = sequence['row']
        per = {}
        for name, pipe in methods.items():
            pipe._reset_perception_state()
            pipe._i = 0
            pipe._frame_shape = pipe._frame_ts = None
            pipe.signal_profile = 'green_figure' if name != 'baseline' and r['clip'] in FIGURE_CLIPS else 'auto'
            trace = []
            for sample in sequence['frames']:
                frame = ev.frame_at(r, sample['t'])
                detector.current = [Detection('traffic light', Box(*h['box']), h['conf'], h['track_id'],
                                               {'source': h['source']}) for h in sample['heads']]
                out = pipe.process(frame, annotate=False, t=sample['t'])
                facts = summarise_signals(out, frame.shape[1], frame.shape[0])
                facts['timings_ms'] = out.timings_ms
                trace.append(facts)
            per[name] = {'local': trace[-1]['local'], 'frames': trace}
        if per['baseline']['local'] != sequence['local']:
            raise RuntimeError(f"Replay differs from captured baseline on {r['id']}")
        rows.append({'id': r['id'], 'clip': r['clip'], 'split': r['split'], 'subset': r['subset'],
                     'label': r['label'], 'profile': 'green_figure' if r['clip'] in FIGURE_CLIPS else 'auto',
                     'methods': per})
        print(r['id'], r['label'], {k: v['local'] for k, v in per.items()}, flush=True)
        summary = {}
        for name in methods:
            agg = defaultdict(lambda: {'walk': 0, 'walk_n': 0, 'false_walk': 0, 'nonwalk_n': 0})
            for row in rows:
                for group in ['ALL', row['split'], f"{row['split']}/{row['subset']}", row['profile']]:
                    a = agg[group]
                    walk = row['methods'][name]['local'] == 'walk'
                    if row['label'] == 'walk':
                        a['walk_n'] += 1
                        a['walk'] += walk
                    else:
                        a['nonwalk_n'] += 1
                        a['false_walk'] += walk
            summary[name] = dict(agg)
        result = {'configuration': ev.eval_configuration(no_tiles=True),
                  'capture_configuration': source['config'], 'missing': source.get('missing', []),
                  'methods': {name: {'refine_heads': pipe.cfg.reader.refine_heads,
                              'refine_interval_s': pipe.cfg.reader.refine_interval_s,
                              'refine_min_width_fraction': pipe.cfg.reader.refine_min_width_fraction}
                              for name, pipe in methods.items()},
                  'figure_clips': sorted(FIGURE_CLIPS),
                  'note': 'Identical base detector inputs; refinement adds current-frame local candidates. No Meta calls.',
                  'summary': summary, 'rows': rows}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    if args.dense_only:
        result = {'configuration': ev.eval_configuration(no_tiles=True), 'rows': [], 'summary': {}, 'meta_calls': 0}
    if args.dense or args.dense_only:
        result['dense'] = dense_european_scan(model, detector, methods)
        args.out.write_text(json.dumps(result, indent=1))


def capture_baseline(allow_missing, target):
    """Capture automatic-mode inputs once; preserve exact scores and reset tracks per row."""
    cfg = Config()
    cfg.reader.country = 'auto'
    cfg.reader.refine_heads = False
    cfg.vlm.enabled = cfg.sam.enabled = False
    pipe = Pipeline(cfg, use_surface=False)
    ev.FRAME_DIR.mkdir(parents=True, exist_ok=True)
    data, missing = [], []
    for r in ev.load_rows():
        t = float(r['time_s'])
        still = bool(r.get('file')) and Path(r['file']).suffix.lower() in ev.STILL_EXT
        stamps = [round(t - .25 * k, 2) if not still else float(4 - k) for k in range(4, -1, -1)]
        stamps = [tt for tt in stamps if tt >= 0]
        try:
            frames = [(tt, ev.frame_at(r, tt)) for tt in stamps]
            if any(frame is None for _, frame in frames):
                raise FileNotFoundError(r['id'])
        except (FileNotFoundError, RuntimeError, ev.subprocess.CalledProcessError):
            if not allow_missing:
                raise
            missing.append(r['id'])
            continue
        pipe._reset_perception_state()
        pipe._i = 0
        pipe._frame_shape = pipe._frame_ts = None
        sequence = []
        for tt, frame in frames:
            out = pipe.process(frame, annotate=False, t=tt)
            heads = []
            for d in out.detections:
                if d.label == 'traffic light':
                    heads.append({'box': [d.box.x1, d.box.y1, d.box.x2, d.box.y2],
                                  'conf': float(d.conf), 'track_id': d.track_id,
                                  'source': d.attrs.get('source') or ('tile' if 'tile' in d.attrs else
                                            'zoom' if 'zoom' in d.attrs else 'local')})
            sequence.append({'t': tt, 'shape': list(frame.shape), 'heads': heads,
                             'local': ev.local_signal_state(out, frame.shape)})
        data.append({'row': r, 'frames': sequence, 'local': sequence[-1]['local']})
        print('capture', r['id'], sequence[-1]['local'], flush=True)
    source = {'config': ev.eval_configuration(no_tiles=True), 'rows': data, 'missing': missing}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_name(target.stem + '-capture.json').write_text(json.dumps(source, indent=1))
    return source


def dense_european_scan(model, detector, methods):
    spans = [(clip, start, end) for clip, start, end in ev.NON_WALK
             if clip in FIGURE_CLIPS or clip in {'h:pexels_6415590.mp4', 'h:ffm_vehicle.mp4'}]
    cfg = Config()
    cfg.reader.country = 'auto'
    cfg.vlm.enabled = cfg.sam.enabled = False
    with patch('streetsense.pipeline.build_detector', lambda cfg: model):
        base = Pipeline(cfg, use_surface=False)
    refined = methods['profile_refine']
    refined.signal_profile = 'green_figure'
    results = []
    for clip, start, end in spans:
        path = ev.HELDOUT_DIR / clip[2:] if clip.startswith('h:') else ROOT / ev.ev.CLIPS[clip]
        cap = ev.cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(path)
        fps = cap.get(ev.cv2.CAP_PROP_FPS) or 25.
        for pipe in [base, refined]:
            pipe._reset_perception_state()
            pipe._i = 0
            pipe._frame_shape = pipe._frame_ts = None
        index, next_t = 0, start
        rows = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stamp = index / fps
            index += 1
            if stamp < next_t - 1e-6:
                continue
            if stamp > end:
                break
            next_t += .25
            before = base.process(frame, annotate=False, t=stamp)
            detector.current = [Detection(d.label, d.box, d.conf, d.track_id,
                                {'source': d.attrs.get('source') or ('zoom' if 'zoom' in d.attrs else 'local')})
                                for d in before.detections if d.label == 'traffic light']
            after = refined.process(frame, annotate=False, t=stamp)
            rows.append({'t': stamp, 'baseline': ev.local_signal_state(before, frame.shape),
                         'refined': ev.local_signal_state(after, frame.shape)})
        cap.release()
        result = {'clip': clip, 'start': start, 'end': end, 'frames': len(rows),
                  'baseline_false_walk': sum(r['baseline'] == 'walk' for r in rows),
                  'refined_false_walk': sum(r['refined'] == 'walk' for r in rows), 'rows': rows}
        results.append(result)
        print('dense', clip, len(rows), result['baseline_false_walk'], result['refined_false_walk'], flush=True)
    return results


if __name__ == '__main__':
    main()
