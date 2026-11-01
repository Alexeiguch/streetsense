#!/usr/bin/env python3
"""Paired, local-only detector experiments through the unchanged signal reader.

Exploratory evaluation, not training or a new production default. Common full-frame
and fallback detections are reused within each labelled sequence; every method gets
a fresh smoother. Reader pixels are always the original image, even for CLAHE.
No Meta requests. Missing assets fail unless --allow-missing is explicit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import eval_local_reader as ev
from streetsense.answers import local_signal_state
from streetsense.config import Config
from streetsense.detector import YoloDetector, _boxes, nms_lights, tile_windows, _iou
from streetsense import pipeline as sp
from streetsense.types import Box, Detection

METHODS = ('baseline', 'zoom_always', 'roi', 'clahe', 'saliency_tile', 'tiles1x2', 'tiles2x3')


def roi_windows(lights, width, height, limit=2):
    """At most two 2x context regions, clipped to the image, no guessed lamp crop."""
    windows = []
    for d in sorted(lights, key=lambda d: -d.conf):
        b = d.box
        cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
        w, h = max(128, b.w * 2), max(128, b.h * 2)
        r = (max(0, int(cx-w/2)), max(0, int(cy-h/2)),
             min(width, int(np.ceil(cx+w/2))), min(height, int(np.ceil(cy+h/2))))
        if r[2] > r[0] and r[3] > r[1] and r not in windows:
            windows.append(r)
        if len(windows) == limit:
            break
    return windows


class PairedDetector:
    def __init__(self, base):
        self.base = base
        self.backend, self.model_path = base.backend, base.model_path
        self.cache = {}
        self.method = 'baseline'
        self.index = 0
        self.frame_ms = 0.0
        self.frame_calls = 0

    def reset(self):
        self.cache.clear()
        predictor = getattr(self.base.model, 'predictor', None)
        for tracker in getattr(predictor, 'trackers', []):
            tracker.reset()

    def cached(self, key, fn):
        if key not in self.cache:
            start = time.perf_counter()
            result = fn()
            self.cache[key] = (deepcopy(result), (time.perf_counter()-start)*1000)
        result, ms = self.cache[key]
        if key not in self.charged:
            self.frame_ms += ms
            self.frame_calls += 1
            self.charged.add(key)
        return deepcopy(result)

    def predict_lights(self, img, source, offset=(0, 0)):
        if getattr(self.base, '_zoom_model', None) is None:
            from ultralytics import YOLO
            self.base._zoom_model = YOLO(self.model_path)
        c = self.base.cfg
        result = self.base._zoom_model.predict(img, imgsz=c.imgsz, conf=.5, iou=c.iou,
                                               device=c.device, verbose=False)[0]
        x, y = offset
        return [Detection('traffic light', Box(float(b[0])+x, float(b[1])+y,
                    float(b[2])+x, float(b[3])+y), float(p), None, {'source': source})
                for b, cls, p in _boxes(result)
                if self.base.names[int(cls)] == 'traffic light' and float(p) >= .5]

    def __call__(self, img):
        self.frame_ms = self.frame_calls = 0
        self.charged = set()
        dets = self.cached(('full', self.index), lambda: self.base(img))
        lights = [d for d in dets if d.label == 'traffic light']
        start = time.perf_counter()
        calls = 0
        extra = []
        if self.method == 'zoom_always':
            extra = self.zoomed_lights(img, self.base.cfg.light_zoom_scales,
                                        self.base.cfg.light_zoom_conf)
        elif self.method == 'roi':
            h, w = img.shape[:2]
            for x1, y1, x2, y2 in roi_windows(lights, w, h):
                extra += self.predict_lights(img[y1:y2, x1:x2], 'roi', (x1, y1))
                calls += 1
        elif self.method == 'clahe':
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            extra = self.predict_lights(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 'clahe')
            calls = 1
        elif self.method == 'saliency_tile' and lights:
            # Pick one ordinary tile by bright saturated pixels inside a found head.
            # Hue is deliberately ignored. This schedules a detector pass, not WALK.
            h, w = img.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            b = max(lights, key=lambda d: d.conf).box.clip(w, h)
            x1, y1, x2, y2 = map(int, (b.x1, b.y1, b.x2, b.y2))
            hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
            mask[y1:y2, x1:x2] = ((hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 120))
            windows = tile_windows(w, h, (2, 3))
            scores = [int(mask[y:y+th, x:x+tw].sum()) for x, y, tw, th in windows]
            if max(scores) > 0:
                x, y, tw, th = windows[int(np.argmax(scores))]
                extra = self.predict_lights(img[y:y+th, x:x+tw], 'saliency_tile', (x, y))
                # Keep original heads: a small candidate must not erase a red parent.
                extra = [d for d in extra if all(_iou(d.box, old.box) < .5 for old in lights)]
                calls = 1
        elif self.method.startswith('tiles'):
            grid = (1, 2) if self.method == 'tiles1x2' else (2, 3)
            extra = self.base.tiled_lights(img, grid, .5)
            calls = grid[0] * grid[1]
        if self.method != 'zoom_always':
            self.frame_ms += (time.perf_counter()-start)*1000
            self.frame_calls += calls
        if not extra:
            return dets
        if self.method == 'saliency_tile':
            return dets + nms_lights(extra)
        return [d for d in dets if d.label != 'traffic light'] + nms_lights(lights + extra)

    def zoomed_lights(self, img, scales, conf):
        if self.method.startswith('tiles'):
            return []  # Production always-tiles replaces the fallback zoom pass.
        return self.cached(('zoom', self.index, tuple(scales), conf),
                           lambda: self.base.zoomed_lights(img, scales, conf))


def local_config():
    c = Config()
    c.vlm.enabled = c.sam.enabled = c.depth.enabled = c.open_vocab.enabled = False
    return c


def run_sequence(detector, frames, methods):
    detector.reset()
    results = {}
    for method in methods:
        detector.method = method
        hist, trace, times, calls = [], [], [], []
        with patch.object(sp, 'build_detector', return_value=detector):
            pipe = sp.Pipeline(local_config(), use_surface=False)
        for i, (t, img) in enumerate(frames):
            detector.index = i
            out = pipe.process(img, annotate=False, t=t)
            hist.append(local_signal_state(out, img.shape))
            trace.append(ev.heads(out))
            times.append(detector.frame_ms)
            calls.append(detector.frame_calls)
        results[method] = {'state': hist[-1], 'history': hist, 'heads': trace[-1],
                           'trace': trace, 'detector_ms': times, 'detector_call_groups': calls}
    return results


def frames_for(row):
    t = float(row['time_s'])
    still = bool(row.get('file')) and Path(row['file']).suffix.lower() in ev.STILL_EXT
    result = []
    for k in range(4, -1, -1):
        tt = round(t-.25*k, 2) if not still else float(4-k)
        if tt < 0:
            continue
        img = ev.frame_at(row, tt)
        if img is None:
            raise FileNotFoundError(f"unreadable frame {row['id']} at {tt}")
        result.append((tt, img))
    return result


def aggregate(rows, methods):
    agg = {m: defaultdict(lambda: {'walk': 0, 'walk_total': 0, 'false_walk': 0, 'nonwalk_total': 0})
           for m in methods}
    for r in rows:
        for m in methods:
            for key in ('ALL', r['split']+'/ALL', r['split']+'/'+r['subset']):
                a = agg[m][key]
                if r['label'] == 'walk':
                    a['walk_total'] += 1
                    a['walk'] += r['methods'][m]['state'] == 'walk'
                else:
                    a['nonwalk_total'] += 1
                    a['false_walk'] += r['methods'][m]['state'] == 'walk'
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--clips', default='', help='comma-separated clip names; omitted means all labels')
    ap.add_argument('--allow-missing', action='store_true')
    ap.add_argument('--dense', action='store_true')
    ap.add_argument('--dense-methods', default='', help='subset of methods for the longer dense scan')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    methods = args.methods.split(',')
    if not methods or any(m not in METHODS for m in methods) or len(methods) != len(set(methods)):
        ap.error('choose distinct methods from '+','.join(METHODS))
    # Explicit settings avoid inherited tile/CNN experiments contaminating a baseline.
    for name, value in {'STREETSENSE_SAM_FALLBACK': '0', 'STREETSENSE_LIGHT_TILES': '0',
                        'STREETSENSE_LIGHT_TILES_EVERY': '0', 'STREETSENSE_LIGHT_ZOOM': '0.5,0.3',
                        'STREETSENSE_SIGNAL_CNN_VERSION': 'v1', 'STREETSENSE_SIGNAL_CNN': '1',
                        'STREETSENSE_SIGNAL_RESET_ON_VETO': '0'}.items():
        os.environ[name] = value
    dense_methods = args.dense_methods.split(',') if args.dense_methods else methods
    if any(m not in methods for m in dense_methods) or len(dense_methods) != len(set(dense_methods)):
        ap.error('--dense-methods must be distinct members of --methods')
    selected = set(args.clips.split(',')) if args.clips else None
    ev.FRAME_DIR.mkdir(parents=True, exist_ok=True)
    configuration = ev.eval_configuration(no_tiles=True)
    configuration['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    detector = PairedDetector(YoloDetector(local_config().detector))
    if detector.backend != 'openvino':
        raise RuntimeError('this experiment requires OpenVINO; refusing a silent backend change')
    rows, missing, dense = [], [], {}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)

    def save():
        timing = {}
        for m in methods:
            # First sequence includes model compilation, exclude it from latency summaries.
            ms = [x for r in rows[1:] for x in r['methods'][m]['detector_ms']]
            timing[m] = {'frames': len(ms), 'median_ms': statistics.median(ms) if ms else None,
                         'p95_ms': float(np.percentile(ms, 95)) if ms else None}
        target.write_text(json.dumps({'configuration': configuration, 'methods': methods,
            'rows': rows, 'summary': aggregate(rows, methods), 'timing': timing,
            'missing': missing, 'dense_methods': dense_methods, 'dense': dense}, indent=2)+'\n')

    for r in ev.load_rows():
        if selected and r['clip'] not in selected:
            continue
        try:
            frames = frames_for(r)
        except (FileNotFoundError, RuntimeError, __import__('subprocess').CalledProcessError):
            if not args.allow_missing:
                raise
            missing.append(r['id'])
            continue
        res = run_sequence(detector, frames, methods)
        rows.append({k: r[k] for k in ('id', 'clip', 'subset', 'split', 'label')} | {'methods': res})
        print(r['id'], r['label'], ' '.join(m+'='+res[m]['state'] for m in methods), flush=True)
        save()
    if args.dense:
        for clip, a, b in ev.NON_WALK + ev.DENSE_INFO:
            short = clip[2:].split('.')[0] if clip.startswith('h:') else clip
            if selected and short not in selected and clip not in selected:
                continue
            path = ev.HELDOUT_DIR / clip[2:] if clip.startswith('h:') else ROOT / ev.ev.CLIPS[clip]
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                if not args.allow_missing:
                    raise FileNotFoundError(path)
                missing.append(f'dense:{clip}@{a}-{b}')
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            detector.reset()
            pipes = {}
            for m in dense_methods:
                with patch.object(sp, 'build_detector', return_value=detector):
                    pipes[m] = sp.Pipeline(local_config(), use_surface=False)
            counts = {m: {'frames': 0, 'walk': 0, 'walk_at': [], 'raw_walk_heads': 0}
                      for m in dense_methods}
            i, next_t = 0, a
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                t = i/fps
                i += 1
                if t < next_t-1e-6:
                    continue
                if t > b:
                    break
                next_t += .25
                detector.cache.clear()  # one source frame at a time, bounded memory
                detector.index = i
                for m in dense_methods:
                    detector.method = m
                    out = pipes[m].process(img, annotate=False, t=t)
                    st = local_signal_state(out, img.shape)
                    counts[m]['frames'] += 1
                    counts[m]['walk'] += st == 'walk'
                    if st == 'walk':
                        counts[m]['walk_at'].append(round(t, 4))
                    counts[m]['raw_walk_heads'] += sum(d['raw'] == 'walk' for d in ev.heads(out))
            cap.release()
            if not counts[dense_methods[0]]['frames']:
                raise RuntimeError(f'no dense frames: {clip}')
            dense[f'{clip}@{a}-{b}'] = counts
            print('dense', clip, {m: dense[f'{clip}@{a}-{b}'][m]['walk'] for m in dense_methods}, flush=True)
            save()
    print(json.dumps(aggregate(rows, methods), indent=2), flush=True)


if __name__ == '__main__':
    main()
