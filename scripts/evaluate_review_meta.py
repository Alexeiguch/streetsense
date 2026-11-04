#!/usr/bin/env python3
"""Small paid Muse/SAM smoke check on public sample clips, not an accuracy benchmark.

python scripts/evaluate_review_meta.py --env-file .env
Saves sanitized responses, latency, estimated cost and SAM previews under outputs/review.
Only loads MODEL_API_KEY/LLAMA_API_KEY from the explicitly selected env file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streetsense.answers import local_signal_state
from streetsense.config import Config, VisionLLMConfig
from streetsense.meta_api import (
    MetaSamSegmenter, MetaVisionAssistant, fuse_crossing_decision,
    overlay_sam_segments, validate_crossing_reply,
)
from streetsense.meta_usage import get_ledger, mask_secret
from streetsense.pipeline import Pipeline


def frame_at(path, seconds):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'Cannot decode sample at {seconds}s: {path.name}')
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--env-file', type=Path)
    ap.add_argument('--repeats', type=int, default=2)
    args = ap.parse_args()
    if args.env_file:
        for line in args.env_file.read_text().splitlines():
            name, sep, value = line.partition('=')
            if sep and name.strip() in {'MODEL_API_KEY', 'LLAMA_API_KEY'}:
                os.environ.setdefault(name.strip(), value.strip().strip('\"\''))
    secret = os.environ.get('MODEL_API_KEY') or os.environ.get('LLAMA_API_KEY')
    if not secret:
        ap.error('Set MODEL_API_KEY or provide --env-file; never put credentials in arguments')
    target = ROOT / 'outputs/review'
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('STREETSENSE_META_LEDGER', str(target / 'meta-usage.jsonl'))
    os.environ.setdefault('STREETSENSE_META_BUDGET_USD', '1')
    os.environ['STREETSENSE_SAM_FALLBACK'] = '0'
    muse_cfg = VisionLLMConfig()
    muse = MetaVisionAssistant(muse_cfg)
    sam = MetaSamSegmenter(max_image_side=768)
    rows = []
    # Exact times and labels already reviewed in eval/muse_labels.csv.
    samples = [('dc_red', 'dc_ped_light.ogv', 3.0, 'red'),
               ('dc_walk', 'dc_ped_light.ogv', 13.8, 'walk'),
               ('dc_countdown', 'dc_ped_light.ogv', 51.88, 'countdown_ending'),
               ('tyumen', 'tyumen_crossing.webm', 6.0, 'unclear')]
    for name, filename, seconds, label in samples:
        cfg = Config()
        cfg.vlm.enabled = False
        pipe = Pipeline(cfg, use_surface=False)
        for k in range(4, -1, -1):
            stamp = seconds - .25 * k
            frame = frame_at(ROOT / 'samples' / filename, stamp)
            local_out = pipe.process(frame, annotate=False, t=stamp)
        local = local_signal_state(local_out, frame.shape)
        cv2.imwrite(str(target / f'{name}-source.jpg'), frame)
        for repeat in range(args.repeats):
            started = time.perf_counter()
            raw = muse.ask_sync(frame, 'Is it safe to cross the street right now?')
            validated = validate_crossing_reply(raw)
            row = {'kind': 'muse', 'sample': name, 'label': label, 'repeat': repeat,
                   'local_state': local, 'wall_s': round(time.perf_counter() - started, 3),
                   'validated': validated['ok'], 'reply': raw,
                   'fusion': fuse_crossing_decision(local, validated.get('data'))}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ['sample', 'repeat', 'wall_s', 'validated', 'fusion']}), flush=True)
        if name in {'dc_walk', 'tyumen'}:
            prompt = 'traffic light' if name == 'dc_walk' else 'crosswalk'
            started = time.perf_counter()
            try:
                seg = sam.segment(frame, prompt, decode_masks=True)
                preview = target / f'{name}-sam.jpg'
                cv2.imwrite(str(preview), overlay_sam_segments(frame, [seg]))
                row = {'kind': 'sam', 'sample': name, 'prompt': prompt,
                       'wall_s': round(time.perf_counter() - started, 3), 'api_s': seg.latency_s,
                       'estimated_cost_usd': seg.cost_usd, 'boxes': seg.boxes,
                       'mask_count': len(seg.masks), 'preview': preview.name}
            except Exception as exc:
                row = {'kind': 'sam', 'sample': name, 'error': mask_secret(str(exc), secret)[:200]}
            rows.append(row)
            print(json.dumps(row), flush=True)
        # Incremental output survives a later API failure without repeating paid calls.
        result = {'note': 'Smoke check only; four samples, not dataset accuracy or live safety validation.',
                  'model': muse_cfg.model, 'max_tokens': muse_cfg.max_tokens,
                  'crossing_image_side': muse_cfg.crossing_max_image_side,
                  'rows': rows, 'usage': get_ledger().summary()}
        (target / 'live-meta.json').write_text(mask_secret(json.dumps(result, indent=2), secret))


if __name__ == '__main__':
    main()
