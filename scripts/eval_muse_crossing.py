#!/usr/bin/env python3
"""Muse Spark crossing eval on the labelled frames in eval/muse_labels.csv.

For each labelled (clip, time) this extracts the frame at t and the frames 0.5 s and
1.0 s earlier, runs the local light reader over t-1.0 .. t (0.25 s steps, fresh
pipeline per sample, no model calls), then asks Muse Spark once per variant.

Responses are cached in eval/muse_cache/<sha256>.json keyed on the exact request
(model, max_tokens, reasoning_effort, messages), so reruns do not re-pay. Only cache
misses hit the API and append to the usage ledger. The API key is never printed.

  STREETSENSE_META_BUDGET_USD=3.2 .venv/bin/python scripts/eval_muse_crossing.py --variants A,B
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streetsense.config import Config, VisionLLMConfig  # noqa: E402
from streetsense.frame_history import history_summary_text  # noqa: E402
from streetsense.meta_api import (  # noqa: E402
    MetaVisionAssistant,
    build_user_content,
    encode_jpeg_data_url,
    fuse_crossing_decision,
    validate_crossing_reply,
)
from streetsense.meta_usage import get_ledger  # noqa: E402

CLIPS = {
    "dc": "samples/dc_ped_light.ogv",
    "tyumen": "samples/tyumen_crossing.webm",
    "cdmx": "samples/cdmx_crossing_720p.webm",
    "mazatlan": "samples/mazatlan_sidewalk.webm",
    "tomaszow": "samples/tomaszow_scooter.ogv",
}
QUESTION = "Is it safe to cross the street right now?"

# name -> (prompt_variant, n_history_frames, send_local_history_text, max_image_side)
VARIANTS = {
    "A": ("strict", 0, False, 512),  # baseline: current default
    "B": ("describe", 0, False, 512),
    "C": ("describe", 2, False, 512),  # t-1.0, t-0.5, t
    "D": ("describe", 2, True, 512),  # C + local reader history text
    "E": ("describe", 0, False, 1024),
    "F": ("describe", 2, False, 1024),
    "G": ("strict", 0, False, 1024),  # resolution only, old prompt
}
NON_WALK = {"red", "flashing", "countdown_ending", "no_signal"}  # "unclear" and "walk_symbol_off" are reported, not scored


def extract(clip: str, t: float, cache_dir: Path):
    path = cache_dir / f"{clip}_{t:06.2f}.png"
    if not path.is_file():
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(ROOT / CLIPS[clip]),
             "-frames:v", "1", "-y", str(path)],
            check=True,
        )
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return img


def local_reader(clip: str, t: float, frame_dir: Path, cache: dict) -> dict:
    key = f"{clip}@{t:.2f}"
    if key in cache:
        return cache[key]
    from streetsense.answers import local_signal_state
    from streetsense.pipeline import Pipeline

    cfg = Config()
    cfg.vlm.enabled = False  # never call Muse from the local pass
    pipe = Pipeline(cfg, use_surface=False)
    states = []
    for k in range(4, -1, -1):
        tt = round(t - 0.25 * k, 2)
        if tt < 0:
            continue
        img = extract(clip, tt, frame_dir)
        out = pipe.process(img, annotate=False, t=tt)
        states.append((round(t - tt, 2), local_signal_state(out, img.shape)))
    res = {"state": states[-1][1], "history": states}
    cache[key] = res
    return res


def request_key(vlm: MetaVisionAssistant, content: list[dict], repeat: int = 0) -> str:
    """Cache key = the exact request. repeat > 0 adds a salt so a repeat run re-asks (and pays)."""
    blob = json.dumps(
        {**({"repeat": repeat} if repeat else {}), "model": vlm.cfg.model, "max_tokens": vlm.max_tokens,
         "reasoning_effort": vlm.cfg.reasoning_effort, "messages": vlm.build_messages(content)},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def run_variant(name: str, rows: list[dict], local: dict, frame_dir: Path, cache_dir: Path,
                dry: bool, repeat: int = 0) -> list[dict]:
    prompt, n_hist, use_text, side = VARIANTS[name]
    vlm = MetaVisionAssistant(
        VisionLLMConfig(prompt_variant=prompt, max_image_side=side, crossing_max_image_side=side)
    )
    out = []
    for r in rows:
        t = float(r["time_s"])
        cur = extract(r["clip"], t, frame_dir)
        images = []
        for k in range(n_hist, 0, -1):
            tt = round(t - 0.5 * k, 2)
            if tt >= 0:
                images.append((0.5 * k, encode_jpeg_data_url(extract(r["clip"], tt, frame_dir), side)))
        images.append((0.0, encode_jpeg_data_url(cur, side)))
        lr = local[f"{r['clip']}@{t:.2f}"]
        text = history_summary_text([(a, s) for a, s in lr["history"]]) if use_text else None
        content = build_user_content(QUESTION, images, text)
        key = request_key(vlm, content, repeat)
        cpath = cache_dir / f"{key}.json"
        cached = cpath.is_file()
        if cached:
            data = json.loads(cpath.read_text())
        elif dry:
            data = None
        else:
            if not vlm.enabled:
                raise SystemExit("MODEL_API_KEY not set")
            data = vlm._call(content, QUESTION)
            cpath.write_text(json.dumps({k: v for k, v in data.items()}, indent=1))
        row = {"id": r["id"], "label": r["label"], "variant": name, "cached": cached,
               "local": lr["state"], "n_images": len(images)}
        if data is not None:
            v = validate_crossing_reply(data)
            d = v["data"] if v["ok"] else None
            fused = fuse_crossing_decision(lr["state"], d)
            if not v["ok"] and fused["decision"] == "go":
                fused["decision"] = "unsure"
            u = data.get("_usage") or {}
            row.update({
                "valid": v["ok"], "error": v["error"],
                "muse_safe": d["safe_to_cross"] if d else "invalid",
                "muse_signal": d["pedestrian_signal"] if d else None,
                "countdown": d.get("countdown_seconds") if d else None,
                "desc": (d or {}).get("signal_description") or (d or {}).get("summary"),
                "fused": fused["decision"],
                "latency_s": data.get("_latency_s"),
                "cost_usd": u.get("estimated_cost_usd", 0.0),
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
            })
        out.append(row)
        print(json.dumps({k: row.get(k) for k in ("id", "label", "local", "muse_safe", "muse_signal",
                                                    "countdown", "fused", "latency_s", "cached")}))
    return out


def summarise(rows: list[dict]) -> dict:
    by = {}
    for lab in ("walk", "walk_symbol_off", "countdown_ending", "flashing", "red", "no_signal", "unclear"):
        rs = [r for r in rows if r["label"] == lab and "muse_safe" in r]
        by[lab] = {k: sum(1 for r in rs if r["muse_safe"] == k) for k in ("yes", "no", "unsure", "invalid")}
        by[lab]["n"] = len(rs)
        by[lab]["fused_go"] = sum(1 for r in rs if r["fused"] == "go")
    rs = [r for r in rows if "muse_safe" in r]
    walk = by["walk"]
    lat = [r["latency_s"] for r in rs if r.get("latency_s")]
    return {
        "per_label": by,
        "walk_yes_rate": f"{walk['yes']}/{walk['n']}",
        "false_yes_non_walk": sum(by[lab]["yes"] for lab in NON_WALK),
        "fused_go_non_walk": sum(by[lab]["fused_go"] for lab in NON_WALK),
        "fused_go_walk": f"{walk['fused_go']}/{walk['n']}",
        "unsure_or_invalid_rate": round(sum(1 for r in rs if r["muse_safe"] in ("unsure", "invalid")) / max(1, len(rs)), 3),
        "mean_latency_s": round(statistics.mean(lat), 3) if lat else None,
        "p95_latency_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 3) if lat else None,
        "cost_usd": round(sum(r.get("cost_usd") or 0 for r in rs), 6),
        "mean_prompt_tokens": round(statistics.mean([r["prompt_tokens"] for r in rs if r.get("prompt_tokens")]), 1) if rs else None,
        "n": len(rs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="eval/muse_labels.csv")
    ap.add_argument("--variants", default="A")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list of ids")
    ap.add_argument("--dry-run", action="store_true", help="local reader + cache only, no API")
    ap.add_argument("--out", default="docs/benchmark/muse_eval.json")
    ap.add_argument("--repeat", type=int, default=0, help="repeat index; >0 re-asks and stores as <variant>#<n>")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(ROOT / a.labels)))
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["id"] in keep]
    if a.limit:
        rows = rows[: a.limit]
    frame_dir = ROOT / "outputs/muse_eval_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "eval/muse_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    lpath = ROOT / "eval/muse_local_reader.json"
    local = json.loads(lpath.read_text()) if lpath.is_file() else {}
    t0 = time.time()
    for r in rows:
        local_reader(r["clip"], float(r["time_s"]), frame_dir, local)
    lpath.write_text(json.dumps(local, indent=1, sort_keys=True))
    print(f"local reader ready ({time.time() - t0:.1f}s)", file=sys.stderr)
    outp = ROOT / a.out
    results = json.loads(outp.read_text()) if outp.is_file() else {"variants": {}}
    for name in a.variants.split(","):
        rs = run_variant(name, rows, local, frame_dir, cache_dir, a.dry_run, a.repeat)
        key = f"{name}#{a.repeat}" if a.repeat else name
        if not a.only and not a.limit:
            prompt, n_hist, use_text, side = VARIANTS[name]
            results["variants"][key] = {
                "config": {"prompt": prompt, "history_frames": n_hist, "history_text": use_text,
                           "max_image_side": side},
                "summary": summarise(rs), "rows": rs,
            }
        print(name, json.dumps(summarise(rs)), file=sys.stderr)
    results["ledger"] = {k: v for k, v in get_ledger().summary().items() if k != "ledger"}
    if not a.only and not a.limit:
        outp.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
