#!/usr/bin/env python3
"""Score the local light reader alone (no model calls, $0).

For every labelled frame in eval/muse_labels.csv, eval/muse_labels_broad.csv and
eval/reader_labels_heldout.csv, run a fresh Pipeline over the last 1 s (5 frames,
0.25 s apart; a still is repeated 5 times) and take the crossing decision on the
last frame. Report, per subset and split (dev / held-out, split by clip):

* walk recall   - reader says walk on frames labelled walk
* false walk    - reader says walk on any frame NOT labelled walk (must be 0)
* combined go   - reader walk AND cached Muse yes (E@768 answers in
                  docs/benchmark/muse_eval*.json); new held-out frames have no Muse answer

--dense also streams known non-walk intervals at 4 fps through one Pipeline and
counts frames whose decision is walk (must be 0) and raw per-head walk reads.

    .venv/bin/python scripts/eval_local_reader.py [--dense] [--out docs/benchmark/local_reader_eval.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_muse_crossing as ev  # noqa: E402
from streetsense.answers import local_signal_state  # noqa: E402
from streetsense.config import Config  # noqa: E402
from streetsense.pipeline import Pipeline  # noqa: E402

HELDOUT_DIR = ROOT / "samples" / "heldout"
FRAME_DIR = ROOT / "outputs" / "muse_eval_frames"
# Clips the reader was tuned on. Everything else is held out (split by clip, not by frame).
DEV_CLIPS = {"dc", "eu_ped_a", "eu_ped_b", "night_ped", "most11", "us_ped", "tirana"}
LABEL_FILES = ["eval/muse_labels.csv", "eval/muse_labels_broad.csv", "eval/reader_labels_heldout.csv"]
MUSE_FILES = ["docs/benchmark/muse_eval.json", "docs/benchmark/muse_eval_broad.json"]
STILL_EXT = {".jpg", ".jpeg", ".png"}

# Known non-walk stretches (seconds) for the dense scan. Taken only from inside labelled
# red / flashing / countdown-ending / no-signal phases, away from transitions.
NON_WALK = [
    ("dc", 0.0, 9.5), ("dc", 53.5, 62.8),
    ("us_ped", 7.5, 23.4), ("most11", 0.0, 5.5), ("most11", 9.5, 16.5),
    ("eu_ped_a", 0.0, 1.0), ("eu_ped_b", 0.0, 3.0), ("night_ped", 3.5, 10.0),
    ("eu_ped_side", 0.0, 1.0), ("most02", 6.0, 8.7), ("most04", 12.0, 16.0),
    ("night_hand", 0.0, 10.2), ("beacon_a", 0.0, 3.9), ("beacon_b", 0.0, 3.1),
    ("night_square", 0.0, 28.0), ("tirana", 0.0, 12.8), ("tyumen", 0.0, 20.5),
    ("cdmx", 0.0, 112.0), ("mazatlan", 0.0, 5.1), ("tomaszow", 0.0, 11.7),
    ("h:pexels_6415590.mp4", 0.0, 6.5), ("h:nz.ogv", 3.4, 4.1), ("h:ffm_vehicle.mp4", 0.0, 5.4),
]

# US countdown-ending stretch: the reader cannot read digits and says walk here by design;
# Muse vetoes it (see docs/MUSE_EVAL.md). Reported separately, must not grow.
DENSE_INFO = [("dc", 48.5, 53.0)]


def load_rows() -> list[dict]:
    rows = []
    for lf in LABEL_FILES:
        for r in csv.DictReader(open(ROOT / lf)):
            r["source"] = lf
            r["subset"] = r.get("subset") or "dc_orig"
            r["split"] = "dev" if r["clip"] in DEV_CLIPS else "heldout"
            rows.append(r)
    return rows


def frame_at(r: dict, t: float):
    if r.get("file"):
        p = HELDOUT_DIR / r["file"]
        if p.suffix.lower() in STILL_EXT:
            return cv2.imread(str(p))
        out = FRAME_DIR / f"h_{r['clip']}_{t:06.2f}.png"
        if not out.is_file():
            import subprocess
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(p), "-frames:v", "1",
                            "-y", str(out)], check=True)
        return cv2.imread(str(out))
    return ev.extract(r["clip"], t, FRAME_DIR)


def new_pipe() -> Pipeline:
    cfg = Config()
    cfg.vlm.enabled = False
    return Pipeline(cfg, use_surface=False)


def heads(out) -> list[dict]:
    return [{"kind": d.attrs.get("signal_kind"), "raw": d.attrs.get("light_raw"),
             "state": d.attrs.get("light_state"), "conf": round(float(d.conf), 2),
             "wh": [int(d.box.w), int(d.box.h)]}
            for d in out.detections if d.label == "traffic light"]


def read_row(r: dict) -> dict:
    t = float(r["time_s"])
    still = bool(r.get("file")) and Path(r["file"]).suffix.lower() in STILL_EXT
    pipe = new_pipe()
    hist = []
    out = img = None
    for k in range(4, -1, -1):
        tt = round(t - 0.25 * k, 2) if not still else float(4 - k)
        if tt < 0:
            continue
        img = frame_at(r, tt)
        out = pipe.process(img, annotate=False, t=tt)
        hist.append(local_signal_state(out, img.shape))
    return {"state": hist[-1], "history": hist, "heads": heads(out)}


def muse_answers() -> dict:
    ans = {}
    for mf in MUSE_FILES:
        p = ROOT / mf
        if p.is_file():
            for row in json.load(open(p))["variants"].get("E@768", {}).get("rows", []):
                ans[row["id"]] = row.get("muse_safe")
    return ans


def dense_scan(spans) -> dict:
    res = {}
    for clip, a, b in spans:
        path = HELDOUT_DIR / clip[2:] if clip.startswith("h:") else ROOT / ev.CLIPS[clip]
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        pipe = new_pipe()
        n = walk = raw_walk = 0
        hits = []
        next_t = a
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            if t < next_t - 1e-6:
                continue
            if t > b:
                break
            next_t += 0.25
            out = pipe.process(img, annotate=False, t=t)
            n += 1
            st = local_signal_state(out, img.shape)
            rw = sum(1 for d in out.detections if d.label == "traffic light" and d.attrs.get("light_raw") == "walk")
            raw_walk += rw
            if st == "walk":
                walk += 1
                hits.append(round(t, 2))
        cap.release()
        res[f"{clip}@{a}-{b}"] = {"frames": n, "walk": walk, "raw_walk_heads": raw_walk, "walk_at": hits[:20]}
        print(f"  dense {clip} {a}-{b}: frames {n} walk {walk} raw-walk heads {raw_walk}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    muse = muse_answers()
    rows = load_rows()
    per = []
    for r in rows:
        res = read_row(r)
        m = muse.get(r["id"])
        per.append({"id": r["id"], "subset": r["subset"], "split": r["split"], "label": r["label"],
                    "local": res["state"], "muse_safe": m,
                    "go": res["state"] == "walk" and m == "yes", **res})
    agg = defaultdict(lambda: {"walk_frames": 0, "reader_walk": 0, "nonwalk_frames": 0,
                               "false_walk": 0, "muse_cached": 0, "go": 0, "false_go": 0})
    for p in per:
        for key in (f"{p['split']}/{p['subset']}", f"{p['split']}/ALL", "ALL"):
            a = agg[key]
            if p["label"] == "walk":
                a["walk_frames"] += 1
                a["reader_walk"] += p["local"] == "walk"
                a["muse_cached"] += p["muse_safe"] is not None
                a["go"] += p["go"]
            else:
                a["nonwalk_frames"] += 1
                a["false_walk"] += p["local"] == "walk"
                a["false_go"] += p["go"]
    print(f"{'split/subset':28s} walk-recall  false-walk  go(cached Muse)  false-go")
    for k in sorted(agg):
        a = agg[k]
        print(f"{k:28s} {a['reader_walk']:>3}/{a['walk_frames']:<3}      {a['false_walk']:>3}/{a['nonwalk_frames']:<3}"
              f"     {a['go']:>3}/{a['muse_cached']:<3}          {a['false_go']}")
    bad = [p for p in per if p["label"] != "walk" and p["local"] == "walk"]
    for p in bad:
        print("  FALSE WALK:", p["id"], p["label"], p["heads"])
    result = {"summary": dict(agg), "rows": per}
    if args.dense:
        result["dense"] = dense_scan(NON_WALK)
        tot = sum(v["walk"] for v in result["dense"].values())
        print("dense non-walk frames with walk decision:", tot)
        result["dense_info"] = dense_scan(DENSE_INFO)
    if args.out:
        Path(ROOT / args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for p in per:
        if p["label"] == "walk" and p["local"] != "walk":
            print("  miss:", p["id"], p["local"], p["heads"])


if __name__ == "__main__":
    main()
