#!/usr/bin/env python3
"""PTLD (PUCRS Brazil pedestrian lights) GT boxes -> signal crops for signal_cnn v4_ptld.

LICENCE UNCONFIRMED: the PTLD readme states no licence or usage terms. Crops built here may only feed the
opt-in model v4_ptld (STREETSENSE_SIGNAL_CNN_VERSION=v4_ptld), never the default, never shipped.

Uses PTLD **train** -> split "train" and PTLD **valid** -> split "val" (model selection + t_walk).
PTLD **test** is never read here: it stays the held-out benchmark for scripts/eval_ptld.py.
Labels (as get_ptld_crops.py / boxes.csv): go -> W (walk), stop -> S, off -> O (other).
Note: many PTLD go heads show green digits beside the walk figure; PTLD labels them go, so they become W here
(our own labels put walk+digits in C). This is a deliberate label-scheme difference for this opt-in model.

  python scripts/signal_data/add_ptld_crops.py --base signal_ds_v3.pkl --out signal_ds_v4_ptld.pkl
"""
import argparse, collections, csv, pickle, sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_crops import CL, cut  # noqa: E402

LAB = {"walk": "W", "stop": "S", "other": "O"}
SPLIT = {"train": "train", "valid": "val"}  # PTLD test deliberately absent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/datasets/ptld")
    ap.add_argument("--base", default="/workspace/datasets/crops/signal_ds_v3.pkl")
    ap.add_argument("--out", default="/workspace/datasets/crops/signal_ds_v4_ptld.pkl")
    a = ap.parse_args(); root = Path(a.root)
    D = pickle.load(open(a.base, "rb")); n0 = len(D)
    assert not any(r["src"].startswith("ptld") for r in D), "base already has PTLD crops"
    rows = [r for r in csv.DictReader(open(root / "boxes.csv")) if r["split"] in SPLIT]
    assert all(r["split"] != "test" for r in rows)
    img, cur = None, None
    for r in sorted(rows, key=lambda r: r["file"]):
        if r["file"] != cur:
            cur = r["file"]; img = cv2.imread(str(root / cur))
        c, pad, s = cut(img, *(int(r[k]) for k in ("x1", "y1", "x2", "y2")))
        D.append(dict(src="ptld", grp=f"ptld:{cur}", split=SPLIT[r["split"]], y=CL[LAB[r["label"]]], crop=c, pad=pad,
                      scale=s, uid=f"ptld_{Path(cur).stem}_{r['i']}"))
    print("added", len(D) - n0, collections.Counter((d["split"], d["y"]) for d in D[n0:]))
    pickle.dump(D, open(a.out, "wb")); print("->", a.out)


if __name__ == "__main__":
    main()
