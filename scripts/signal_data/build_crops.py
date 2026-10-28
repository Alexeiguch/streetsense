#!/usr/bin/env python3
"""Rebuild the training crop set from eval/signal_labels.csv.gz + fetched raw data.

Output pickle rows: src, grp, split, y (0 walk,1 stop,2 countdown,3 other), crop (BGR, 15 %
pad, max side 160), pad (l,t,r,b px in source), scale. Held-out eval clips are not in the CSV.
  python scripts/signal_data/build_crops.py --data /workspace/datasets --out /workspace/datasets/crops/signal_ds.pkl
"""
import argparse, collections, csv, gzip, hashlib, pickle, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
CL = {"W": 0, "S": 1, "C": 2, "O": 3}


def cut(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    pw, ph = int(0.15 * (x2 - x1)), int(0.15 * (y2 - y1))
    X1, Y1, X2, Y2 = max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph)
    c = img[Y1:Y2, X1:X2]
    s = min(1.0, 160 / max(c.shape[:2]))
    if s < 1:
        c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return c.copy(), (x1 - X1, y1 - Y1, X2 - x2, Y2 - y2), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/datasets")
    ap.add_argument("--out", default="/workspace/datasets/crops/signal_ds.pkl")
    a = ap.parse_args()
    D = Path(a.data)
    rows = list(csv.DictReader(gzip.open(ROOT / "eval/signal_labels.csv.gz", "rt")))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["src"], r["key"], r["t"])].append(r)
    out = []

    def emit(img, rs, src, grp):
        for r in rs:
            c, pad, s = cut(img, *(int(r[k]) for k in ("x1", "y1", "x2", "y2")))
            out.append(dict(src=src, grp=grp, split=r["split"], y=CL[r["label"]], crop=c, pad=pad, scale=s))

    # PTL (full-resolution zips)
    from ptl_zip import entries, read
    want = {k[1]: v for k, v in by.items() if k[0] == "ptl"}
    for z in sorted((D / "ptl").glob("full*.zip")):
        for zi, off in entries(str(z)):
            name = zi.filename.rsplit("/", 1)[-1]
            if name in want:
                img = cv2.imdecode(np.frombuffer(read(str(z), zi, off), np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    emit(img, want.pop(name), "ptl", name)
    print("ptl missing images:", len(want))
    # Commons
    miss = 0
    for (src, key, _), rs in by.items():
        if src != "commons":
            continue
        fn = D / "commons/img" / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".jpg")
        img = cv2.imread(str(fn))
        if img is None:
            miss += 1; continue
        emit(img, rs, "commons", key)
    print("commons missing images:", miss)
    # dev clips (samples/, see scripts/eval_muse_crossing.py CLIPS)
    import eval_muse_crossing as ev
    clips = collections.defaultdict(list)
    for (src, key, t), rs in by.items():
        if src == "dev":
            clips[key].append((float(t), rs))
    for clip, items in clips.items():
        cap = cv2.VideoCapture(ev.CLIPS[clip]); fps = cap.get(cv2.CAP_PROP_FPS)
        for t, rs in sorted(items):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, img = cap.read()
            if ok:
                for r in rs:  # dev crops were tight boxes, no pad
                    x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
                    out.append(dict(src="dev:" + clip, grp=f"{clip}:{int(t // 10)}", split=r["split"], y=CL[r["label"]],
                                    crop=img[y1:y2, x1:x2].copy(), pad=(0, 0, 0, 0), scale=1.0))
    pickle.dump(out, open(a.out, "wb"))
    print(len(out), "crops ->", a.out, collections.Counter((r["split"], "WSCO"[r["y"]]) for r in out))


if __name__ == "__main__":
    main()
