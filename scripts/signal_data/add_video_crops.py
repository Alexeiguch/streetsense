#!/usr/bin/env python3
"""Append labelled video crops (label_video_crops.py output) to a signal_ds pickle -> new pickle.

Split is by clip (no frame leakage): clips in --train go to train, everything else to test.
  python scripts/signal_data/add_video_crops.py --base /workspace/datasets/crops/signal_ds.pkl \
     --dir /workspace/datasets/commons_video --out /workspace/datasets/crops/signal_ds_v2.pkl
"""
import argparse, collections, csv, pickle, sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_crops import CL, cut  # noqa: E402

TRAIN_CLIPS = {"crossing_fr", "toronto_diag", "dus_r2g", "faulty", "wien", "chelyabinsk", "toronto_pxo"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/datasets/crops/signal_ds.pkl")
    ap.add_argument("--dir", default="/workspace/datasets/commons_video")
    ap.add_argument("--out", default="/workspace/datasets/crops/signal_ds_v2.pkl")
    ap.add_argument("--src", default="cvid")
    a = ap.parse_args()
    D = Path(a.dir)
    files = {r["id"]: r["file"] for r in csv.DictReader(open(D / "clips.csv"))}
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(D / "labels.csv")):
        by[r["clip"]].append(r)
    out = pickle.load(open(a.base, "rb"))
    n0 = len(out)
    for clip, rs in by.items():
        cap = cv2.VideoCapture(str(D / files[clip])); fps = cap.get(cv2.CAP_PROP_FPS) or 25
        want = collections.defaultdict(list)
        for r in rs:
            want[int(round(float(r["t"]) * fps))].append(r)
        k = 0
        while want:
            ok, f = cap.read()
            if not ok:
                break
            for r in want.pop(k, []):
                c, pad, s = cut(f, *(int(r[q]) for q in ("x1", "y1", "x2", "y2")))
                out.append(dict(src=f"{a.src}:{clip}", grp=f"{clip}:{int(float(r['t']) // 5)}",
                                split="train" if clip in TRAIN_CLIPS else "test", y=CL[r["label"]],
                                crop=c, pad=pad, scale=s, name=r["name"]))
            k += 1
    new = out[n0:]
    print(len(new), "video crops", collections.Counter((r["split"], "WSCO"[r["y"]]) for r in new))
    pickle.dump(out, open(a.out, "wb"))


if __name__ == "__main__":
    main()
