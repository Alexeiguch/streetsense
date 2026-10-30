#!/usr/bin/env python3
"""Cut per-light crops from PTLD (PUCRS Brazil pedestrian lights; licence UNCONFIRMED -> test only).

Expects the extracted archive in /workspace/datasets/ptld/{train,valid,test}/{split}/ with one
<image>.txt per image: one line per light, "cls x y w h img_w img_h" (x, y = top-left corner, pixels;
cls 0 go, 1 stop, 2 off; matches the VOC xml in {split}.ann.GoStopOff). Writes
crops/<split>/<label>/<stem>_<i>.jpg and boxes.csv (split,label,ptld_label,file,i,x1,y1,x2,y2,img_w,img_h,crop)
plus contact_<split>_<label>.jpg sheets. Labels: go->walk, stop->stop, off->other (repo CLASSES).

    python scripts/get_ptld_crops.py [--root /workspace/datasets/ptld]
"""
import argparse, csv, random
from pathlib import Path
import cv2
import numpy as np

PTLD = {0: "go", 1: "stop", 2: "off"}
MAP = {"go": "walk", "stop": "stop", "off": "other"}


def sheet(paths, out, n=80, cell=96, cols=10):
    paths = random.Random(0).sample(paths, min(n, len(paths)))
    rows = (len(paths) + cols - 1) // cols
    S = np.zeros((rows * cell, cols * cell, 3), np.uint8)
    for k, p in enumerate(paths):
        im = cv2.imread(str(p)); h, w = im.shape[:2]; s = cell / max(h, w)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        r, c = divmod(k, cols); S[r * cell:r * cell + im.shape[0], c * cell:c * cell + im.shape[1]] = im
    cv2.imwrite(str(out), S)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/workspace/datasets/ptld"); a = ap.parse_args()
    root = Path(a.root); rows = []; by = {}
    for split in ("train", "valid", "test"):
        d = root / split / split
        for t in sorted(d.glob("*.txt")):
            img = next((p for p in d.glob(t.stem + ".*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")), None)
            lines = [l.split() for l in t.read_text().splitlines() if l.strip()]
            if img is None or not lines:
                continue
            im = cv2.imread(str(img)); H, W = im.shape[:2]
            for i, f in enumerate(lines):
                c, x, y, w, h = (int(float(v)) for v in f[:5])
                x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                lab = MAP[PTLD[c]]; out = root / "crops" / split / lab / f"{t.stem}_{i}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(out), im[y1:y2, x1:x2])
                rows.append([split, lab, PTLD[c], str(img.relative_to(root)), i, x1, y1, x2, y2, W, H, str(out.relative_to(root))])
                by.setdefault((split, lab), []).append(out)
    with open(root / "boxes.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow("split,label,ptld_label,file,i,x1,y1,x2,y2,img_w,img_h,crop".split(",")); w.writerows(rows)
    for (s, l), ps in sorted(by.items()):
        print(s, l, len(ps)); sheet(ps, root / f"contact_{s}_{l}.jpg")
    print("crops", len(rows))


if __name__ == "__main__":
    main()
