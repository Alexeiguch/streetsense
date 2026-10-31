#!/usr/bin/env python3
"""Hard negatives for the signal CNN: round green lamps (vehicle lights, generic round greens, blurry green blobs)
labelled O ("other" = not walk). Built from TRAIN/VAL-eligible data only: records whose split is "test" are never
read, and PTLD test is not in the base pickle at all (see add_ptld_crops.py).

1. mined: existing O crops (train / val split) that the HSV reader calls green AND whose largest lit green blob is
   round (aspect 0.7-1.45, fill >= 0.55, circularity >= 0.6), so walk figures labelled O (bike signals, novelty
   figures) are not used. Each gets --mined-copies blurred / low-res copies (the original is already in the data).
   A copy keeps the original's split and group, so no leak across splits.
2. synthetic: rendered heads with a lit round green lamp (1, 2 or 3 lamps, unlit lamps dark), plus bare
   round green blobs with a white core, on varied backgrounds, then blur / low-res / noise / jpeg. Separate seeds
   for train and val.

  python scripts/signal_data/add_hard_negatives.py --base signal_ds_v4_ptld.pkl --out signal_ds_v5_ptld_hn.pkl
"""
import argparse, collections, math, pickle, random, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from streetsense.traffic_light import classify_light, signal_kind  # noqa: E402
from train_signal_cnn import tight_box  # noqa: E402

O = 3


def round_green(c):
    """Largest lit green blob is round (lamp), not a figure. Returns (ok, stats)."""
    h0, w0 = c.shape[:2]
    if h0 < 6 or w0 < 4:
        return False, {}
    H = 128; W = max(8, round(w0 * H / h0))
    hsv = cv2.cvtColor(cv2.resize(c, (W, H)), cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gh = (hue >= 40) & (hue <= 100)
    lit = (gh & (sat >= 50) & (val >= 120)) | ((sat < 70) & (val >= 230))
    lit = cv2.morphologyEx(lit.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(lit, connectivity=8)
    if n <= 1:
        return False, {}
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); x, y, w, h, a = (int(v) for v in st[i])
    if a < 30 or not (gh & (lab == i)).sum() >= 0.3 * a:
        return False, {}
    cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = max(1.0, cv2.arcLength(cnts[0], True))
    circ, aspect, fill = 4 * math.pi * a / per ** 2, h / max(1, w), a / max(1, w * h)
    return (0.7 <= aspect <= 1.45 and fill >= 0.55 and circ >= 0.6), dict(aspect=aspect, fill=fill, circ=circ)


def degrade(im, rng, strong=True):
    h, w = im.shape[:2]; s = max(h, w)
    sig = rng.uniform(0.03, 0.12 if strong else 0.06) * s
    im = cv2.GaussianBlur(im, (0, 0), max(0.6, sig))
    if rng.random() < 0.5:
        f = rng.uniform(0.15, 0.5)
        im = cv2.resize(cv2.resize(im, (max(2, int(w * f)), max(2, int(h * f))), interpolation=cv2.INTER_AREA),
                        (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        im = np.clip(im.astype(np.int16) + np.random.default_rng(rng.randrange(1 << 30)).integers(-10, 11, im.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, rng.randrange(30, 85)])
        im = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else im
    return im


def green_bgr(rng):
    hue = rng.uniform(45, 92)  # OpenCV hue: true green .. LED cyan
    v = np.uint8([[[int(hue), rng.randrange(150, 256), rng.randrange(200, 256)]]])
    return cv2.cvtColor(v, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


def background(h, w, rng):
    kind = rng.random()
    if kind < 0.35:  # sky
        top = np.array([rng.uniform(150, 255), rng.uniform(120, 220), rng.uniform(80, 200)], np.float32)
        bot = top * rng.uniform(0.7, 1.05)
        g = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        im = top * (1 - g) + bot * g; im = np.repeat(im, w, 1)
    elif kind < 0.65:  # night / dark
        im = np.full((h, w, 3), rng.uniform(0, 50), np.float32)
    else:  # building / foliage texture
        base = np.array([rng.uniform(30, 200) for _ in range(3)], np.float32)
        noise = cv2.GaussianBlur(np.random.default_rng(rng.randrange(1 << 30)).normal(0, 40, (h, w, 3)).astype(np.float32), (0, 0), 3)
        im = base + noise
    return im


def synth(rng):
    """One synthetic O crop: lit round green lamp in a 1/2/3-lamp housing, or a bare round green blob."""
    t = rng.random()
    if t < 0.2:  # bare blob, roughly round
        r = rng.uniform(10, 40); bh, bw = int(r * rng.uniform(2.2, 3)), int(r * rng.uniform(2.2, 3))
        im = background(bh, bw, rng) * rng.uniform(0.1, 0.6)
        cx, cy = bw / 2 + rng.uniform(-3, 3), bh / 2 + rng.uniform(-3, 3); lamps = [(cx, cy, r)]; lit = 0
        box = (0, 0, bw, bh)
    else:
        n = 3 if t < 0.6 else (2 if t < 0.85 else 1)
        r = rng.uniform(6, 30); gap = r * rng.uniform(0.25, 0.6)
        hw = 2 * r + 2 * gap; hh = n * 2 * r + (n + 1) * gap
        if rng.random() < 0.3 and n > 1:  # horizontal head
            hw, hh = hh, hw
        bw, bh = int(hw * rng.uniform(1.2, 1.6)), int(hh * rng.uniform(1.2, 1.6))
        im = background(bh, bw, rng)
        x0, y0 = (bw - hw) / 2, (bh - hh) / 2
        col = rng.uniform(5, 60)
        cv2.rectangle(im, (int(x0), int(y0)), (int(x0 + hw), int(y0 + hh)), (col, col, col * rng.uniform(0.9, 1.2)), -1)
        horiz = hw > hh
        lamps = []
        for k in range(n):
            if horiz:
                lamps.append((x0 + gap + r + k * (2 * r + gap), y0 + hh / 2, r))
            else:
                lamps.append((x0 + hw / 2, y0 + gap + r + k * (2 * r + gap), r))
        lit = n - 1 if rng.random() < 0.8 else rng.randrange(n)  # green usually the bottom / last lamp
        for k, (cx, cy, rr) in enumerate(lamps):
            if k != lit:
                tint = [(20, 20, 60), (20, 50, 60), (25, 25, 25)][k % 3]
                cv2.circle(im, (int(cx), int(cy)), int(rr), tuple(float(v) * rng.uniform(0.6, 1.4) for v in tint), -1)
        box = (int(x0), int(y0), int(x0 + hw), int(y0 + hh))
    cx, cy, rr = lamps[lit]
    g = green_bgr(rng)
    yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + ((yy - cy) / rng.uniform(0.9, 1.1)) ** 2) / rr
    disc = np.clip(1.2 - d, 0, 1)[..., None] ** 0.5
    glow = np.exp(-np.maximum(d - 1, 0) ** 2 / rng.uniform(0.05, 0.6))[..., None] * rng.uniform(0.2, 0.8)
    core = np.clip(1 - d / rng.uniform(0.3, 0.7), 0, 1)[..., None] * rng.uniform(0, 1)
    lamp = np.maximum(disc, glow)
    im = im * (1 - lamp) + g * lamp
    im = im * (1 - core) + 255 * core
    if rng.random() < 0.3:  # LED dot grid
        dots = ((xx.astype(int) % 3 == 0) & (yy.astype(int) % 3 == 0) & (d < 1))[..., None]
        im = np.where(dots, im * 1.1, im * 0.85)
    im = np.clip(im, 0, 255).astype(np.uint8)
    im = degrade(im, rng, strong=rng.random() < 0.7)
    x1, y1, x2, y2 = box
    return im, (x1, y1, im.shape[1] - x2, im.shape[0] - y2), 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/datasets/crops/signal_ds_v4_ptld.pkl")
    ap.add_argument("--out", default="/workspace/datasets/crops/signal_ds_v5_ptld_hn.pkl")
    ap.add_argument("--mined-copies", type=int, default=2)
    ap.add_argument("--synth-train", type=int, default=1500)
    ap.add_argument("--synth-val", type=int, default=300)
    ap.add_argument("--sheet", default="/workspace/datasets/crops/hn_contact.jpg")
    a = ap.parse_args()
    D = pickle.load(open(a.base, "rb")); n0 = len(D)
    assert not any(r["src"].startswith("hn_") for r in D)
    rng = random.Random(0); add = []
    for r in D:
        if r["split"] not in ("train", "val") or r["y"] != O:  # test never read
            continue
        x1, y1, x2, y2 = tight_box(r); c = r["crop"][y1:y2, x1:x2]
        if classify_light(c, signal_kind(x2 - x1, y2 - y1)).state != "green" or not round_green(c)[0]:
            continue
        for k in range(a.mined_copies):
            add.append(dict(src=f"hn_mined:{r['src'].split(':')[0]}", grp=r["grp"], split=r["split"], y=O,
                            crop=degrade(r["crop"].copy(), rng), pad=r["pad"], scale=r["scale"],
                            uid=f"hn_{r.get('uid', r.get('name', ''))}_{k}"))
    for split, n, seed in (("train", a.synth_train, 1), ("val", a.synth_val, 2)):
        srng = random.Random(seed)
        for k in range(n):
            im, pad, s = synth(srng)
            add.append(dict(src="hn_synth", grp=f"hn_synth:{split}:{k}", split=split, y=O, crop=im, pad=pad, scale=s,
                            uid=f"hn_synth_{split}_{k}"))
    D += add
    print("added", len(add), collections.Counter((d["split"], d["src"]) for d in add))
    pickle.dump(D, open(a.out, "wb")); print("->", a.out, len(D), "from", n0)
    if a.sheet:  # contact sheet for eyeballing (not committed)
        pick = random.Random(3).sample(add, min(80, len(add))); cells = []
        for d in pick:
            c = d["crop"]; s = 80 / max(c.shape[:2]); c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))
            o = np.zeros((80, 80, 3), np.uint8); o[:c.shape[0], :c.shape[1]] = c; cells.append(o)
        cells += [np.zeros((80, 80, 3), np.uint8)] * (-len(cells) % 10)
        cv2.imwrite(a.sheet, np.vstack([np.hstack(cells[i:i + 10]) for i in range(0, len(cells), 10)]))


if __name__ == "__main__":
    main()
