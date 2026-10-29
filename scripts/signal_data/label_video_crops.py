#!/usr/bin/env python3
"""Auto-label mined video light boxes (boxes.csv) with per-clip rules + HSV lit buckets.

Rules (see eval/SOURCES.md "Commons signal videos"): vehicle-only clips -> O; pedestrian clips:
h/w > 2.6 -> O (vehicle head), lit red -> S (C if wide hand+digits / digits on top), lit green/white
-> W candidate (every W candidate was then reviewed by eye; fixes in --fixes JSON {crop_name: label}),
else O. Output: labels CSV with clip,t,x1..y2,label,how.
"""
import argparse, csv, json
from pathlib import Path
import cv2, numpy as np

VEHICLE = {"wien", "gau", "chelyabinsk", "montreal", "sanjose", "us_blinkred", "tw_flicker", "toronto_pxo", "faulty"}
EXCLUDE = {"tw_kaohsiung"}  # mixed bicycle/pedestrian heads, not labelled


def lit(c):
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV); h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = v > 150
    red = bright & (s > 90) & ((h < 12) | (h > 165))
    amber = bright & (s > 90) & (h >= 12) & (h < 35)
    green = bright & (s > 60) & (h >= 45) & (h < 100)
    white = (v > 200) & (s < 50)
    return {k: float(m.mean()) for k, m in dict(red=red, amber=amber, green=green, white=white).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/datasets/commons_video")
    ap.add_argument("--fixes", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    D = Path(a.dir)
    fixes = json.load(open(a.fixes)) if a.fixes else {}
    out = []
    per = {}
    for r in csv.DictReader(open(D / "boxes.csv")):
        if r["clip"] in EXCLUDE:
            continue
        name = r["name"]
        c = cv2.imread(str(D / "crops" / (name + ".jpg")))
        w, h = int(r["x2"]) - int(r["x1"]), int(r["y2"]) - int(r["y1"])
        L = lit(c)
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        housing = float(((hsv[..., 0] >= 15) & (hsv[..., 0] < 35) & (hsv[..., 1] > 90) & (hsv[..., 2] > 100)).mean())
        if r["clip"] in VEHICLE or h / max(w, 1) > 2.6:
            y = "O"
        elif r["clip"] == "toronto_diag" and L["green"] > 0.005:  # green arrows (vehicle)
            y = "O"
        elif r["clip"] == "crossing_fr" and L["white"] > 0.03 and L["red"] < 0.005:
            teal = float(((hsv[..., 0] >= 65) & (hsv[..., 0] < 100) & (hsv[..., 1] > 80) & (hsv[..., 2] > 50)).mean())
            y = "W" if teal > 0.2 else "O"  # white man on a green-backlit panel
        elif r["clip"] == "crossing_fr":
            y = "S" if L["red"] > 0.01 else "O"
        elif r["clip"] == "china":  # only round vehicle lamps + red-man/digit heads here
            y = "S" if L["red"] > 0.01 and L["red"] > L["green"] else "O"
        elif L["green"] > (0.12 if r["clip"] == "crossing_fr" else 0.008) and L["green"] > L["red"]:
            y = "W"
            if r["clip"] in ("tw_chaozhou", "tw_pingtung") and lit(c[: c.shape[0] // 2])["green"] > 0.01:
                y = "C"
        elif r["clip"] == "toronto_diag" and L["white"] > 0.03 and housing < 0.2 and L["green"] < 0.005:
            y = "W"
        elif L["red"] > 0.01 or (L["amber"] > 0.02 and r["clip"] == "toronto_diag"):
            y = "S"
            if r["clip"] == "toronto_diag" and w / max(h, 1) >= 0.75:
                y = "C"
        else:
            y = "O"
        how = "auto"
        if name in fixes:
            y, how = fixes[name], "manual"
        if y == "X":
            continue
        out.append(dict(name=name, clip=r["clip"], t=r["t"], x1=r["x1"], y1=r["y1"], x2=r["x2"], y2=r["y2"], label=y, how=how))
    w = csv.DictWriter(open(a.out or D / "labels.csv", "w"), fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    import collections
    print(collections.Counter((o["clip"], o["label"]) for o in out))


if __name__ == "__main__":
    main()
