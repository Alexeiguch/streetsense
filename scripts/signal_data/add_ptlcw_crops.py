#!/usr/bin/env python3
"""PTL-Crosswalk-Base -> signal crops (licence UNCONFIRMED: opt-in models only, keep rollback-able).

Labels come from the authors' CSVs (ronaldosm/PedestrianTrafficLightsAndCrosswalkDetection, Datasets/Annotations/
PTL-Crosswalk Dataset/*.csv): light_class 0=red->stop, 1=green->walk, 2=none->other. The CSV gives a crosswalk
midline, not a light box, so the box comes from our detector (YOLO full + 2x3 tiles, NMS). A frame is kept only if
it has exactly ONE light >= --conf that is not vehicle-shaped (signal_kind != "vehicle"), so the image label can be
pinned to that box.

Split: the authors' folders ("training set" / "testing set", FR/ likewise) are separate recordings; we keep them.
Val = every 5th block of 50 frame numbers from train (block-level, so no neighbouring frames leak).
cond: NHT=night, RNY=rain, else day (FR = France).

  mine:  add_ptlcw_crops.py mine   (resumable; writes <root>/boxes.csv)
  build: add_ptlcw_crops.py build --base signal_ds_v2.pkl --out signal_ds_v3.pkl --manifest eval/ptlcw_manifest.csv
"""
import argparse, ast, collections, csv, pickle, re, sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
PT = Path("/workspace/datasets/ptlcw")
IMG = PT / "GIT - PTL-Crosswalk Dataset/base"
ANN = PT / "repo/Datasets/Annotations/PTL-Crosswalk Dataset"
LAB = {0: "S", 1: "W", 2: "O"}


def labels():
    out = {}
    for f in ("training.csv", "validation.csv", "testing.csv"):
        for r in csv.DictReader(open(ANN / f)):
            if r["image_file"].startswith("PTL-Crosswalk-Base/"):
                out[r["image_file"]] = LAB[int(float(r["light_class"]))]
    return out


def meta(rel):
    p = Path(rel); stem = p.stem
    m = re.match(r"(\d+)([A-Z]*)$", stem)
    num, suf = int(m.group(1)), m.group(2)
    fr = "/FR/" in rel
    split = "test" if "testing set" in rel else ("val" if (num // 50) % 5 == 4 else "train")
    cond = "night" if suf == "NHT" else "rain" if suf == "RNY" else "day"
    seq = f"{'FR' if fr else 'BR'}{suf or '-'}:{num // 50}"
    return split, cond, seq


def mine(a):
    from streetsense.config import DetectorConfig
    from streetsense.detector import YoloDetector, nms_lights
    from streetsense.traffic_light import signal_kind
    out = PT / "boxes.csv"
    done = {r["file"] for r in csv.DictReader(open(out))} if out.exists() else set()
    f = open(out, "a", newline=""); w = csv.writer(f)
    if not done:
        w.writerow(["file", "i", "x1", "y1", "x2", "y2", "conf", "kind"])
    det = YoloDetector(DetectorConfig(track=False, conf=0.25, imgsz=960))
    L = labels(); n = 0
    for rel in sorted(L):
        if rel in done:
            continue
        img = cv2.imread(str(IMG / rel))
        if img is None:
            w.writerow([rel, -1, 0, 0, 0, 0, 0, "missing"]); continue
        d = [x for x in det(img) if x.label == "traffic light"] + det.tiled_lights(img, (2, 3), 0.4)
        ds = nms_lights(d)
        if not ds:
            w.writerow([rel, -1, 0, 0, 0, 0, 0, "none"])
        for i, x in enumerate(ds):
            b = x.box
            w.writerow([rel, i, int(b.x1), int(b.y1), int(b.x2), int(b.y2), round(x.conf, 3), signal_kind(b.w, b.h)])
        n += 1
        if n % 50 == 0:
            f.flush(); print("mined", n + len(done), flush=True)
    f.close()


def build(a):
    from build_crops import CL, cut
    L = labels()
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(PT / "boxes.csv")):
        if int(r["i"]) >= 0:
            by[r["file"]].append(r)
    D = pickle.load(open(a.base, "rb")) if a.base else []
    n0 = len(D); man = []
    for rel, lab in sorted(L.items()):
        split, cond, seq = meta(rel)
        hi = [r for r in by.get(rel, []) if float(r["conf"]) >= a.conf]
        ped = [r for r in hi if r["kind"] != "vehicle"]
        why = "ok" if len(hi) == 1 and len(ped) == 1 else ("no_box" if not hi else "vehicle_shape" if not ped else "multi_box")
        row = dict(file=rel, label=lab, split=split, cond=cond, seq=seq, kept=int(why == "ok"), why=why,
                   x1="", y1="", x2="", y2="", conf="")
        if why == "ok":
            b = ped[0]; row.update({k: b[k] for k in ("x1", "y1", "x2", "y2", "conf")})
            img = cv2.imread(str(IMG / rel))
            c, pad, s = cut(img, *(int(b[k]) for k in ("x1", "y1", "x2", "y2")))
            D.append(dict(src=f"ptlcw:{cond}", grp=f"ptlcw:{seq}", split=split, y=CL[lab], crop=c, pad=pad, scale=s,
                          name=f"ptlcw_{Path(rel).stem}_{'FR' if '/FR/' in rel else 'BR'}"))
        man.append(row)
    w = csv.DictWriter(open(a.manifest, "w", newline=""), fieldnames=list(man[0])); w.writeheader(); w.writerows(man)
    print("frames", len(man), "kept", sum(r["kept"] for r in man), collections.Counter(r["why"] for r in man))
    print(collections.Counter((r["split"], r["cond"], r["label"]) for r in man if r["kept"]))
    if a.out:
        pickle.dump(D, open(a.out, "wb")); print("added", len(D) - n0, "->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mine", "build"])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--base", default="/workspace/datasets/crops/signal_ds_v2.pkl")
    ap.add_argument("--out", default="/workspace/datasets/crops/signal_ds_v3.pkl")
    ap.add_argument("--manifest", default=str(ROOT / "eval/ptlcw_manifest.csv"))
    a = ap.parse_args()
    mine(a) if a.cmd == "mine" else build(a)
