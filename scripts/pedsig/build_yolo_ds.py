"""Build the YOLO fine-tuning set for the opt-in pedestrian-signal detector.

Writes to /workspace/datasets/pedsig/yolo (gitignored, outside the repo):
images/{train,val}/<id>.jpg are symlinks, labels/... are YOLO txt files.

Per image the labels are:
  * class 9 "traffic light": every GT signal box (PTLD train/valid GT, Commons GT),
    plus teacher (stock YOLO11s) traffic lights >= TEACHER_TL that do not overlap GT
    (IoU < 0.3), so vehicle lights are not taught as background,
  * class 9 also for SAM 3.1 boxes that Muse Spark verified "yes" on frames without
    full GT (Commons stills without a teacher light, Commons-video train frames),
    when they do not overlap an existing light box (IoU < 0.3),
  * every other COCO class from the teacher at conf >= TEACHER_OTHER (anti-forgetting).
Val = PTLD valid built the same way (checkpoints are picked on its GT signal recall).
Never uses the PTLD test split, the own crop test split, dev clips or held-out clips.
"""
import argparse, hashlib, json, os, sys
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/datasets/pedsig")
TEACHER_TL, TEACHER_OTHER = 0.5, 0.4


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "yolo"))
    ap.add_argument("--labels", nargs="*", default=[str(ROOT / "commons_nolight.jsonl"), str(ROOT / "cvid_all.jsonl")])
    a = ap.parse_args()
    from ultralytics import YOLO
    names = YOLO("models/yolo11s.pt").names
    cls_id = {v: k for k, v in names.items()}
    TL = cls_id["traffic light"]
    frames = json.load(open(ROOT / "frames.json"))
    agreed = {}
    for lp in a.labels:
        if not os.path.exists(lp):
            sys.exit(f"missing {lp}")
        for line in open(lp):
            r = json.loads(line)
            agreed[r["path"]] = [b["box"] for b in r["boxes"] if b["muse"] == "yes"]
    out = Path(a.out); stats = Counter()
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True); (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    for f in frames:
        if f["src"] == "commons" and f["split"] == "val":
            continue  # keep commons val out of training and selection
        sp = "val" if f["split"] == "val" else "train"
        W, H = f["wh"]
        gt = [g[:4] for g in (f["gt"] or [])]
        lights = list(gt); stats[f"{sp}_gt"] += len(gt)
        for t in f["teacher"]:
            if t[4] == "traffic light" and t[5] >= TEACHER_TL and all(iou(t[:4], g) < 0.3 for g in gt):
                lights.append(t[:4]); stats[f"{sp}_teacher_tl"] += 1
        if f["src"] != "ptld":
            for b in agreed.get(f["path"], []):
                if all(iou(b, l) < 0.3 for l in lights):
                    lights.append(b); stats[f"{sp}_sam_muse"] += 1
        rows = [(TL, b) for b in lights]
        rows += [(cls_id[t[4]], t[:4]) for t in f["teacher"] if t[4] != "traffic light" and t[5] >= TEACHER_OTHER]
        stem = hashlib.sha1(f["path"].encode()).hexdigest()[:16]
        img = out / "images" / sp / f"{stem}.jpg"
        if not img.exists():
            img.symlink_to(f["path"])
        lines = []
        for c, (x1, y1, x2, y2) in rows:
            x1, x2 = max(0, min(W, x1)), max(0, min(W, x2)); y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        (out / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[f"{sp}_images"] += 1; stats[f"{sp}_boxes"] += len(lines)
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {k}: {v}\n" for k, v in names.items()))
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()
