#!/usr/bin/env python3
"""Step 1 of the pedestrian-signal labelling helper: where does the default detector miss a signal head?

Runs the default YOLO11s (OpenVINO, full frame, no tiles/zoom) on TRAIN/VAL-eligible frames only and stores,
per frame, the teacher detections (all COCO classes, conf >= 0.25) and any ground-truth signal boxes:
  ptld:train, ptld:valid   PTLD (licence UNCONFIRMED) GT boxes from boxes.csv. The test split is never read.
  commons:train|val        Commons stills; GT = boxes in eval/signal_labels.csv.gz (auto + manual). Images with
                           no label row are treated as train (no image of the commons test split is read).
  cvid:<clip>              Commons video clips of the signal_ds TRAIN split only, one frame every 2 s.
Dev clips (eval_local_reader dev set), samples/heldout and every test split are excluded.
Output: /workspace/datasets/pedsig/frames.json (gitignored dataset dir; contains PTLD-derived boxes).
"""
import collections, csv, gzip, hashlib, json, sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]
D = Path("/workspace/datasets")
OUT = D / "pedsig"
CVID_TRAIN = ["chelyabinsk", "crossing_fr", "dus_r2g", "faulty", "toronto_diag", "toronto_pxo", "wien"]


def main():
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "models/yolo11s_openvino_model"), task="detect")
    names = model.names
    frames = []
    # PTLD train/valid (never test)
    gt = collections.defaultdict(list)
    for r in csv.DictReader(open(D / "ptld/boxes.csv")):
        if r["split"] in ("train", "valid"):
            gt[r["file"]].append([int(r[k]) for k in ("x1", "y1", "x2", "y2")] + [r["label"]])
    for split in ("train", "valid"):
        for p in sorted((D / "ptld" / split / split).iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rel = str(p.relative_to(D / "ptld"))
                frames.append(dict(src="ptld", split="train" if split == "train" else "val", path=str(p), gt=gt.get(rel, [])))
    # Commons stills (labelled train/val + unlabelled; test images skipped)
    lab = collections.defaultdict(list); split_of = {}
    for r in csv.DictReader(gzip.open(ROOT / "eval/signal_labels.csv.gz", "rt")):
        if r["src"] != "commons":
            continue
        split_of[r["key"]] = r["split"]
        lab[r["key"]].append([int(r[k]) for k in ("x1", "y1", "x2", "y2")] + [r["label"]])
    info = json.load(open(D / "commons/info.json"))
    for key in sorted(info):
        sp = split_of.get(key, "train")
        if sp == "test":
            continue
        p = D / "commons/img" / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".jpg")
        if p.exists():
            frames.append(dict(src="commons", split=sp, path=str(p), key=key, gt=lab.get(key, []), gt_partial=key not in split_of))
    # Commons video, train clips, every 2 s
    vdir = OUT / "cvid_frames"; vdir.mkdir(parents=True, exist_ok=True)
    for clip in CVID_TRAIN:
        f = next((D / "commons_video").glob(clip + ".*"))
        cap = cv2.VideoCapture(str(f)); fps = cap.get(cv2.CAP_PROP_FPS) or 25; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(0, n, int(round(fps * 2))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, im = cap.read()
            if not ok:
                break
            p = vdir / f"{clip}_{i / fps:07.2f}.jpg"; cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            frames.append(dict(src="cvid:" + clip, split="train", path=str(p), gt=None))
    print(len(frames), collections.Counter((f["src"].split(":")[0], f["split"]) for f in frames), flush=True)
    for i, f in enumerate(frames):
        res = model.predict(f["path"], imgsz=640, conf=0.25, iou=0.5, device="cpu", verbose=False)[0]
        im_h, im_w = res.orig_shape
        f["wh"] = [int(im_w), int(im_h)]
        f["teacher"] = [[round(float(v), 1) for v in b] + [names[int(c)], round(float(p), 3)]
                        for b, c, p in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy())]
        if i % 500 == 0:
            print(i, flush=True)
    (OUT / "frames.json").write_text(json.dumps(frames))


if __name__ == "__main__":
    main()
