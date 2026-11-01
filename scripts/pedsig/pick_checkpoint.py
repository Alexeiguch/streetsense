"""Pick the pedsig detector checkpoint on PTLD VALID only (never test).

For each checkpoint: full-frame predict at conf 0.35 (the pipeline floor), imgsz 640, and count
PTLD-valid GT signal boxes matched by a traffic-light detection at IoU >= 0.3 (all / walk),
plus traffic-light detections that match no GT (a precision proxy; PTLD GT only labels
pedestrian heads, so some of these are real vehicle lights). Score = GT recall.

  python scripts/pedsig/pick_checkpoint.py /workspace/sr_logs/pedsig_yolo/yolo11s_pedsig/weights/epoch*.pt models/yolo11s.pt
"""
import json, os, re, sys
from ultralytics import YOLO

sys.path.insert(0, "scripts/pedsig")
from build_yolo_ds import iou  # noqa: E402

CACHE = {}
frames = [f for f in json.load(open("/workspace/datasets/pedsig/frames.json")) if f["src"] == "ptld" and f["split"] == "val"]
# PTLD is split by video, but most valid videos were shot on the same dates as train videos
# (same walks, often the same crossings). "new_date" = valid images whose date is not in train.
def seq(p):
    return re.sub(r"-\(F\d+\)", "", os.path.splitext(os.path.basename(p))[0])


train_dates = {seq(f)[:8] for f in json.load(open("/workspace/datasets/pedsig/frames.json")) if f["src"] == "ptld" and f["split"] == "train"}
IOUS = [float(x) for x in os.environ.get("PICK_IOUS", "0.3").split(",")]
res = {}
for w in [x for x in sys.argv[1:] if not x.startswith("--")]:
    m = YOLO(w); out = {}
    for th in IOUS:
        for grp in ("all", "new_date"):
            hit = walk_hit = n = walk_n = extra = 0
            for f in frames:
                if grp == "new_date" and seq(f["path"])[:8] in train_dates:
                    continue
                key = (w, f["path"])
                if key not in CACHE:
                    r = m.predict(f["path"], imgsz=640, conf=0.35, device="cpu", verbose=False)[0]
                    CACHE[key] = [b.tolist() for b, c in zip(r.boxes.xyxy, r.boxes.cls) if m.names[int(c)] == "traffic light"]
                tl = CACHE[key]
                for g in f["gt"]:
                    ok = any(iou(g[:4], d) >= th for d in tl); n += 1; hit += ok
                    if g[4] == "walk":
                        walk_n += 1; walk_hit += ok
                extra += sum(1 for d in tl if all(iou(g[:4], d) < th for g in f["gt"]))
            out[f"iou{th}_{grp}"] = {"gt_recall": round(hit / n, 4), "gt_hit": hit, "gt": n, "walk_hit": walk_hit, "walk_gt": walk_n, "unmatched_tl": extra}
    res[w] = out
    print(w, json.dumps(out), flush=True)
json.dump(res, open(os.environ.get("PICK_OUT", "/workspace/sr_logs/pedsig_pick.json"), "w"), indent=1)
