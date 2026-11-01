"""Per-crop reader features for tuning reader rules offline (no model API calls).

Rows: own crop sets (signal_ds_v3.pkl, PTLD excluded; every split, tagged) and PTLD GT crops
(boxes.csv, --ptld-splits). Each row: src, split, label (walk/stop/countdown/other), crop h/w, kind,
HSV state + countdown flag + figure, and CNN probs for every --models dir. Tuning must use val rows
only; test rows (own crop test, PTLD test) are for the final report.

  .venv/bin/python scripts/reader_fix/crop_features.py --ptld-splits valid --out /workspace/sr_logs/reader_fix/feat_val.jsonl
"""
import argparse, csv, json, pickle, sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from streetsense.signal_cnn import CLASSES, SignalCNN  # noqa: E402
from streetsense.traffic_light import classify_light, signal_kind  # noqa: E402
from train_signal_cnn import tight_box  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", default="/workspace/datasets/crops/signal_ds_v3.pkl")
    ap.add_argument("--own-splits", default="val")
    ap.add_argument("--ptld-splits", default="valid")
    ap.add_argument("--models", nargs="+", default=["models/signal_cnn", "models/signal_cnn_v5_ptld_hn"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cnns = {Path(m).name: SignalCNN(m) for m in a.models}
    out = open(a.out, "w")

    def emit(src, split, label, c, kind, frame=None, box=None, extra=None):
        hr = classify_light(c, kind)
        row = dict(src=src, split=split, label=label, h=int(c.shape[0]), w=int(c.shape[1]), kind=kind,
                   hsv=hr.state, cd=bool(hr.countdown), fig=bool(hr.figure), probs={k: [round(v, 5) for v in m(c).probs] for k, m in cnns.items()})
        if extra:
            row.update(extra)
        out.write(json.dumps(row) + "\n")

    own_splits = set(a.own_splits.split(",")) if a.own_splits else set()
    if own_splits:
        for r in pickle.load(open(a.own, "rb")):
            if r["src"].startswith("ptld") or r["split"] not in own_splits:
                continue
            x1, y1, x2, y2 = tight_box(r); c = r["crop"][y1:y2, x1:x2]
            emit(r["src"].split(":")[0], r["split"], CLASSES[r["y"]], c, signal_kind(x2 - x1, y2 - y1), extra={"uid": r.get("uid")})
    root = Path("/workspace/datasets/ptld")
    sp = set(a.ptld_splits.split(",")) if a.ptld_splits else set()
    rows = [r for r in csv.DictReader(open(root / "boxes.csv")) if r["split"] in sp]
    cache = {}
    for r in rows:
        f = r["file"]
        if f not in cache:
            cache.clear(); cache[f] = cv2.imread(str(root / f))
        img = cache[f]; x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
        lab = {"walk": "walk", "stop": "stop", "other": "other"}[r["label"]]
        emit("ptld", r["split"], lab, img[y1:y2, x1:x2], signal_kind(x2 - x1, y2 - y1), extra={"file": f, "box": [x1, y1, x2, y2]})
    out.close()


if __name__ == "__main__":
    main()
