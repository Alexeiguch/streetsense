"""Validation-only walk-threshold sweep for signal_cnn v4_ptld (own val split + PTLD valid). Never reads any test split.

  STREETSENSE_SAM_FALLBACK=0 .venv/bin/python scripts/sweep_signal_cnn_threshold_val.py  -> /workspace/sr_logs/eval7_v4_val_sweep.{json,log}
"""
import csv, json, pickle, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path("/workspace/meta-hackathon"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from streetsense.signal_cnn import SignalCNN
from streetsense.traffic_light import apply_signal_cnn, classify_light, figure_walk_guard, signal_kind
from streetsense.types import Box
from train_signal_cnn import tight_box
items = []  # (group, y_is_walk, crop, kind, hsv, frame, box)
for r in pickle.load(open("/workspace/datasets/crops/signal_ds_v3.pkl", "rb")):
    if r["split"] != "val": continue
    x1, y1, x2, y2 = tight_box(r); c = r["crop"][y1:y2, x1:x2]; k = signal_kind(x2 - x1, y2 - y1)
    items.append(("own_val", r["y"] == 0, c, k, classify_light(c, k), r["crop"], Box(x1, y1, x2, y2)))
root = Path("/workspace/datasets/ptld"); cache = {}
for r in csv.DictReader(open(root / "boxes.csv")):
    if r["split"] != "valid": continue
    assert r["split"] != "test"
    if r["file"] not in cache: cache.clear(); cache[r["file"]] = cv2.imread(str(root / r["file"]))
    im = cache[r["file"]]; x1, y1, x2, y2 = (int(r[q]) for q in ("x1", "y1", "x2", "y2")); c = im[y1:y2, x1:x2].copy()
    k = signal_kind(x2 - x1, y2 - y1)
    items.append(("ptld_valid", r["label"] == "walk", c, k, classify_light(c, k), im, Box(x1, y1, x2, y2)))
res = {}
for name, m in (("v1", "models/signal_cnn"), ("v4", "models/signal_cnn_v4_ptld")):
    net = SignalCNN(ROOT / m); P = [net(it[2]).probs for it in items]
    nonwalk = [p[0] for it, p in zip(items, P) if not it[1]]
    res[name] = {"own_t_walk": net.t_walk, "max_nonwalk_pw": float(max(nonwalk)),
                 "max_nonwalk_pw_by_group": {g: float(max(p[0] for it, p in zip(items, P) if it[0] == g and not it[1])) for g in ("own_val", "ptld_valid")}}
    ts = [net.t_walk] if name == "v1" else sorted(set([round(x, 3) for x in np.arange(0.70, 0.995, 0.01)] + [net.t_walk]))
    sw = {}
    for t in ts:
        row = {}
        for g in ("own_val", "ptld_valid", "ALL"):
            sel = [(it, p) for it, p in zip(items, P) if g == "ALL" or it[0] == g]
            rfw = rw = cfw = nw = 0
            for it, p in sel:
                _, isw, c, k, hr, fr, b = it
                rd = figure_walk_guard(apply_signal_cnn(hr, p, k, t, net.t_veto), fr, b)
                if isw: nw += 1; rw += rd.state == "walk"
                else: rfw += rd.state == "walk"; cfw += p[0] >= t
            row[g] = dict(walk=int(rw), walk_n=int(nw), reader_fw=int(rfw), cnn_fw=int(cfw))
        sw[str(round(t, 4))] = row
    res[name]["sweep"] = sw
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "sweep"} for k, v in res.items()}, indent=1))
for name in res:
    for t, row in res[name]["sweep"].items():
        print(name, t, " ".join(f"{g}: walk {v['walk']}/{v['walk_n']} rfw {v['reader_fw']} cfw {v['cnn_fw']}" for g, v in row.items()))
json.dump(res, open("/workspace/sr_logs/eval7_v4_val_sweep.json", "w"), indent=1)
