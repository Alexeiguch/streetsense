"""Where PTLD-test walk heads get lost in the light reader (evaluation only, no training).

Uses GT walk boxes as the detected box (pedsig_ptld finds 215/217 walk heads at IoU >= 0.3).
Per crop: height bin, HSV rule state, CNN argmax / p_walk vs t_walk, final reader state.
One CNN version per process: STREETSENSE_SIGNAL_CNN_VERSION=v1|v5_ptld_hn.
"""
import collections, csv, json, os, sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from streetsense import signal_cnn  # noqa: E402
from streetsense.traffic_light import apply_signal_cnn, classify_light, figure_walk_guard, signal_kind  # noqa: E402
from streetsense.types import Box  # noqa: E402

root = Path("/workspace/datasets/ptld")
rows = [r for r in csv.DictReader(open(root / "boxes.csv")) if r["split"] == "test" and r["label"] == "walk"]
cnn = signal_cnn.get(); ver = os.environ.get("STREETSENSE_SIGNAL_CNN_VERSION", "v1")
by = collections.defaultdict(collections.Counter)
for r in rows:
    img = cv2.imread(str(root / r["file"])); x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
    crop = img[y1:y2, x1:x2]; kind = signal_kind(x2 - x1, y2 - y1); h = y2 - y1
    hb = "h<40" if h < 40 else "h40-80" if h < 80 else "h80-160" if h < 160 else "h>=160"
    hr = classify_light(crop, kind); p = cnn(crop)
    top = signal_cnn.CLASSES[max(range(4), key=lambda k: p.probs[k])]
    cn = "walk>=t" if p.p_walk >= cnn.t_walk else ("walk<t" if top == "walk" else top)
    rd = figure_walk_guard(apply_signal_cnn(hr, p.probs, kind, cnn.t_walk, cnn.t_veto), img, Box(x1, y1, x2, y2))
    for k in ("all", hb):
        by[k]["n"] += 1; by[k]["final_" + rd.state] += 1; by[k]["hsv_" + hr.state] += 1; by[k]["cnn_" + cn] += 1
out = {"version": ver, "t_walk": cnn.t_walk, "bins": {k: dict(v) for k, v in by.items()}}
print(json.dumps(out, indent=1))
