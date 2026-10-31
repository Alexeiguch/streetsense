#!/usr/bin/env python3
"""False-walk audit of signal CNN versions on our own labelled crop sets (no model API calls).

For every crop in a signal_ds pickle (all splits; train crops were seen by the models, so they are reported
separately), run the crop path of the local reader: classify_light (HSV) -> apply_signal_cnn (each model at its
own t_walk / t_veto) -> figure_walk_guard on the stored padded crop. Reports, per model and src/split group:
  reader_false_walk  non-walk crops (stop / countdown / other) the reader says walk on
  cnn_false_walk     non-walk crops with p_walk >= t_walk (CNN alone)
  walk_recall        walk crops the reader says walk on
  promotable_hi      non-walk crops HSV reads green/unknown (non-vehicle, no countdown) with p_walk >= 0.9
  vehicle_green      vehicle-shaped (h/w > 2.6) non-walk crops HSV reads green: n and max p_walk

  .venv/bin/python scripts/eval_signal_cnn_falsewalk.py --models models/signal_cnn models/signal_cnn_v4_ptld \
      --data /workspace/datasets/crops/signal_ds_v3.pkl --out x.json
"""
import argparse, collections, json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from streetsense.signal_cnn import SignalCNN  # noqa: E402
from streetsense.traffic_light import apply_signal_cnn, classify_light, figure_walk_guard, signal_kind  # noqa: E402
from streetsense.types import Box  # noqa: E402
from train_signal_cnn import tight_box  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/datasets/crops/signal_ds_v3.pkl")
    ap.add_argument("--models", nargs="+", default=["models/signal_cnn", "models/signal_cnn_v4_ptld"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    D = [r for r in pickle.load(open(a.data, "rb")) if not r["src"].startswith("ptld")]  # PTLD scored by eval_ptld.py
    pre = []
    for r in D:
        x1, y1, x2, y2 = tight_box(r); c = r["crop"][y1:y2, x1:x2]
        kind = signal_kind(x2 - x1, y2 - y1)
        pre.append((c, kind, classify_light(c, kind), Box(x1, y1, x2, y2)))
    res = {"data": a.data}
    for m in a.models:
        net = SignalCNN(ROOT / m)
        g = collections.defaultdict(lambda: collections.Counter()); pw_max = collections.defaultdict(float)
        for r, (c, kind, hr, box) in zip(D, pre):
            p = net(c); pw = p.p_walk
            rd = figure_walk_guard(apply_signal_cnn(hr, p.probs, kind, net.t_walk, net.t_veto), r["crop"], box)
            keys = [f"{r['split']}:ALL", f"{r['split']}:{r['src'].split(':')[0]}"]
            for k in keys:
                G = g[k]; G["n"] += 1
                if r["y"] == 0:
                    G["walk_n"] += 1; G["walk_recall"] += rd.state == "walk"
                else:
                    G["nonwalk_n"] += 1; G["reader_false_walk"] += rd.state == "walk"
                    G["cnn_false_walk"] += pw >= net.t_walk
                    G["promotable_hi"] += (kind != "vehicle" and hr.state in ("green", "unknown") and not hr.countdown and pw >= 0.9)
                    pw_max[k] = max(pw_max[k], pw)
                    if kind == "vehicle" and hr.state == "green":
                        G["vehicle_green_n"] += 1; pw_max[k + "|vehicle_green"] = max(pw_max[k + "|vehicle_green"], pw)
        out = {"t_walk": net.t_walk, "t_veto": net.t_veto}
        for k in sorted(g):
            out[k] = dict(g[k]); out[k]["nonwalk_max_p_walk"] = round(pw_max[k], 4)
            if g[k]["vehicle_green_n"]:
                out[k]["vehicle_green_max_p_walk"] = round(pw_max[k + "|vehicle_green"], 4)
        res[m] = out
        print(m, "t_walk", round(net.t_walk, 4))
        for k, v in out.items():
            if isinstance(v, dict):
                print(f"  {k:18s} n={v['n']:5d} walk {v.get('walk_recall', 0)}/{v.get('walk_n', 0)} reader_fw={v.get('reader_false_walk', 0)}"
                      f" cnn_fw={v.get('cnn_false_walk', 0)} promotable>=0.9={v.get('promotable_hi', 0)} max_pw_nonwalk={v['nonwalk_max_p_walk']}"
                      f" veh_green={v.get('vehicle_green_n', 0)} max={v.get('vehicle_green_max_p_walk')}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
