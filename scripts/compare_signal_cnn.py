#!/usr/bin/env python3
"""Compare signal CNN versions on the crop test split of a signal_ds pickle (each at its own t_walk).

  .venv/bin/python scripts/compare_signal_cnn.py --data /workspace/datasets/crops/signal_ds_v2.pkl \
      --models models/signal_cnn models/signal_cnn_v2 [--out docs/benchmark/signal_cnn_compare.json]
"""
import argparse, collections, json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from streetsense.signal_cnn import CLASSES, SignalCNN  # noqa: E402
from train_signal_cnn import tight_box  # noqa: E402


def metrics(P, y, t):
    walk = P[:, 0] >= t; pred = P.argmax(1)
    tp = int((walk & (y == 0)).sum()); fw = int((walk & (y != 0)).sum())
    pc = {CLASSES[k]: (round(float((pred[y == k] == k).mean()), 3), int((y == k).sum())) for k in range(4) if (y == k).any()}
    return dict(n=int(len(y)), acc=round(float((pred == y).mean()), 3), walk_n=int((y == 0).sum()),
                walk_recall=round(tp / max(1, (y == 0).sum()), 3), false_walk=fw,
                walk_precision=round(tp / (tp + fw), 3) if tp + fw else None, per_class_acc=pc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/datasets/crops/signal_ds_v2.pkl")
    ap.add_argument("--models", nargs="+", default=["models/signal_cnn", "models/signal_cnn_v2"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    D = [r for r in pickle.load(open(a.data, "rb")) if r["split"] == a.split]
    groups = collections.defaultdict(list)
    for r in D:
        s = r["src"]; groups[s.split(":")[0]].append(r)
        if s.startswith("cvid:"):
            groups[s].append(r)
    res = {}
    for m in a.models:
        net = SignalCNN(ROOT / m)
        P = {id(r): np.array(net(r["crop"][tight_box(r)[1]:tight_box(r)[3], tight_box(r)[0]:tight_box(r)[2]]).probs) for r in D}
        res[m] = {"t_walk": net.t_walk}
        for g, rs in sorted(groups.items()):
            res[m][g] = metrics(np.stack([P[id(r)] for r in rs]), np.array([r["y"] for r in rs]), net.t_walk)
        print(m, "t_walk", round(net.t_walk, 3))
        for g, v in res[m].items():
            if g != "t_walk":
                print(f"  {g:22s} n={v['n']:4d} acc={v['acc']:.3f} walk {v['walk_n']:3d} rec={v['walk_recall']:.3f} fw={v['false_walk']} prec={v['walk_precision']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
