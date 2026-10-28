#!/usr/bin/env python3
"""Train the tiny traffic-light crop classifier (streetsense/signal_cnn.py) on CPU.

Input: a crop dataset pickle built by scripts/signal_data/build_dataset.py (not committed; see
eval/SOURCES.md and scripts/get_signal_data.sh). Held-out eval clips are never in it.
Output: models/signal_cnn/signal_cnn.onnx + signal_cnn.json (classes, threshold, metrics).

  .venv/bin/python scripts/train_signal_cnn.py --data datasets/crops/signal_ds.pkl
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from streetsense.signal_cnn import CLASSES, SIZE, preprocess  # noqa: E402


def tight_box(r):
    """Tight light box inside the stored (padded, maybe downscaled) crop."""
    h, w = r["crop"].shape[:2]
    l, t, rr, b = (v * r["scale"] for v in r["pad"])
    return int(l), int(t), max(int(l) + 1, int(w - rr)), max(int(t) + 1, int(h - b))


def augment(r, rng):
    c = r["crop"]
    H, W = c.shape[:2]
    x1, y1, x2, y2 = tight_box(r)
    bw, bh = x2 - x1, y2 - y1
    m = [rng.uniform(-0.06, 0.12) for _ in range(4)]
    X1 = int(np.clip(x1 - m[0] * bw, 0, W - 1)); Y1 = int(np.clip(y1 - m[1] * bh, 0, H - 1))
    X2 = int(np.clip(x2 + m[2] * bw, X1 + 1, W)); Y2 = int(np.clip(y2 + m[3] * bh, Y1 + 1, H))
    im = c[Y1:Y2, X1:X2].astype(np.float32)
    if rng.random() < 0.5:
        im = im[:, ::-1]
    # brightness / contrast / gamma (no big hue shifts: they would swap red and green)
    im = im * rng.uniform(0.6, 1.4) + rng.uniform(-25, 25)
    im = np.clip(im, 0, 255)
    if rng.random() < 0.3:
        im = 255 * (im / 255) ** rng.uniform(0.7, 1.5)
    im = im.astype(np.uint8)
    if rng.random() < 0.4:  # low-resolution simulation (distant heads)
        h, w = im.shape[:2]
        s = rng.uniform(0.15, 0.6)
        if min(h, w) * s >= 3:
            im = cv2.resize(cv2.resize(im, (max(2, int(w * s)), max(2, int(h * s))), interpolation=cv2.INTER_AREA),
                            (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.25:
        k = rng.choice([3, 5])
        im = cv2.GaussianBlur(im, (k, k), 0)
    if rng.random() < 0.3:  # sensor noise
        im = np.clip(im.astype(np.int16) + np.random.randint(-12, 13, im.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # jpeg artefacts
        ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(30, 80))])
        if ok:
            im = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return preprocess(np.ascontiguousarray(im))


def plain(r):
    x1, y1, x2, y2 = tight_box(r)
    return preprocess(r["crop"][y1:y2, x1:x2])


class TinyNet(nn.Module):
    def __init__(self, n=4, w=(24, 48, 96, 128)):
        super().__init__()
        layers, cin = [], 3
        for i, c in enumerate(w):
            layers += [nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            if i < len(w) - 1:
                layers.append(nn.MaxPool2d(2))
            cin = c
        self.f = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(cin, n)

    def forward(self, x):
        x = self.f(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(self.drop(x))


def build(arch):
    if arch == "tiny":
        return TinyNet()
    import torchvision
    m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 4)
    return m


@torch.no_grad()
def predict(model, X, bs=512):
    model.eval()
    out = [F.softmax(model(torch.from_numpy(X[i:i + bs])), 1).numpy() for i in range(0, len(X), bs)]
    return np.concatenate(out) if out else np.zeros((0, 4))


def report(name, P, y, src, t_walk):
    pred = P.argmax(1)
    res = {"n": int(len(y)), "acc": float((pred == y).mean()) if len(y) else None}
    walk = P[:, 0] >= t_walk
    res["walk_recall_at_T"] = float(walk[y == 0].mean()) if (y == 0).any() else None
    res["false_walk_at_T"] = int((walk & (y != 0)).sum())
    res["per_src"] = {}
    for s in sorted(set(src)):
        k = np.array([q == s for q in src])
        res["per_src"][s] = {"n": int(k.sum()), "acc": float((pred[k] == y[k]).mean()),
                             "walk_n": int((y[k] == 0).sum()),
                             "walk_recall_at_T": float(walk[k & (y == 0)].mean()) if (k & (y == 0)).any() else None,
                             "false_walk_at_T": int((walk & k & (y != 0)).sum())}
    cm = np.zeros((4, 4), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    res["confusion_rows_true"] = cm.tolist()
    print(f"[{name}] n={res['n']} acc={res['acc']:.3f} walk_recall@T={res['walk_recall_at_T']} false_walk@T={res['false_walk_at_T']}")
    for s, v in res["per_src"].items():
        print(f"   {s:14s} n={v['n']:5d} acc={v['acc']:.3f} walk={v['walk_n']} rec@T={v['walk_recall_at_T']} fw@T={v['false_walk_at_T']}")
    print("   confusion (rows true W,S,C,O):", cm.tolist())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/datasets/crops/signal_ds.pkl")
    ap.add_argument("--arch", default="tiny", choices=["tiny", "mnv3"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ckpt", default="", help="save best state here each time it improves")
    ap.add_argument("--out", default=str(ROOT / "models/signal_cnn"))
    ap.add_argument("--margin", type=float, default=0.02, help="added to max non-walk p(walk) on val")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng = random.Random(a.seed)
    torch.set_num_threads(a.threads)
    D = pickle.load(open(a.data, "rb"))
    tr = [r for r in D if r["split"] == "train"]
    va = [r for r in D if r["split"] == "val"]
    te = [r for r in D if r["split"] == "test"]
    print("train", collections.Counter(CLASSES[r["y"]] for r in tr), "val", len(va), "test", len(te))
    # class- and source-balanced sampling weights
    cnt = collections.Counter((r["src"].split(":")[0], r["y"]) for r in tr)
    wts = np.array([1.0 / cnt[(r["src"].split(":")[0], r["y"])] for r in tr])
    boost = {"dev": 2.0}
    wts *= np.array([boost.get(r["src"].split(":")[0], 1.0) for r in tr])
    wts /= wts.sum()
    model = build(a.arch)
    print("params", sum(p.numel() for p in model.parameters()))
    steps = max(1, len(tr) // a.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps)
    Xva = np.stack([plain(r) for r in va]); yva = np.array([r["y"] for r in va])
    best, best_state = -1, None
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0
        for _ in range(steps):
            idx = np.random.choice(len(tr), a.bs, p=wts)
            x = torch.from_numpy(np.stack([augment(tr[i], rng) for i in idx]))
            y = torch.tensor([tr[i]["y"] for i in idx])
            loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss.detach())
        P = predict(model, Xva)
        acc = float((P.argmax(1) == yva).mean())
        # selection metric: walk recall at the zero-false-walk threshold + accuracy
        T = float(P[yva != 0, 0].max()) + 1e-6
        rec = float((P[yva == 0, 0] >= T).mean())
        score = rec + acc
        print(f"ep {ep} loss {tot / steps:.3f} val acc {acc:.3f} rec@0fw {rec:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
            if a.ckpt:
                Path(a.ckpt).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": ep, "score": score, "state": best_state, "arch": a.arch}, a.ckpt)
    model.load_state_dict(best_state)
    P = predict(model, Xva)
    t_walk = min(0.99, max(0.5, float(P[yva != 0, 0].max()) + a.margin))
    print("t_walk", t_walk)
    meta = {"classes": list(CLASSES), "size": SIZE, "arch": a.arch, "t_walk": t_walk, "t_veto": 0.8,
            "preprocess": "tight box, aspect letterbox on black to 64x64, BGR, x/255-0.5",
            "train_counts": dict(collections.Counter(CLASSES[r["y"]] for r in tr))}
    meta["val"] = report("val", P, yva, [r["src"].split(":")[0] for r in va], t_walk)
    Xte = np.stack([plain(r) for r in te]); yte = np.array([r["y"] for r in te])
    meta["test"] = report("test", predict(model, Xte), yte, [r["src"] for r in te], t_walk)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE), str(out / "signal_cnn.onnx"), input_names=["x"],
                      output_names=["logits"], dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17,
                      dynamo=False)
    (out / "signal_cnn.json").write_text(json.dumps(meta, indent=1))
    print("saved", out, (out / "signal_cnn.onnx").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
