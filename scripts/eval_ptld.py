#!/usr/bin/env python3
"""Test-only evaluation of the local light reader on PTLD (licence UNCONFIRMED: never train on it).

Needs boxes.csv from scripts/get_ptld_crops.py. No model API calls. Pick the CNN with
STREETSENSE_SIGNAL_CNN_VERSION=v1|v2|v3 (one version per run; the CNN instance is cached).

  crops : each GT box -> classify_light (HSV rules) -> apply_signal_cnn -> figure_walk_guard (as in
          Pipeline, without the detector/smoother). Also reports hand rules alone and CNN alone (p_walk >= t_walk).
  --images N : full test images through Pipeline (detector path, 5 repeats like a still in
          eval_local_reader) -> local_signal_state. Image label: walk if any go box, else stop if any
          stop box, else other if any off box (images with no box are 'none').

  STREETSENSE_SAM_FALLBACK=0 STREETSENSE_SIGNAL_CNN_VERSION=v2 python scripts/eval_ptld.py [--images 478] [--out x.json]
"""
import argparse, collections, csv, json, os, sys, time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streetsense import signal_cnn  # noqa: E402
from streetsense.traffic_light import apply_signal_cnn, classify_light, figure_walk_guard, signal_kind  # noqa: E402
from streetsense.types import Box  # noqa: E402


def summary(conf):
    """conf[label][pred_state] counts -> walk recall, false walk rate per non-walk label."""
    out = {}
    for lab, c in sorted(conf.items()):
        n = sum(c.values()); w = c.get("walk", 0)
        out[lab] = {"n": n, "said_walk": w, "rate": round(w / n, 3) if n else None, "states": dict(c)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/datasets/ptld"); ap.add_argument("--split", default="test")
    ap.add_argument("--images", type=int, default=0); ap.add_argument("--out", default="")
    a = ap.parse_args(); root = Path(a.root)
    rows = [r for r in csv.DictReader(open(root / "boxes.csv")) if r["split"] == a.split]
    cnn = signal_cnn.get(); ver = os.environ.get("STREETSENSE_SIGNAL_CNN_VERSION", "v1")
    res = {"version": ver, "t_walk": cnn.t_walk if cnn else None}
    full, rules, alone = (collections.defaultdict(collections.Counter) for _ in range(3))
    cache = {}
    for r in rows:
        f = r["file"]
        if f not in cache:
            cache.clear(); cache[f] = cv2.imread(str(root / f))
        img = cache[f]; x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
        crop = img[y1:y2, x1:x2]; kind = signal_kind(x2 - x1, y2 - y1)
        hr = classify_light(crop, kind); rules[r["label"]][hr.state] += 1
        p = cnn(crop) if cnn else None
        rd = apply_signal_cnn(hr, p.probs, kind, cnn.t_walk, cnn.t_veto) if p else hr
        rd = figure_walk_guard(rd, img, Box(x1, y1, x2, y2))
        full[r["label"]][rd.state] += 1
        if p:
            top = signal_cnn.CLASSES[max(range(4), key=lambda k: p.probs[k])]  # argmax walk below t_walk is NOT walk
            alone[r["label"]]["walk" if p.p_walk >= cnn.t_walk else ("argmax_walk_below_t" if top == "walk" else top)] += 1
    res["crops_reader"] = summary(full); res["crops_rules_only"] = summary(rules); res["crops_cnn_only"] = summary(alone)
    if a.images:
        from streetsense.answers import local_signal_state
        from streetsense.config import Config
        from streetsense.pipeline import Pipeline
        lab = {}
        for r in rows:
            o = {"walk": 0, "stop": 1, "other": 2}[r["label"]]
            lab[r["file"]] = min(lab.get(r["file"], 9), o)
        imgs = sorted((root / a.split / a.split).glob("*.txt"))[: a.images]
        conf = collections.defaultdict(collections.Counter); t0 = time.time()
        for t in imgs:
            ip = next(p for p in t.parent.glob(t.stem + ".*") if p.suffix.lower() != ".txt")
            rel = str(ip.relative_to(root)); L = {0: "walk", 1: "stop", 2: "other"}.get(lab.get(rel), "none")
            cfg = Config(); cfg.vlm.enabled = False; pipe = Pipeline(cfg, use_surface=False)
            im = cv2.imread(str(ip)); st = "unknown"
            for k in range(5):
                st = local_signal_state(pipe.process(im, annotate=False, t=float(k)), im.shape)
            conf[L][st] += 1
        res["images_pipeline"] = summary(conf); res["images_s"] = round(time.time() - t0, 1)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
