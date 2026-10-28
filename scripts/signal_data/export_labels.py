#!/usr/bin/env python3
"""One-off: write eval/signal_labels.csv.gz + eval/signal_commons_manifest.csv from the
labelled crop pickles (provenance record; build_crops.py rebuilds the crops from it).

Box coordinates are in the source image: PTL = full-resolution 4032x3024 photo,
Commons = the 1280 px-wide thumbnail fetched by fetch_commons.py, dev = the eval clip frame.
"""
import csv, gzip, json, pickle, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/datasets/crops")
ds = pickle.load(open(D / "signal_ds.pkl", "rb"))
P = pickle.load(open(D / "ptl/ptl.pkl", "rb"))
C = pickle.load(open(D / "commons/commons.pkl", "rb"))
V = {o["id"]: o for o in pickle.load(open(D / "dev.pkl", "rb"))}
fix = json.load(open(D / "fixes.json")) if (D / "fixes.json").is_file() else {}
man = set(json.load(open(D / "review_order.json"))) | set(json.load(open(D / "review_order2.json")))
src = {f"p{i}": o for i, o in enumerate(P)} | {f"c{i}": o for i, o in enumerate(C)}
rows = []
for r in ds:
    u = r["uid"]
    if r["src"].startswith("dev"):
        o = V[u]; key, t, how = o["clip"], o["t"], "rule+eye"
    else:
        o = src[u]; key = o.get("file") or o["title"]; t = ""
        how = "manual-fix" if u in fix else "manual" if u in man else "auto"
    x1, y1, x2, y2 = (int(v) for v in o["box"])
    rows.append([r["src"].split(":")[0], key, t, r["split"], x1, y1, x2, y2, "WSCO"[r["y"]], how])
out = ROOT / "eval/signal_labels.csv.gz"
with gzip.open(out, "wt", newline="") as f:
    w = csv.writer(f); w.writerow(["src", "key", "t", "split", "x1", "y1", "x2", "y2", "label", "how"]); w.writerows(rows)
used = sorted({o["title"] for o in C})
with open(ROOT / "eval/signal_commons_manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["title", "licence", "artist", "page"])
    first = {}
    for o in C:
        first.setdefault(o["title"], o)
    for t in used:
        o = first[t]; w.writerow([t, o["lic"], o["artist"], o["page"]])
print(len(rows), "rows;", len(used), "commons files")
