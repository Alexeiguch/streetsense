#!/usr/bin/env bash
# Fetch the Wikimedia Commons signal videos (eval/signal_video_clips.csv; per-file licences there and in
# eval/SOURCES.md) to $DATA/commons_video, mine light boxes, label, and append crops to the v2 crop set.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-/workspace/datasets}; PY=${PY:-.venv/bin/python}; D="$DATA/commons_video"
mkdir -p "$D"
cp eval/signal_video_clips.csv "$D/clips.csv"
$PY - "$D" <<'PY'
import csv, sys, urllib.request, os
d = sys.argv[1]
for r in csv.DictReader(open(f"{d}/clips.csv")):
    p = f"{d}/{r['file']}"
    if r["split"] != "excluded" and not os.path.exists(p):
        q = urllib.request.Request(r["url"], headers={"User-Agent": "StreetSense-research/0.1"})
        open(p, "wb").write(urllib.request.urlopen(q).read()); print("got", r["file"])
PY
# Boxes come from our detector (YOLO11s OpenVINO, imgsz 960, zoom 0.5 fallback, 2x3 tiles); labels from rules + eye fixes.
[ -f "$D/boxes.csv" ] || $PY scripts/signal_data/mine_video_lights.py --clips "$D/clips.csv"
$PY scripts/signal_data/label_video_crops.py --dir "$D" --fixes eval/signal_video_fixes.json
cmp -s "$D/labels.csv" eval/signal_video_labels.csv || echo "WARN: labels differ from eval/signal_video_labels.csv (detector version drift)"
$PY scripts/signal_data/add_video_crops.py --base "$DATA/crops/signal_ds.pkl" --dir "$D" --out "$DATA/crops/signal_ds_v2.pkl"
