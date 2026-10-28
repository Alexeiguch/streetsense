#!/usr/bin/env bash
# Fetch the public training data for the signal crop classifier and rebuild the crop set.
# Nothing here is committed: data goes to $DATA (default /workspace/datasets, outside git).
# Sources + licences: eval/SOURCES.md. PTL full-res zips are ~23 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-/workspace/datasets}
PY=${PY:-.venv/bin/python}
mkdir -p "$DATA/ptl" "$DATA/commons/img" "$DATA/crops"
# PTL dataset (samuelyu2002/PTL-Dataset, MIT): labels from GitHub, images from the authors' Google Drive.
for f in training_file.csv validation_file.csv testing_file.csv LICENSE; do
  [ -f "$DATA/ptl/$f" ] || curl -fsSL -o "$DATA/ptl/$f" "https://raw.githubusercontent.com/samuelyu2002/PTL-Dataset/master/$f" || true
done
command -v gdown >/dev/null || $PY -m pip install -q gdown
[ -f "$DATA/ptl/full1.zip" ] || gdown -O "$DATA/ptl/full1.zip" 13bTo9IO7aYXKtEpzyhsNd6csJpVhA8lE
[ -f "$DATA/ptl/full2.zip" ] || gdown -O "$DATA/ptl/full2.zip" 1JDXCwaFHJG3Uh_xHyMHJHImO6UiP-Yc2
# Wikimedia Commons (per-file CC/PD licences in eval/signal_commons_manifest.csv)
$PY scripts/signal_data/fetch_commons.py "$DATA/commons/img"
# Crops from eval/signal_labels.csv.gz (dev clips need samples/ from scripts/get_*; held-out clips never used)
$PY scripts/signal_data/build_crops.py --data "$DATA" --out "$DATA/crops/signal_ds.pkl"
echo "train: $PY scripts/train_signal_cnn.py --data $DATA/crops/signal_ds.pkl"
