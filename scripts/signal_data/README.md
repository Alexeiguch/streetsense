# Signal crop data

- `../get_signal_data.sh` — fetch PTL + Commons, rebuild crops from `eval/signal_labels.csv.gz`.
- `build_crops.py` — labels CSV + raw images → `signal_ds.pkl` used by `scripts/train_signal_cnn.py`.
- `fetch_commons.py` — Commons thumbnails listed in `eval/signal_commons_manifest.csv`.
- `ptl_zip.py` — reader for the PTL full-res zips (broken zip64 offsets).
- `export_labels.py` — how the CSV was written from the labelled pickles (provenance).

How the labels were made: YOLO11 "traffic light" boxes (PTL at 2016 px / imgsz 1280 / conf 0.25;
Commons imgsz 960 / conf 0.3 with a 0.5 zoom-out retry). Each tight crop got an HSV bucket:
h/w > 2.6 → vehicle (O); green-lit ≥ 0.8 % → green; red-lit ≥ 1 % → red (S); white-lit ≥ 3 % →
white; else dark (O). All Commons green/white crops and 1,540 PTL green crops were reviewed by eye
on contact sheets as W (lit walk figure, no digits), C (any digits) or O (round lamp, unlit,
too small or unclear); unreviewed PTL greens are dropped. Doubtful crops went to O.
