# YOLO11n vs YOLO26n benchmark (StreetSense)

**Date:** 27 Oct 2026 (Europe/London)  
**Machine:** box CPU, 8 cores, no GPU, ~16 GB RAM  
**Ultralytics:** 8.4.190 (YOLO26 family; YOLO11n still loads and `model.track` works)  
**Settings:** input resized to 1280×720, detector `imgsz=640`, `conf=0.35`, `iou=0.5`, ByteTrack on, device=cpu  
**Script:** `scripts/benchmark_models.py` → raw JSON in `docs/benchmark/results.json` and `outputs/benchmark/`  

**Model decision: keep YOLO11n as the default weights.** YOLO26n is selectable via `--weights models/yolo26n.pt`.  
Reason: on this CPU, PyTorch+track latency is comparable-to-slower on busy clips; coco128 mAP50-95 and a small hand-labelled sample slightly favour 11n; DC key alerts match for both, but 26n is noisier. The advertised “up to 43% faster CPU ONNX” for 26n was **not** reproduced here.

**Backend decision (27 Oct 2026): default inference is OpenVINO** (`--backend openvino`). PyTorch remains `--backend torch`. See [§5](#5-pytorch-vs-openvino-backend-yolo11n) below.

---

## 1. Speed (like-for-like = PyTorch + ByteTrack)

Warm-up excluded (10 frames). Two repeats per clip for detector-only; variance is in `per_repeat` in the JSON.

### Detector-only latency (ms / frame)

| Clip | Frames | YOLO11n mean / p50 / p95 | YOLO26n mean / p50 / p95 |
|---|---|---|---|
| Tyumen crossing | 603 measured | **18.9 / 18.4 / 22.4** | 20.9 / 20.7 / 23.8 |
| DC ped light | 1559 measured | 18.8 / 17.9 / 22.8 | **16.4 / 16.1 / 19.4** |
| CDMX crossing (20–68 s) | 1190 measured | **15.2 / 15.0 / 17.8** | 17.6 / 17.3 / 20.6 |

Frame-weighted mean ≈ **17.5 ms (11n)** vs **17.6 ms (26n)** — effectively tied; 26n is faster on the sparse DC clip and slower on crowded scenes.

### Full pipeline FPS (250 frames/clip cap, warm-up 5, SegFormer every 6th, no VLM)

| Clip | 11n overlay on / off | 26n overlay on / off |
|---|---|---|
| Tyumen | 22.9 / 36.8 | 23.3 / 38.6 |
| DC | 23.0 / 38.0 | 23.6 / 37.1 |
| CDMX | 23.5 / 39.4 | 22.9 / 36.1 |

Pipeline FPS is dominated by detect + amortised SegFormer + overlay; models are interchangeable at ~23 FPS with overlay.

### Model load, weights size, peak RSS

| | YOLO11n | YOLO26n |
|---|---|---|
| Weights file | 5.35 MB | 5.29 MB |
| Cold load (after import) | ~0.03–0.04 s | ~0.03–0.04 s |
| Peak RSS during bench | ~4.7 GB (incl. SegFormer + frames) | same ballpark |

### ONNX / OpenVINO (predict only, no ByteTrack; 75 frames after warm-up on Tyumen subset)

| Format | YOLO11n mean ms | YOLO26n mean ms |
|---|---|---|
| ONNX Runtime CPU | 60.2 | 60.1 |
| OpenVINO CPU | **11.6** | **11.4** |

**Caveat:** these export numbers are *not* the like-for-like comparison (no tracking; different runtime). ONNX was slower than PyTorch here; OpenVINO is clearly faster for both. Do not treat the marketing “43% faster ONNX” claim as measured on this box.

---

## 2. Accuracy (no labelled StreetSense ground truth)

### (a) coco128 val (Ultralytics auto-download — **128-image COCO subset, not full COCO val**)

| Model | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| YOLO11n | **0.671** | **0.503** | 0.663 | **0.589** |
| YOLO26n | 0.649 | 0.486 | **0.734** | 0.549 |

26n is more precise / less complete on this tiny subset. Not a full COCO result.

### (b) Per-class detection counts on our clips (StreetSense KEEP classes, conf≥0.35)

| Clip | Model | person | car | bus | truck | traffic light | other | total |
|---|---|---|---|---|---|---|---|---|
| Tyumen | 11n | 3351 | 3178 | 1 | 68 | 1402 | — | 8000 |
| Tyumen | 26n | 3191 | 2925 | 3 | 47 | 1517 | moto 1 | 7684 |
| DC | 11n | 27 | 209 | — | — | 3126 | — | 3362 |
| DC | 26n | 28 | 181 | — | — | 3082 | — | 3291 |
| CDMX | 11n | 8527 | 687 | 62 | — | — | suitcase 1034 | 10310 |
| CDMX | 26n | 8883 | 734 | **537** | — | — | suitcase 1017 | 11171 |

**Agreement** (greedy class-aware match, IoU≥0.5):

| Clip | Fraction of 11n dets matched by 26n | Fraction of 26n dets matched by 11n |
|---|---|---|
| Tyumen | 0.855 | 0.890 |
| DC | 0.973 | 0.994 |
| CDMX | 0.853 | 0.787 |

CDMX `bus` counts diverge sharply (62 vs 537) — likely false positives on 26n in the night crowd / taxi scene; worth distrusting raw count equality there.

### (c) Hand-labelled sample (14 frames across 3 clips)

GT = per-class **counts** from inspecting extracted JPEGs (`outputs/benchmark/hand_frames/`), not drawn boxes. Crowded CDMX counts are approximate mid/near people. Score = count-based TP/FP/FN.

| Model | Precision | Recall | F1 | TP / FP / FN |
|---|---|---|---|---|
| YOLO11n | **0.874** | **0.478** | **0.618** | 97 / 14 / 106 |
| YOLO26n | 0.843 | 0.448 | 0.585 | 91 / 17 / 112 |

Small sample; 11n edges it. Details: `docs/benchmark/hand_label_eval.json`.

### (d) Downstream DC alerts

Key timeline (both models):

- **~10.80 s** — `walk signal on, countdown running`
- **~52.76 s** — `pedestrian light is red`

| | YOLO11n | YOLO26n |
|---|---|---|
| Total spoken alerts | 12 | 17 |
| Key walk @ 10.8 s | yes | yes |
| Key red @ 52.8 s | yes | yes |
| Extra vs 11n | — | `car approaching on your left` @ 16.3 s; several interleaved `traffic light is green` while ped walk is also announced |

26n preserves the critical ped-signal events but is slightly noisier (vehicle-head green mixed with ped walk).

---

## 3. Side-by-side frames

| Clip | PNG |
|---|---|
| Tyumen mid | [compare_tyumen_mid.png](benchmark/compare_tyumen_mid.png) |
| DC mid | [compare_dc_mid.png](benchmark/compare_dc_mid.png) |
| CDMX mid | [compare_cdmx_mid.png](benchmark/compare_cdmx_mid.png) |

---

## 4. How to reproduce

```bash
. .venv/bin/activate
# YOLO26 weights (gitignored, same as yolo11n)
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')" && mv -n yolo26n.pt models/

python scripts/benchmark_models.py --out outputs/benchmark --repeats 2 --pipeline-max-frames 250

# optional single-model run with 26n
python -m streetsense run --source samples/dc_ped_light.ogv \
  --weights models/yolo26n.pt --no-vlm \
  --alerts outputs/dc_26n.jsonl --stats outputs/dc_26n_stats.json
```

### Caveats

- No labelled StreetSense video GT; clip “accuracy” is counts, cross-model agreement, and a 14-frame hand count sample.
- coco128 ≠ full COCO val2017.
- Pipeline FPS used 250 frames/clip (not full video) for wall-clock; detector-only used full clips.
- ONNX/OpenVINO benches are predict-only (tracking not wired through those backends in this script).
- Peak RSS includes SegFormer weights and decoded frames held in memory during detector-only passes.

---

## 5. PyTorch vs OpenVINO backend (YOLO11n)

**Date:** 27 Oct 2026 · **Script:** `scripts/benchmark_backends.py` · raw JSON: `docs/benchmark/backend_results.json`  
Same clips/settings as above; detector warm-up 10 frames; pipeline 200 frames/clip; ByteTrack on.

First-run OpenVINO export (from `.pt`, scratch copy): **~2.0 s**. Export dir `models/yolo11n_openvino_model/` is gitignored and rebuilt automatically if missing. If `openvino` is missing or export fails, the detector logs a clear line and falls back to PyTorch.

### Detector-only latency (ms / frame, 390 measured / clip)

| Clip | Torch mean / p50 / p95 | OpenVINO mean / p50 / p95 | Speedup |
|---|---|---|---|
| Tyumen | 14.6 / 14.1 / 17.7 | **11.6 / 11.1 / 13.7** | ~21% |
| DC | 14.3 / 13.6 / 16.8 | **11.0 / 10.4 / 13.2** | ~23% |
| CDMX | 15.7 / 15.1 / 18.6 | **12.3 / 12.1 / 14.7** | ~22% |

Track IDs present on every measured frame for both backends. `ApproachTracker` “approaching” frames: Tyumen 371 vs 370; CDMX 17 vs 15; DC 0 vs 0 — logic unchanged.

### Full-pipeline FPS (overlay on / off)

| Clip | Torch | OpenVINO |
|---|---|---|
| Tyumen | 21.1 / 36.8 | **22.8 / 40.6** |
| DC | 23.8 / 37.5 | **24.7 / 42.0** |
| CDMX | 21.9 / 33.7 | **24.1 / 35.3** |

### Agreement (IoU≥0.5, class-aware)

| Clip | Torch dets matched by OV | OV dets matched by torch |
|---|---|---|
| Tyumen | **0.996** | 0.971 |
| DC | **0.997** | **1.000** |
| CDMX | 0.981 | 0.876 |

CDMX night crowd still has more OV boxes (same pattern as model-to-model noise); Tyumen/DC are effectively identical.

### DC alerts

| | Torch | OpenVINO |
|---|---|---|
| Count | 12 | 13 (one extra `traffic light is green` @ 21.0 s) |
| Walk + countdown @ 10.80 s | yes | yes |
| Pedestrian red @ 52.76 s | yes | yes |

### Web `/frame` (Flask test client, Tyumen JPEG ~640 px, surface on, 40 frames after warm-up)

| | Torch server p50 / mean | OpenVINO server p50 / mean |
|---|---|---|
| `server_ms` | 16.9 / 26.9 | **12.6 / 21.1** |

### Reproduce

```bash
python scripts/benchmark_backends.py
python -m streetsense run --source samples/dc_ped_light.ogv --backend openvino --no-vlm --alerts /tmp/a.jsonl
python -m streetsense run --source samples/dc_ped_light.ogv --backend torch --no-vlm --alerts /tmp/b.jsonl
```
