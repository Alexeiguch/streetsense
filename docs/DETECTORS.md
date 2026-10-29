# Traffic-light detectors: benchmark and the opt-in tiled pass

Benchmark run on the dev box (8 vCPU x86, no GPU), 28–29 Oct 2026. Raw numbers: [benchmark/detectors.json](benchmark/detectors.json). Tiled-mode reader eval: [benchmark/local_reader_eval_tiles_always.json](benchmark/local_reader_eval_tiles_always.json).

## Method

- **Ground truth:** hand-drawn boxes for the signal heads (vehicle and pedestrian lights) in `gt.json`, on the same frames that `scripts/eval_local_reader.py` uses. A detection counts as a hit when IoU >= 0.3 with a GT box. Anything else is a false positive (FP).
- **Splits:** *dev* is the set the reader was tuned on. *Held-out* is clips the reader never saw (EU-style, still and vehicle-light sequences; 14 labelled walk frames). Treat held-out as the honest number.
- **Reader replay:** I cached each detector's boxes per frame and pushed them through the real `Pipeline` light stage (reader + 64 px signal CNN + smoother + rules), with the detector swapped for a replay. "walk read" means the local reader's final state was `walk`. "go" means local walk AND cached Muse Spark `yes`. No new Meta calls.
- **Dense scan:** a 4 fps stream over known non-walk stretches (1148 frames), plus the DC countdown-ending interval (19 frames). The column counts frames where the decision was walk.
- **Latency:** clean CPU median and p90 over the same 30 eval frames (2011x1557), 3 warm-ups, one model at a time, with nothing else heavy running (background load ~2 from system daemons). Ultralytics/OpenVINO models used OpenVINO LATENCY mode on CPU (torch threads 7). RF-DETR ran in torch with 8 threads. "Fallback" rows give an estimated mean: base + fallback rate x fallback model.
- "not run", "not replayed", "not measured" and "–" mean the measurement was not made. No numbers are estimated except the fallback-row latency means.

## Results

| detector (thr) | recall dev | recall held-out | FP/frame all eval frames | FP/frame no-signal frames | held-out walk frames with a head box | held-out walk read (reader+CNN) | held-out go (cached Muse) | dev walk read | false walk labelled (dev+held-out) | dense non-walk walk | CPU ms/frame median (clean) | size | licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11s OV 640, no zoom @0.35 | 113/188 (60%) | 84/162 (52%) | 0.06 | 0.12 | 9/14 | 1/14 | 0/5 | 17/22 | 1/72 | not run | not measured clean | 37 MB OV FP32 | AGPL-3.0 |
| **YOLO11s OV 640 + zoom-out (current)** @0.35 | 113/188 (60%) | 86/162 (53%) | 0.05 | 0.12 | 12/14 | 1/14 | 0/5 | 15/22 | 1/72 | 0/1148 (DC countdown-ending 0/19) | 20 (p90 81, 30 fr) | 37 MB OV FP32 | AGPL-3.0 |
| YOLO11s OV 1280 @0.35 | 111/188 (59%) | 82/162 (51%) | 0.09 | 0.24 | 9/14 | 1/14 | 0/5 | 7/22 | 0/72 | not run | not measured clean | 37 MB OV FP32 | AGPL-3.0 |
| YOLO11s OV full + 2x3 tiles @0.5 | 116/188 (62%) | 84/162 (52%) | 0.13 | 0.18 | 11/14 | 5/14 | 0/5 | 14/22 | 0/72 | 0/1148 (DC countdown-ending 0/19) | 113 (p90 135, 30 fr) | 37 MB OV FP32 | AGPL-3.0 |
| YOLO11m OV 640 @0.35 | 110/188 (59%) | 76/162 (47%) | 0.02 | 0.12 | 13/14 | 2/14 | 1/5 | 9/22 | 0/72 | not run | 26 (p90 30, 30 fr) | 78 MB OV FP32 | AGPL-3.0 |
| RT-DETR-L OV 640 @0.5 | 128/188 (68%) | 98/162 (60%) | 0.07 | 0.35 | 14/14 | 1/14 | 0/5 | 15/22 | 0/72 | not run | 45 (p90 58, 30 fr) | 124 MB OV FP32 | AGPL-3.0 (Ultralytics impl.) |
| RT-DETR-L OV 640 @0.35 | 157/188 (84%) | 120/162 (74%) | 0.29 | 0.88 | 14/14 | 1/14 | 0/5 | 13/22 | 0/72 | not run | 45 (p90 58, 30 fr) | 124 MB OV FP32 | AGPL-3.0 (Ultralytics impl.) |
| RF-DETR nano (torch) @0.35 | 149/188 (79%) | 89/162 (55%) | 0.08 | 0.24 | 13/14 | 1/14 | 0/5 | 9/22 | 0/72 | not run | not measured clean | not measured | Apache-2.0 |
| RF-DETR base (torch) @0.35 | 153/188 (81%) | 105/162 (65%) | 0.06 | 0.24 | 12/14 | 1/14 | 0/5 | 14/22 | 0/72 | not run | not measured clean | not measured | Apache-2.0 |
| RF-DETR medium (torch) @0.35 | 160/188 (85%) | 102/162 (63%) | 0.08 | 0.35 | 14/14 | 1/14 | 0/5 | 14/22 | 0/72 | not run | 126 (p90 176, 30 fr) | not measured | Apache-2.0 |
| YOLO-World v2 s, 5 prompts (torch) @0.2 | 139/188 (74%) | 72/162 (44%) | 0.46 | 0.59 | 14/14 | 2/14 | 1/5 | 17/22 | 1/72 | not run | not measured clean | 25 MB + CLIP ViT-B/32 text enc. (prompt time) | AGPL-3.0 |
| YOLOE-11s-seg, 5 prompts (torch) @0.3 | 109/188 (58%) | 65/162 (40%) | 0.18 | 0.18 | 14/14 | 2/14 | 1/5 | 17/22 | 1/72 | not run | not measured clean | 27 MB + MobileCLIP 572 MB text enc. (prompt time) | AGPL-3.0 |
| OWLv2 base ens. (torch) @0.2 | 184/188 (98%) | 148/162 (91%) | 5.04 | 3.65 | 14/14 | 2/14 | 1/5 | – | 0/35 | not run | not measured clean | not measured | Apache-2.0 |
| Grounding DINO tiny (torch) @0.3 | 112/188 (60%) | 94/162 (58%) | 0.36 | 0.76 | 13/14 | not replayed | – | – | – | not run | not measured clean | not measured | Apache-2.0 |
| Florence-2 base `<OD>` (torch) @0.5 | 89/188 (47%) | 95/162 (59%) | 0.83 | 1.12 | 14/14 | 2/14 | 1/5 | – | 0/35 | not run | not measured clean | not measured | MIT |
| Meta SAM 3.1 API 'traffic light' (held-out only) @0.5 | not run | 121/162 (75%) | 0.24 | 0.38 | 14/14 | 2/14 | 1/5 | – | 0/35 | not run | remote ~608 | remote | Meta API terms |
| Fallback: current, then YOLO11m if no light @0.35 | 114/188 (61%) | 88/162 (54%) | 0.05 | 0.12 | 14/14 | 2/14 | 1/5 | 15/22 | 1/72 | not run | 21 mean (7% of frames +26) | 37 + 78 MB | AGPL-3.0 |
| Fallback: current, then RF-DETR medium if no light @0.35 | 116/188 (62%) | 93/162 (57%) | 0.05 | 0.12 | 14/14 | 2/14 | 1/5 | 16/22 | 1/72 | not run | 29 mean (7% of frames +126) | 37 MB + RF-DETR-M | AGPL + Apache |
| Fallback: current, then RT-DETR-L if no light @0.35 | 116/188 (62%) | 92/162 (57%) | 0.08 | 0.29 | 14/14 | 2/14 | 1/5 | 16/22 | 1/72 | not run | 23 mean (7% of frames +45) | 37 + 124 MB | AGPL-3.0 |

Slow models (OWLv2, Florence-2, SAM 3.1) were replayed on held-out sequences only. That is why their dev-read column shows "–" and their false-walk denominator is 35. Grounding DINO boxes were not replayed. OWLv2's high recall comes with about 5 FP per frame.

### Tiled pass in the real pipeline (`scripts/eval_local_reader.py`, YOLO11s OV)

| mode | held-out walk read | dev walk read | false walk labelled | false go | dense non-walk walk | DC countdown-ending |
|---|---|---|---|---|---|---|
| current (tiles off, fd45421 eval) | 1/14 | see READER.md | 1/72 | 0 | 0/1148 | 0/19 |
| `STREETSENSE_LIGHT_TILES=2x3 STREETSENSE_LIGHT_TILES_MODE=always` | **5/14** | 16/22 | 1/72 (the existing dc_48.72) | 0 | **0/1148** | 0/19 |
| `STREETSENSE_LIGHT_TILES=2x3` (fallback: tile only when no light >= 0.35) | 1/14 | 16/22 | 1/72 | 0 | 0/1148 | 0/19 |
| fallback + `STREETSENSE_LIGHT_TILES_EVERY=3` | 1/14 | 15/22 | 1/72 | 0 | not run | not run |

In fallback mode the tiles rarely fire, because a confident light is usually already in view. They do not lift held-out reads: the gain comes from extra or tighter boxes on frames where the full frame already found something. Only `always` delivers the 5/14; the cheaper modes measured here gave no held-out gain. `always` costs about 113 ms per frame against 20 ms (median, clean). Held-out go stays 0/5 in every mode, because cached Muse answers on those frames are not `yes`.

## Findings

1. **Detection is no longer the main bottleneck.** The current setup (YOLO11s OV 640 + zoom-out) puts a box on 12 of 14 held-out walk frames but reads walk on only 1 of 14. Detectors with higher recall (RT-DETR-L, RF-DETR-M, OWLv2: 14/14 boxed) still read 1–2 of 14. The reader and classifier now limit performance.
2. **Tiling helps the reader.** Full frame + 2x3 tiles lifts held-out walk reads from 1/14 to 5/14, with 0/1148 dense false walks and 0/19 on the DC countdown-ending interval. Labelled false walks stay at the existing 1/72, and false go stays 0. Cost: about 113 ms per frame median, against 20 ms.
3. **SAM 3.1 (Meta API)** has the best held-out recall of any low-FP option, 121/162 (75%) at 0.24 FP per frame. OWLv2 scores 91% but at about 5 FP per frame, and RT-DETR-L @0.35 scores 74% at 0.29, at about 608 ms per frame remote. The full benchmark run cost $0.64. Use it only as an occasional fallback check. It can never authorise a crossing.
4. The cheap fallbacks (YOLO11m / RT-DETR-L / RF-DETR-M only when no light is found) box 14/14 held-out walk frames for about 21–29 ms mean, but read only 2/14.

## Safety

The rule is unchanged: **go only if the local reader says walk AND Muse Spark says yes. Any doubt means wait.** Tile detections are only extra traffic-light candidate boxes. They are NMS-merged (IoU 0.5) with the full-frame lights and go through the same reader, CNN veto and smoother. They have no path to a walk decision of their own, and Muse must still agree.

## Switches

- `STREETSENSE_LIGHT_TILES=2x3`: rows x cols, 30 % overlap. Default off.
- `STREETSENSE_LIGHT_TILES_MODE=fallback|always`: default `fallback`.
- `STREETSENSE_LIGHT_TILES_EVERY=N`: in fallback mode, also tile every Nth frame.
- `detector.light_tile_conf` (0.5): minimum confidence for tile boxes. On frames where tiles run, they replace the zoom-out pass, as benchmarked.

## Licences

- Ultralytics models (YOLO11, YOLO-World, YOLOE) and Ultralytics' RT-DETR implementation: AGPL-3.0.
- RF-DETR: Apache-2.0.
- OWLv2, Grounding DINO: Apache-2.0.
- Florence-2: MIT.
- SAM 3.1 via the Meta Model API: Meta API terms.

## Caveats

- Small samples: 14 held-out walk frames and 5 with cached Muse. One frame moves a rate by 7 points.
- The bench tile row applies 0.5 to all boxes. The pipeline keeps full-frame lights at the normal threshold and applies 0.5 to tile boxes only. The held-out result is the same (5/14).
- Latency is on this box only, at image size 2011x1557. Phone or glasses streams at lower resolution will differ.
- Box sizes for the RF-DETR, OWLv2, Grounding DINO and Florence-2 models were not measured.
