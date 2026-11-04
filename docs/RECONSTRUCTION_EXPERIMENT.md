# 3D reconstruction pilot — 4 November 2026

**Continue as an offline geometry experiment. The initial results do not justify adding reconstruction to live guidance.** A real Meta MapAnything model ran locally and produced a colored street point cloud, estimated cameras and depth. It improved one moving clip's correspondence consistency slightly over the existing depth output, at a substantially larger runtime cost. Every tested window failed at least one inspection gate.

[Open the self-contained demo](benchmark/reconstruction_demo.html) in a browser: orbit the cloud and switch between MapAnything and the existing depth output. The numbered orange points are estimated camera centers. The source views are below the cloud. No model or server is needed to open this saved demo.

The module is [experiments/reconstruction](../experiments/reconstruction/README.md). It has its own setup, environment and caches, with no import or data path into production perception or crossing decisions. The full [continuation prompt](prompts/3D_RECONSTRUCTION.md) is ready to paste into another work session.

## Measured results

Two public source clips, three frame windows, all at 518 × 294 model resolution. The moving close window was repeated with a resident model for performance; repetitions are not additional scenes. Raw reports, exact frame indices, decoded times and hashes are in [reconstruction_20261104.json](benchmark/reconstruction_20261104.json).

| Window | MapAnything checks within 3 px | Depth baseline checks within 3 px | Gate |
|---|---:|---:|---|
| Moving street, 0.20 / 0.30 / 0.40 s | 355/894 (39.7%) | 313/894 (35.0%), OpenVINO | Reject: one pair below 50% consistency |
| Same street, 0.20 / 0.60 / 1.00 s | 1/370 (0.3%) | Not run | Reject: poor consistency |
| Nearly static camera, 0.20 / 0.60 / 1.00 s | 2224/4010 (55.5%) | 2292/4010 (57.2%), torch | Reject: negligible parallax |

The moving close window's median pixel reprojection errors were **3.75 / 2.71 px** for MapAnything and **4.13 / 2.91 px** for the OpenVINO depth baseline across its two adjacent pairs. Valid-projection p95 errors were 11.26 / 5.10 px versus 79.67 / 54.37 px. This suggests better depth shape consistency in this particular clip, including fewer large mismatches. It does not establish metric accuracy, obstacle recall or navigation benefit.

The static control's median predicted ray angles were approximately **0.020° and 0°**. Its relatively good reprojection score therefore does not demonstrate useful triangulation. This is why parallax is checked separately.

SIFT matches are noisy proxies, and directional checks share points and views. They are not independent accuracy trials. Dynamic vehicles and people were not excluded. The two moving windows were selected to inspect overlap after the initial wide-spacing failure; these are development results, not held-out validation. No measured distances, curb heights, camera calibration or trajectory ground truth are available in these clips.

## Runtime and memory

Hardware: macOS ARM64, 8 CPU cores, 24 GiB unified memory, Apple MPS available. MapAnything used float32, memory-efficient inference and minibatch size 1; MPS fallback was enabled. Explicit accelerator synchronization brackets timing.

| Measurement | Observed value |
|---|---|
| MapAnything, three resident-model inferences on the close window | 3.753 / 3.145 / 3.157 s **per three-view batch** |
| Model load for that run, cached checkpoint | 24.064 s |
| Existing OpenVINO depth reader, same three processed views | 152 / 118 / 108 ms **per view** |
| Baseline initial load/export/compile | 7.253 s; includes first OpenVINO export |
| Process peak RSS during the repeated model + baseline run | 6.94 GB |
| MPS driver allocation sampled after inference | 6.58 GB |

RSS and MPS allocation can overlap in unified memory; do not add them together. The MPS figure is a sampled allocation, not a measured peak. These are a few pilot timings, not a stable p95 latency benchmark. The initial wide-spacing run took 6.95 s and the static control took 3.92 s; cache and warm-up effects matter. Checkpoint download time is excluded from model inference. No full StreetSense pipeline was benchmarked with reconstruction because it is not integrated.

## Comparison contract

MapAnything predicts metric-scale geometry, but its metres remain **unvalidated predictions** here. The baseline uses the existing `DepthEstimator`, whose relative output is inverted, then independently aligned per view to the median MapAnything depth. It borrows MapAnything intrinsics and poses. This is an optimistic depth-shape comparison under shared cameras, not an independent reconstruction baseline. Neither learned scale nor production box-height priors count as ground truth.

All mutual SIFT ratio matches between adjacent processed views enter the check. Both directions are projected. Invalid, masked and behind-camera points count as failures in the 3 px denominator. Median/p95 errors exclude invalid projections; validity coverage remains in the raw reports. Matching is not filtered using MapAnything geometry. Repeated textures and dynamic objects can still produce incorrect correspondences.

Provisional gates: at least 30 matches per pair, median predicted ray angle at least 0.5°, at least 50% of checks within 3 px in every pair, and no duplicate decoded frames. These are inspection thresholds, not calibrated safety criteria. A passing experiment would return `inspect`, never authorize movement.

The viewer displays a deterministic sample of at most 24,000 points and clips the farthest 2% for legibility. Metrics use all valid points. It can expose stretching, duplicated moving cars and poor geometry when orbiting; attractive rendering is not evidence that those surfaces are measured correctly.

## Reproduce

```bash
bash experiments/reconstruction/setup.sh
bash experiments/reconstruction/run.sh \
  --source samples/mazatlan_sidewalk.webm \
  --times 0.2,0.3,0.4 --device mps --repeats 3 --baseline \
  --out outputs/reconstruction/my-close-run

bash experiments/reconstruction/run.sh \
  --source samples/dc_ped_light.ogv \
  --times 0.2,0.6,1.0 --device mps --baseline --baseline-backend torch \
  --out outputs/reconstruction/my-static-control
```

Full geometry/model output remains locally in `outputs/reconstruction/`. Images and videos are ignored by Git; [sample source instructions](../samples/README.md) describe obtaining them. The existing model caches have been retained in `.cache/reconstruction/` so the next run does not repeat the checkpoint download. The setup was checked in the durable `.venv-3d`; production `.venv` remains separate.

Validation: **479 tests passed, 1 skipped**, including eight new tests for geometric conventions, invalid projections, depth alignment, rejection controls and import isolation. The actual Meta model ran on the Apple GPU; the production OpenVINO depth backend also ran locally. Browser interaction checks cover orbit, baseline selection and reset.

## Next experiment

Record a short, slow walking sequence with known camera intrinsics, several physically measured distances, curb-height references and camera translation. Keep a second scene held out. Compare raw and dynamic-object-excluded geometry, and use an independent geometric camera estimate rather than borrowing learned cameras for every baseline. If pose refinement is tried, fit and evaluate on disjoint feature tracks/views.

Only reconsider an asynchronous geometry advisory layer after this shows lower metric distance and curb-height errors without worse obstacle/free-space false negatives, and after measuring warm latency and peak memory over enough varied sequences. The signal reader and Muse conjunction remains the only crossing authorization rule.

## Models and source provenance

- [Meta MapAnything](https://github.com/facebookresearch/map-anything), source `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`, UniCeption 0.1.7. [Apache checkpoint](https://huggingface.co/facebook/map-anything-apache), revision `00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a`. The upstream default checkpoint has different licensing and is not used.
- Setup pins [DINOv2](https://github.com/facebookresearch/dinov2) to `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Its Python tree fingerprint matches the Torch Hub code in the recorded repeated run. Torch 2.14.1, torchvision 0.29.1; other versions are recorded in the raw report. Older pilot reports predate the expanded package fingerprint fields.
- The [existing depth reader](../streetsense/depth.py) uses [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), Apache 2.0. The OpenVINO export is generated locally. Its report has no checkpoint revision because the existing reader does not expose one for the exported backend.
- Demo footage: [Video de la Avenida Ejército Mexicano, Mazatlán, 25 de febrero de 2022](https://commons.wikimedia.org/wiki/File:Video_de_la_Avenida_Ej%C3%A9rcito_Mexicano,_Mazatl%C3%A1n,_25_de_febrero_de_2022.webm), DogeGamer2015MZT, CC0. The saved demo contains derived views and point colors from this source.
- Static control: [Pedestrian crossing light washington dc](https://commons.wikimedia.org/wiki/File:Pedestrian_crossing_light_washington_dc.ogv), Sebastian Wallroth, CC BY 3.0. Its local output is not embedded in the saved demo.
