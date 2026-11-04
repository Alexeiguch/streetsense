# Offline 3D reconstruction

An isolated Meta **MapAnything** experiment: video frames → estimated cameras, depth and colored point cloud → image correspondence checks → an interactive demo. It is never imported by `streetsense/` and cannot publish navigation or crossing decisions.

The initial measurements are in [the experiment report](../../docs/RECONSTRUCTION_EXPERIMENT.md). The [continuation prompt](../../docs/prompts/3D_RECONSTRUCTION.md) describes the next work and its acceptance criteria.

## Run

From the repository root, with the existing StreetSense environment installed:

```bash
bash experiments/reconstruction/setup.sh
bash experiments/reconstruction/run.sh \
  --source samples/mazatlan_sidewalk.webm \
  --times 0.2,0.3,0.4 --device mps --repeats 3 --baseline \
  --out outputs/reconstruction/my-run
```

`setup.sh` creates `.venv-3d` and `.cache/reconstruction/`, both ignored by Git. It inherits the installed torch/torchvision pair, installs inference dependencies separately, and pins MapAnything, UniCeption and the DINOv2 source used by Torch Hub. It does not modify `.venv`. Optional first argument: the Python executable that already has the matching StreetSense dependencies. Tested on Python 3.14, torch 2.14.1 / torchvision 0.29.1, macOS ARM64 with 24 GiB unified memory. CPU and CUDA are selectable but were not benchmarked for MapAnything.

The setup uses a minimal inference dependency set. MapAnything's training, audio, telemetry and upstream demo dependencies are omitted with `--no-deps`; an unrestricted `pip check` will report those omitted extras. The actual model inference imports and runs have been checked. For the complete upstream research environment, follow the [official installation](https://github.com/facebookresearch/map-anything#installation) in another environment.

The first run downloads the approximately 4.9 GB `facebook/map-anything-apache` checkpoint. Its revision is pinned in `backend.py`; both its weights and the MapAnything code are Apache 2.0. The upstream default checkpoint has a different license, so it is not used here. No API key or external inference service is required. On this machine, accelerator access and OpenVINO CPU initialization require running outside the restrictive tool sandbox.

`run.sh` sets the dedicated model caches and enables PyTorch's MPS fallback. Reports identify the actual device and fallback setting; this does not imply every operation executed on the GPU. `--repeats` measures up to five inferences with one resident model, synchronizes the accelerator, and exports the final prediction. Loading and downloads are excluded from inference timing and recorded separately where applicable.

Outputs:

| File | Contents |
|---|---|
| `demo.html` | Self-contained orbitable point cloud, numbered camera centers and source views; comparison selector when a baseline exists |
| `cloud.ply` | At most 40,000 colored points, in the model's predicted world coordinates |
| `reconstruction.npz` | Dense world points, camera Z depth, cam2world poses, intrinsics, confidence, RGB and validity masks |
| `baseline.npz` | Existing relative-depth point maps, if requested |
| `report.json` | Source/frame hashes, model revisions, timing, masks, correspondence metrics and rejection reasons |
| `frame-*.png` | Exact decoded source frames |

Open `demo.html` directly in a browser. Drag or use Orbit/Tilt/Zoom; choose the existing depth output to compare shape. Re-evaluate saved arrays without loading MapAnything:

```bash
bash experiments/reconstruction/run.sh --reuse \
  --out outputs/reconstruction/my-run
```

Add `--baseline` to recompute the baseline; `--baseline-backend torch` selects the alternative CPU implementation. Original run reports should be retained when changing the evaluator. `--size` can reduce the inference resolution; this changes the experiment and must be reported. The baseline uses the production depth reader at its normal input resolution.

## What is checked

Adjacent processed views are matched with mutual SIFT matches and a 0.7 descriptor ratio. Selection does not use the learned 3D geometry. Each valid world point is projected into the other view in both directions. Missing, invalid, masked and behind-camera points fail the 3 px check and stay in its denominator; median/p95 pixel error use valid projections only. These are noisy correspondence proxies, not ground truth. Moving objects and repeated textures can contaminate matches.

The existing Depth Anything V2 Small output is inverted and independently scale-aligned per view to the median MapAnything Z depth. It then uses **MapAnything's intrinsics and camera poses**. This is an optimistic comparison of depth shape under shared cameras, not an independent reconstruction or a test of distance accuracy. The production reader's box-height scale calibration is not used as ground truth.

The demo gate rejects duplicate frames, fewer than 30 matches in any adjacent pair, median predicted triangulation angle below 0.5°, or fewer than 50% of checks within 3 px in any pair. These are provisional inspection thresholds. Passing returns `inspect`, never `safe`, and does not establish metric accuracy. The predicted angle itself depends on the model geometry. A calibrated physical scene is still necessary to validate it.

The display samples 24,000 points after clipping the farthest 2% from the cloud median; evaluations use all valid points. No hidden surfaces are filled in. Dynamic objects are currently included, so the output must not be treated as a free-space or persistent obstacle map.

## Tests

```bash
.venv/bin/python -m pytest tests/test_reconstruction.py -q
```

Tests cover camera conventions, Z versus radial depth, known translated-camera correspondences, behind-camera/invalid checks, inverse relative-depth alignment, static/duplicate/textureless rejection and production import isolation. They require no model download or GPU.
