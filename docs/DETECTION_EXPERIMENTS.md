# Signal detection experiments, 1 November 2026

The useful change is an **opt-in single focused tile**, `STREETSENSE_LIGHT_FOCUS=1`.
It improves local held-out WALK reads from **1/14 to 5/14** on this host, keeps dev
reads at **16/22**, and retains the existing **1/72 labelled false WALK**.
The four recovered samples all come from one close-up clip. This is exploratory
evidence about crop quality, not proof of general accuracy or safe navigation.
These historical held-out clips were inspected and used to select the method
in this study. A fresh test set is still needed.

Default behaviour, CNN v1, its thresholds, the original smoother and the requirement
for local WALK **and** validated Muse Spark yes are unchanged. Six-tile detection
stays off. No Meta requests were made.

## What was tried

All methods use YOLO11s, OpenVINO CPU at 640, original image pixels for reading,
and the existing heuristic, CNN, smoother, distance checks and signal selection.
The paired runner reuses the same full-frame/fallback detections for each method
and gives each method fresh pipeline state. It deep-copies detections so reader
attributes cannot leak across methods. No classifier was retrained.

- **Extra zoom:** run the existing 0.5/0.3 zoom-out pass even if a head was already found.
- **Context re-detection:** expand up to two detected heads by 2x and detect again.
- **CLAHE:** another detection pass with luminance contrast enhancement. Reading still uses the original pixels.
- **One focused tile:** select one ordinary overlapping 2x3 tile using bright, saturated pixels inside the highest-confidence head. Hue is ignored. This only schedules a detector pass.
- **1x2 tiles:** two broad tile passes.
- **2x3 tiles:** six passes, as a reference.

The first study's one-tile method preserves original boxes. The runtime version
adds further safeguards: it only accepts refinements centred inside their parent,
uses a separate 3-of-5 smoother, and discards child WALK evidence after a parent's
red, amber, countdown or strong CNN veto. A missing parent also vetoes the child.
Its matching threshold equals the NMS threshold, so two surviving refinements
cannot cast two votes into one history on the same frame. It requires three fresh
WALK votes after a veto and works on sparse frame streams.
Focus does not stack with a tiled pass on the same frame.

## Measurements

The complete runtime comparison uses all 108 labelled rows: 22 dev WALK, 14
held-out WALK and 72 other labels. Each sample uses the existing five-frame
warm-up over the preceding second; stills are repeated five times. Both variants
use CNN v1 and explicit settings, with SAM and all remote models disabled.

| Runtime | Dev WALK | Held-out WALK | Held-out EU WALK | Labelled false WALK |
|---|---:|---:|---:|---:|
| Default | 16/22 | 1/14 | 0/7 | 1/72 |
| Focus enabled | 16/22 | 5/14 | 4/7 | 1/72 |

Recovered rows: `pex6415590_07.75`, `pex6415590_08.50`, `pex6415590_09.25`,
`pex6415590_10.00`. The New Zealand WALK samples and the Basel figure remain
missed. Dev night recall stays at 1/3 in both runtime variants.
The existing labelled false WALK is `dc_48.72`, in both variants. Do not present
this result as zero false WALKs.

The dense scan is a separate continuous evaluation with fresh pipeline state per
interval. It is not interchangeable with the labelled warm-up evaluation.

Both variants returned **0/1,148 false WALK frames** on the known non-WALK
intervals, plus **0/19** on the separate US countdown-ending interval. The
labelled `dc_48.72` error above remains despite that dense result.

Raw evidence: [runtime comparison](benchmark/detection_experiments_20261101/focused.json),
[exploratory methods](benchmark/detection_experiments_20261101/methods.json),
[asset and model hashes](benchmark/detection_experiments_20261101/assets.json).
The exploratory script is preserved at commit `8d4dc10`; the runtime comparison
ran code at `108c428`, including the per-frame vote-counting fix. Package versions and script hashes are in each JSON.

Validation: **391 tests passed, 2 skipped**. All trials used local models and cost
**$0 in Meta API credits**.

The exploratory comparison covered 107 of 108 labelled rows. `mazatlan_03.00`
was omitted while its download was unavailable; it is a non-WALK row. All
36 WALK rows were present. The subsequent runtime comparison above includes
all 108 rows. The exploratory job was stopped after its labelled comparison;
dense verification is reported for the safeguarded runtime candidate above.

| Method | Dev WALK | Held-out WALK | False WALK on other labels |
|---|---:|---:|---:|
| Default | 16/22 | 1/14 | 1/71 |
| Extra zoom | 12/22 | 1/14 | 1/71 |
| Context re-detection | 15/22 | 1/14 | 1/71 |
| CLAHE | 15/22 | 1/14 | 1/71 |
| One focused tile, exploratory | 16/22 | 5/14 | 1/71 |
| 1x2 tiles | 15/22 | 2/14 | 1/71 |
| 2x3 tiles | 15/22 | 5/14 | 1/71 |

Extra zoom, context re-detection and CLAHE failed to improve held-out recall
and lost dev WALK reads. Two broad tiles gained the Basel still but lost a dev
read. Six tiles matched the four close-up gains but also lost a dev read. The
single focused tile was the only explored variant to keep dev recall and
recover those four held-out samples. The guarded runtime version retained that result.

## Host latency

Serial timing on an 8-CPU ARM64 Mac after the eval jobs stopped. Each clip
cycles the same five source frames, with 10 warm-up and 30 measured frames
per method. Median pipeline milliseconds, rounded; raw samples and p95 are
in [latency.json](benchmark/detection_experiments_20261101/latency.json). This
excludes capture, rendering, surface/depth models and networking.

| Sample | Default | Focus | Six tiles |
|---|---:|---:|---:|
| DC | 53 ms | 238 ms | 790 ms |
| Pexels close-up | 59 ms | 258 ms | 419 ms |
| Night | 194 ms | 323 ms | 827 ms |

Focus is cheaper than six tiles in these samples, but adds substantial work
to the default path. Keep it opt-in. No phone or glasses latency is claimed.

## Reproduce

Fetch the clips with `scripts/get_samples.sh`, `scripts/get_eval_clips.sh` and
`scripts/get_heldout_clips.sh`. The last script now retrieves and verifies all
three held-out stills. Source links and original checksums are in
[eval/SOURCES.md](../eval/SOURCES.md).

```sh
python scripts/experiment_light_detection.py --methods baseline,focused --dense --out outputs/focus_eval.json
python scripts/experiment_light_detection.py --methods baseline,zoom_always,roi,clahe,saliency_tile,tiles1x2,tiles2x3 --out outputs/methods_eval.json
python scripts/benchmark_light_focus.py --warmup 10 --out outputs/focus_latency.json
```

Run the latency script with other inference jobs stopped. Its frame cycling is for
host timing only; it is not an accuracy test or a phone/glasses benchmark.
The experiment runner fails on missing assets unless `--allow-missing` is explicit,
records omitted rows, and marks an interrupted run `complete: false`.

For an opt-in local video trial:

```sh
STREETSENSE_LIGHT_FOCUS=1 STREETSENSE_LIGHT_TILES=0 STREETSENSE_SAM_FALLBACK=0 \
  python -m streetsense run --source samples/heldout/pexels_6415590.mp4 --no-vlm
```

Leave focus off for the default demo until it has been checked on fresh clips and
the actual phone or glasses hardware. There was no new fused-GO evaluation and no
real-glasses test in this study.
