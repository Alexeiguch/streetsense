# UK and European signal reading

Enable the figure-signal mode explicitly:

```bash
streetsense serve --signal-country GB
streetsense run --source samples/heldout/pexels_6415590.mp4 --signal-country EU --no-vlm
```

The same flag works on `guide` and `glasses-sim`. `STREETSENSE_SIGNAL_COUNTRY=GB` is the environment equivalent. `auto` is the default and rollback. Unknown country codes resolve to automatic mode. `UK` aliases `GB`; the European country codes listed in `traffic_light.signal_profile` and the explicit `EU` style hint select the green-figure profile. `US`/`CA` select a white-person prior.

Country hints describe expected symbols. UK crossings themselves have different layouts and phases, including near-side and far-side indicators; see the [Department for Transport guide](https://www.gov.uk/government/publications/know-your-traffic-signs/pedestrian-cycle-and-equestrian-crossings). These profiles do not model national traffic laws, infer location, detect clearance phases, or establish permission to cross. The crossing answer still requires **local walk AND validated Muse yes**.

## Changes

- A green-figure profile rejects an existing WALK reading when the crop lacks green pixels, allowing a pale green/night halo. The white-person prior similarly checks the existing white-symbol heuristic. A profile alone cannot create WALK or change stop evidence.
- The figure profile refines an oversized detected head: at least 33% of frame width and a non-vehicle housing shape. A single overlapping detector tile is chosen by bright saturated pixels within that head, without a hue vote. This can recover a smaller symbol/head box when the full-frame detector returned one large housing/countdown box.
- Original boxes stay in the reader input. Additional boxes must have their centres within the parent and avoid duplicating local heads. They pass the existing HSV/figure checks, v1 CNN thresholds, adjacent-display guard, confirmation and crossing selection.
- Refinement is limited to once per 0.25 seconds of the pipeline clock. Coordinates can be reused for that interval only while the parent overlaps by at least 0.8 IoU. Every emitted box is read again from the **current image**. Movement, disappearance, rewind and geometry changes discard cached coordinates. Refinement errors drop the extra candidates.
- Refined heads have separate negative local IDs, matched one-to-one across refreshes. They cannot share smoothing votes with the parent or other untracked heads. A parent's current red/yellow/countdown reading cancels a child's walk history and requires fresh confirmation before recovery.

No weights were retrained and no confidence threshold or WALK hold was relaxed. Full-grid tile configuration takes precedence over this refinement. Set `Config().reader.refine_heads=False` to retain the symbol prior without refinement.

## Reader measurements

The paired evaluation uses identical captured base detections and exact confidence scores. Automatic replay must match every captured decision. Each labelled row has up to five observations, spaced 0.25s apart; still photos are repeated as in the existing evaluator. Remote models and surface segmentation are off, detector tiles are explicitly off, zoom fallback and CNN v1 remain enabled. The profile is assigned by clip layout, independently of its frame labels.

| Available figure-style samples | Automatic | Profile only | Profile + refinement |
|---|---:|---:|---:|
| WALK frames read, 16 labelled | 4 | 4 | 8 |
| WALK on 27 labelled non-walk frames | 0 | 0 | 0 |

All four recovered rows are `pex6415590_07.75`, `08.50`, `09.25`, `10.00`: correlated frames from **one close-up figure-style clip**, whose country is not established. Its original labels have medium confidence because the countdown is partly out of frame. This is a useful crop-recovery result, not evidence that general UK/European accuracy doubled. The UK green and red stills were already read correctly; UK-specific video recall remains unmeasured.

Across all **97 available rows**, automatic reading found 16/30 WALK rows, refinement 20/30. Both had the existing 1/67 non-walk error (`dc_48.72`, US countdown ending). That existing error remains visible in the report; it is outside the figure-mode subset. Muse was not asked, so these are local-reader results, not GO measurements.

The eleven unavailable rows belong to the missing `night_ped` and `nz` clips. Source retrieval returned 403/404; no replacement footage or changed checksum was accepted. They are explicitly excluded, not counted as correct. The complete historical suite contains 108 rows. The available “held-out” labels have also been examined in previous development: they are not a new independent test set.

Continuous checks at 4 FPS covered **246 non-walk frames** from the available European intervals: **0 WALK decisions before and after**. This does not test UK flashing-green clearance, rain, complete night coverage, real glasses, or live crossing safety. The adjacent-display guard still causes some valid figure frames to wait. A proposed shape-only relaxation could confuse a countdown zero with a round lamp, so that guard remains unchanged.

The four recovered video rows and UK green/red photos were also verified through the real production detector/pipeline with `country=GB`, independently of the fixed-box replay. **471 tests passed, 1 skipped**. Regression coverage includes style priors, the CLI flag, call-rate bounds, motion/rewind invalidation, identity separation, failures and immediate parent-head vetoes. Full labelled/dense counts, missing rows, production checks and asset/model hashes: [measurements](benchmark/signal_profiles_20261104.json).

## CPU cost

The benchmark alternates automatic/GB order over two repeats of three clips. It measures 90 frames per run after ten warmup frames, at 1280px width, YOLO11s OpenVINO 640, CNN v1, no surface/remote/annotation stages. Its live mode uses monotonic time so refinement follows the camera cadence. Full timing rows and the source windows are included in the measurement JSON; these are local CPU measurements, not phone or glasses latency.

| Mean per-frame time over two runs | Automatic | GB profile |
|---|---:|---:|
| EU figure clip A | 64.3ms | 59.3ms |
| EU figure clip B | 54.8ms | 55.3ms |
| Close-up figure/countdown clip | 56.3ms | 76.3ms |
| Pooled throughput, 540 measured frames per mode | 17.10 FPS | 15.72 FPS |

The mean added cost over these windows is **5.15ms/frame**. The close-up case costs about **20ms/frame**, including refinement and reading the additional head; its two GB p95 values were 120.7/111.9ms. The first two clips did not require extra detector passes. Variation on their timings illustrates why these short runs cannot establish a speedup. The mode improves the measured crop recall at an explicit CPU cost. Automatic mode keeps the existing path. Detailed [timings](benchmark/signal_profiles_perf_20261104.json) include runtime versions and configuration.

## Reproduce

```bash
# Requires the licensed sample files described in eval/SOURCES.md and samples/README.md.
python scripts/evaluate_signal_profiles.py --allow-missing --dense --out outputs/profiles.json
python scripts/benchmark_signal_profiles.py --frames 90 --clock live --out outputs/profile-performance.json
python -m pytest -q
```

Omit `--allow-missing` to fail on any unavailable labelled sample. The evaluator writes the capture alongside the result; `--capture` reuses it for paired checks. `--dense-only` runs just the continuous European non-walk intervals. Raw traces and image diagnostics stay in `outputs/reader-uk-europe/`.

No Meta API calls were made for these reader experiments. Country profile, refinement age and parent veto are included in signal traces. The next evidence needed is independent UK near-/far-side video with steady/flash/clearance labels, and more distant, night and rain examples. Country-specific training or phase policies should be evaluated on that data before deployment.
