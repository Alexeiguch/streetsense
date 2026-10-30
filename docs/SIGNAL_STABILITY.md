# DC signal stability, 29 October 2026

GO still requires local WALK and a validated Muse yes. Uncertainty remains WAIT.
No Meta calls were made for this work.

This branch provides timing cleanup, safer confirmation and diagnostics. It does
not establish a performance improvement. The external no-tiles evaluation supplied
by Alex tied main; that evaluation has not been rerun after the corrections below.

The subsequent no-tiles recall report and the controlled detector-input comparison
are documented in [SIGNAL_RECALL_REVIEW.md](SIGNAL_RECALL_REVIEW.md).

## Decisions

- **Use frame time.** The smoother now shares the pipeline clock and accepts each
  explicit frame timestamp. Countdown expiry follows video time during replay.
  Maps and gateway smoother resets retain their pipeline clock.
- **Confirm before holding.** A raw WALK cannot prime the countdown hold before
  three votes confirm it. Only a fresh observation of the confirmed colour renews
  its hold. Countdown-only frames cannot renew it.
- **Cancel WALK on red.** A fresh red reading removes WALK votes and its countdown
  hold immediately, including before three votes have confirmed red.
- **Expire countdown evidence.** Once a countdown has outlasted the one-second hold,
  pre-countdown WALK votes cannot help restore WALK. This also applies when a sparse
  stream resumes with WALK after the hold elapsed between frames.
- **Allow slow streams.** Frame gaps do not erase evidence by default. Steady WALK
  can confirm over three sparse observations. `max_gap_s` remains an optional
  constructor setting for callers with a known cadence. Timestamp rewinds still
  clear signal history and untracked position matches. Gateway photos retain their
  existing per-photo reset; this is not a real-glasses test.
- **Keep reset-on-veto opt-in.** `STREETSENSE_SIGNAL_RESET_ON_VETO=1` discards WALK
  votes and its hold after every confident CNN rejection, including during recovery.
  The existing CNN veto still applies immediately under both policies. The stricter
  reset reduced DC WALK availability too much to enable by default.
- **Keep v1, thresholds and crops.** No classifier retraining, threshold relaxation,
  crop averaging, longer WALK hold or hidden display hold was introduced. The trace
  shows a regular countdown-only interval, consistent with the documented LED/camera
  flicker. Stabilising a box cannot restore a figure absent from a captured frame.

## Review corrections

Review found two defects in `205c599`: a fresh red did not cancel the cached WALK
hold, and its fixed one-second gap expiry prevented steady WALK from confirming on
slow streams. It also identified an older defect: WALK votes survived an expired
countdown hold. All three were reproduced in standalone smoother tests and corrected.

The regression cases cover tracked and position-matched untracked heads, both reset
policies for red cancellation, steady frame gaps of 1.1/2/5 seconds, and countdown
expiry before or between frames, with and without an initially confirmed WALK.
Red and an expired countdown require three fresh WALK votes for recovery. A short
countdown can still hold an already confirmed colour within the existing limit.

Validation after these corrections: **366 passed, 2 skipped**; Python compilation
and `git diff --check` passed. No Meta calls were made. The full clip evaluation,
including tiles, must be rerun on the corrected version before judging merge readiness.

## Historical measurement at 205c599

The table and JSON below record the earlier implementation, before the review
corrections. They are retained as historical measurements, not results for the
current code. The 11-to-7 demo change count did not establish an accuracy gain.

Full DC clip, 628 frames sampled at 10 fps, resized to 1280×720. YOLO11s OpenVINO,
v1 CNN, 2×3 tiles always, surface/depth/remote models off. Local Mac ARM runtime:
OpenVINO 2026.4.1, Ultralytics 8.4.192. Baseline: `e62853f`.

The default and strict policies were replayed over identical recorded boxes, raw
readings and CNN probabilities. A separate full Pipeline run of the strict policy
matched the strict replay on all 628 decisions, with identical raw inputs. Crosswalk
context was regenerated for the original trace and reproduced all baseline selections.
These are local reader measurements on one development clip, not general accuracy
or fused crossing results. Baseline wall-clock expiry also depends on replay speed.

| Measurement | Baseline | Default at 205c599 | Reset-on-veto at 205c599 |
|---|---:|---:|---:|
| WALK readings, full clip (628 frames) | 176 | 188 | 59 |
| WALK readings, 10–47 s (371 frames) | 167 | 180 | 58 |
| WALK/WAIT changes, 10–47 s | 75 | 73 | 46 |
| WALK readings, demo interval 9.8–13.8 s (41 frames) | 15 | 16 | 0 |
| WALK/WAIT changes, demo interval | 11 | 7 | 0 |
| WALK readings, initial red 0–9.5 s (95 frames) | 0 | 0 | 0 |
| WALK readings, countdown ending 48.5–53 s (46 frames) | 2 | 2 | 0 |
| WALK readings, hand/red 53.5–62.8 s (93 frames) | 0 | 0 | 0 |

WALK/WAIT changes count changes in whether the local decision is WALK. Every other
local state is grouped as WAIT for this count. Muse was not asked, so these are
not GO/WAIT measurements. The strict policy's apparent stability in the demo
interval comes from returning no WALK readings there, which is a recall regression.

**The flicker remains.** The default fixes improve timing and confirmation but do
not make this clip continuously readable. The two countdown-ending WALK readings
are at 48.72 and 48.80 s in both baseline and default. The earlier 0/19 result was
sampled at 4 fps; it does not establish zero errors at 10 fps. Neither metric is a
measurement of false GO. The full historical held-out and dense suites have not
been rerun here; the new trace measurements must not replace their reported results.

A separate end-to-end default run on the held-out New Zealand clip sampled 42
frames at 10 fps. It produced no WALK readings, including 0/8 during the labelled
3.4–4.1 s non-walk interval. It still misses that clip's WALK phase. Trace replay
reproduced all 42 local decisions. This check establishes no held-out recall gain.

Validation: 348 tests passed, 2 skipped; Python compilation and `git diff --check`
passed. No detector/model throughput or real-glasses performance was measured.

Machine-readable results: [benchmark/signal_stability_dc.json](benchmark/signal_stability_dc.json).

## Inspect or replay

The trace records track/source/box, all four CNN probabilities, heuristic/raw/smoothed/
final states, selected head, selection reason and distance caution. Images are saved
only when `--crop-dir` is requested, at decision transitions. Generated traces and
crops stay under gitignored `outputs/`.

```sh
STREETSENSE_LIGHT_TILES=2x3 STREETSENSE_LIGHT_TILES_MODE=always \
  .venv/bin/python scripts/trace_signals.py samples/dc_ped_light.ogv \
  --out outputs/dc_trace.jsonl --crop-dir outputs/dc_crops

.venv/bin/python scripts/replay_signal_trace.py outputs/dc_trace.jsonl \
  --out outputs/dc_default.jsonl
.venv/bin/python scripts/replay_signal_trace.py outputs/dc_trace.jsonl \
  --out outputs/dc_reset.jsonl --reset-on-veto
```

The replay reuses recorded inputs and crosswalk context. Use the same camera and
distance configuration as capture. It does not rerun detection or classification,
measure their accuracy, or ask Muse. Demo dumps now include the same diagnostic trace.
