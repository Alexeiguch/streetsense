# Signal recall review, 30 October 2026

No smoother, detector, classifier, threshold or crossing-rule change was made in
this follow-up. Tiles remain off by default. No Meta calls were made.

## What the supplied evaluation shows

The supplied main run (`5b70b28`) reports 4/7 held-out European walks and 2/3 dev
night walks. The merged no-tiles run (`7f41d7c`, containing `eb13a27`) reports 0/7
and 1/3. However, these runs contain different **detector inputs**, not just
different smoothed states:

- At `pex6415590_08.50`, main has three heads, including a 717×595 WALK crop.
  The merged no-tiles run has one 948×1056 head, read as green. The WALK crop is
  absent. All four lost Pexels WALK rows show this pattern.
- Main's tiled run has the same extra crop shapes as its reported no-tiles run.
- At `night_ped_01.60`, the merged no-tiles run has **no signal detections**.
  Main's no-tiles and tiled runs both include a 666×1172 WALK crop.

The detector and detector configuration code are identical between the compared
main and smoother branch. A smoother cannot recover an absent input crop.
The old reports do not record effective tile settings or crop sources, so they
cannot prove how those extra detections were obtained. Inherited tile settings
are a likely explanation; this remains an inference about the supplied runs.

## Controlled local comparison

The licensed clips were re-downloaded and matched their recorded SHA-256 values.
Using CNN v1, YOLO11s OpenVINO CPU, SAM off, up to five samples per labelled row and
explicit `STREETSENSE_LIGHT_TILES=0`, identical detector/classifier inputs were
processed by the exact main and branch smoothers. The tracker was reset between
labelled rows; the detector handle was reused to avoid repeated compilation.
Rows near the start of a clip have fewer than five samples, as in the existing evaluator.

| Local comparison | Main smoother | Branch smoother |
|---|---:|---:|
| Held-out European WALK rows, tiles forced off | 0/7 | 0/7 |
| Dev night WALK rows, tiles forced off | 1/3 | 1/3 |
| Non-walk rows in these clips, tiles forced off | 0/10 false WALK | 0/10 false WALK |
| Pexels WALK rows, diagnostic tiles 2×3 always | 4/4 | 4/4 |

The diagnostic tiled run marks the recovered Pexels WALK boxes as `tile` sourced,
with dimensions matching the extra boxes in the supplied baseline to within the
small run-to-run detector differences. The explicit no-tiles run reproduces the
merged report's missing boxes. This comparison isolates smoothing; it is not a
full end-to-end main-versus-merge evaluation or a dense non-walk rescore.

Machine-readable rows and source-report hashes:
[benchmark/signal_reacq_eval_audit.json](benchmark/signal_reacq_eval_audit.json).

## Evaluation controls added

- `eval_local_reader.py --no-tiles` forces the tile grid off and clears periodic
  tiled passes, even if the shell inherited a previous tiled run's settings.
- JSON and console output record effective tile grid/mode/interval, zoom scales,
  classifier version, requested detector settings, package versions and code revision.
  Rows record the actual detector backend/model and each crop's source and box.
- `--trace` records every warm-up frame's signal stages. Dense results count frames
  on which a tiled pass actually ran. Missing dense-scan clips now fail instead of
  silently contributing zero frames.
- Tile index 0 is now correctly reported as tiled in traces and demo dumps.
- Evaluation explicitly disables both Meta signal fallback and Meta crosswalk SAM.

For the next full branch run:

```sh
STREETSENSE_SAM_FALLBACK=0 python scripts/eval_local_reader.py \
  --no-tiles --trace --dense --out outputs/eval_signal_reacq/branch_explicit_off.json
```

For main's older evaluator, explicitly set `STREETSENSE_LIGHT_TILES=0` and
`STREETSENSE_LIGHT_TILES_EVERY=0` in the command. Omitting a variable does not clear
an inherited value. Compare the same detector/model assets and extracted frames.

The requested 4/7 with tiles off has **not** been demonstrated by the controlled
comparison. There is no evidence here for weakening confirmation to reach it.
The red-cancellation, sparse-frame and expired-countdown safeguards stay intact.

Validation of the evaluation controls: 370 tests passed, 2 skipped; compilation
and `git diff --check` passed. The full dense suite was not rerun in this follow-up.
