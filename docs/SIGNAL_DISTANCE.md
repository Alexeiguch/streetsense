# Signal and sign distance

`streetsense/signal_distance.py` estimates how far each traffic-light head and sign is, picks the head that plausibly controls the crossing ahead, and speaks the distance ("walk signal on, about 15 metres ahead").

**Distance can only add caution.** It never creates a `walk`. It can only turn a `walk` or `green` reading into `unclear` ("…, wait"). `go` still needs local `walk` AND Muse yes (`fuse_crossing_decision`, unchanged). A fuzz test (3000 random scenes, with and without depth cues) checks that the default rules never produce a walk that a permissive config would not.

All distances are **estimates and have not been validated.** None of the clips has ground-truth distance, and the camera field of view is assumed for every clip.

## Method

**Pinhole from an assumed housing size.**
- Formula: `d = f · H_real / h_px`, where `f = (W/2) / tan(HFOV/2)`.
- The result is corrected to slant range by `1/cos(bearing)`.
- Every estimate carries a range from low and high size priors, because the size of a real head varies a lot:

| Object | Nominal height | Range | Notes |
|---|---|---|---|
| Pedestrian head | 0.5 m | 0.3–0.8 m | A single-lamp countdown module is ~0.3 m. A 2-lamp red/green-man housing is ~0.6–0.8 m. |
| Vehicle head | 0.95 m | 0.75–1.1 m | 3-lamp housing |
| Unknown head | 0.7 m | 0.3–1.1 m | |
| Stop sign | 0.75 m | 0.6–0.9 m | |
| Other signs | 0.6 m | 0.3–1.0 m | The COCO detector only emits "stop sign" |

- A box touching any frame edge has **unknown** distance, because it is truncated.

**Camera FOV.**
- `STREETSENSE_CAMERA_HFOV_DEG` (clamped 5–170) wins. Otherwise `STREETSENSE_CAMERA=phone|rayban|glasses` is used.
- Phone: 70°.
- Ray-Ban Meta: 68° horizontal. This is from the spec sheet's portrait capture (99° diagonal, 87° vertical). Newer marketing says "100° FOV", which is the diagonal.
- The FOV only scales the metres. Ranking heads by distance does not depend on it.

**Depth (optional, `--depth`).**
- Depth Anything V2 Small runs as before. A head's metric depth uses the existing person/vehicle-calibrated scale, so it is not independent of the box sizes and needs people or cars in view.
- `fuse_depth`:
  - If depth agrees with the box range (within ×1.5), the estimate is the geometric mean.
  - If they conflict, the estimate is the nearer of the two, the range is widened, and the source is `depth-conflict`.
- Separately, each head stores `signal_depth_rel`, the median relative inverse depth over the centre of the box (larger means nearer). This needs no calibration and is used only to rank heads.

**Near or far.** A head is `near` at 25 m or less (`STREETSENSE_SIGNAL_NEAR_M`) and `far` beyond that. The label is shown in overlays only.

## Selection rules (rules.select_crossing_signal)

These rules are unchanged:
- relevance ranking;
- walk+red conflict → unclear;
- vehicle-only fallback never yields walk;
- countdown → unclear.

New rules:
1. Among relevant walk heads, take the **nearest**.
2. The selected walk head, or the selected green head, becomes `unclear` when any of these holds (codes in `SignalSelection.caution`):
   - `side`: the bearing is more than 25° off-axis (`STREETSENSE_SIGNAL_MAX_BEARING_DEG`, 5–50). It probably belongs to a different crossing.
   - `far`: it is more than 40 m away (`STREETSENSE_SIGNAL_MAX_WALK_M`, 5–80).
   - `unknown`: its distance is unknown and other relevant heads are present, or another relevant non-walk head has unknown distance. In both cases we can't tell which head is ours.
   - `nearer`: a relevant head that does not show walk or green, or an **unreadable** pedestrian or unknown-kind head roughly ahead (for example a bloomed red), is clearly nearer. "Clearly nearer" means either cue suffices: box distance < 0.8 × the walk head's distance, or relative depth > walk depth / 0.8 (`STREETSENSE_SIGNAL_NEARER_MARGIN`, 0.5–1.0).
3. A single walk head whose distance is unknown, for example clipped at the top of the frame, keeps its old behaviour. There is nothing to confuse it with.

Speech for each caution code:
- `far`: "walk signal is far away, not this crossing, wait"
- `side`: "walk signal is off to the side, not this crossing, wait"
- `unknown`: "several signals, can't tell which is ours, wait"
- `nearer`: "a nearer signal is not showing walk, wait"

Otherwise the line is "walk signal on, about 15 metres ahead" or "pedestrian light is red, 4 metres to your left". Distances under 6 m are given to the metre. Beyond that they are rounded to 5 m. "Ahead" means within ±12°.

## Demo

```
.venv/bin/python scripts/demo_signal_distance.py \
  --seg "samples/dc_ped_light.ogv:7:15:70:DC day (HFOV assumed 70)" \
  --seg "samples/ext/most_04_dusk.webm:3:13:10:Most CZ dusk, zoomed (HFOV assumed 10):depth" \
  --still-at 0:12.0 --still-at 1:8.0 \
  --out outputs/signal_distance_demo.mp4 --stills outputs/signal_distance_still
```

The demo is local only, with no Meta calls. It writes an 18 s, 960×842 H.264/yuv420p video at 10 fps, about 2.2 MB.

**DC day.** Red 8–10 s, then walk.
- The ped head is selected: "walk signal on, about 10 metres to your left, countdown running".
- The vehicle head on the same pole reads ~7 m and the ped head ~8 m. Both priors give consistent numbers on the same pole. The absolute values rest on the assumed 70° FOV: if the footage was zoomed, they are proportionally too short.

**Most dusk.**
- The detector finds the mid-distance green-man head and the "41" countdown head. It misses the near red-man head entirely. The 41 head is unreadable (bloomed red digits).
- Box size alone ranks the 41 head as *farther* (16 m against 10 m). Its box covers one lamp, while the green box covers a 2-lamp housing.
- Relative depth ranks it nearer (4.0 against 3.1). So, with depth on, the green man is turned into "a nearer signal is not showing walk, wait" on 18 of 100 frames. That matches the label (`most04_08.00` is unclear because the near head is red).
- Without depth, the box rule cannot catch this case.

## Why the local reader rarely says walk on EU and night frames

Update: the fixes and before/after numbers are in [READER.md](READER.md).

These are the concrete causes, from `traffic_light.classify_light` and the actual eval frames:

1. **No box at all on 8 of the 15 frames.** COCO YOLO11s at 640 px finds no "traffic light" on:
   - `eu_ped_b_04.50`;
   - all 3 `eu_ped_side` frames;
   - `night_ped` 00.40 and 01.60;
   - both `night_jp` frames.

   These are close-ups where the head fills the frame, or dark night scenes.
2. **The EU figure test fails, so the head reads `green`, not `walk`** (`eu_ped_a`, 4 frames, and the most11 green men). The box covers the 2-lamp housing, and the lit LED green man is dotted. The checks fail as follows:
   - After the 64×32 resize, only about 30 saturated pixels remain.
   - The largest connected component is a fragment.
   - The aspect ratio is 0–1.75, short of the ≥1.8 that `_figure_silhouette` needs.
   - The height is under 0.35 of the crop.

   `green` never allows go.
3. **Washout plus ghost red** (`eu_ped_b`, true walk head).
   - In sun, the lit green figure is pale. About 87 bright pixels fall below `MIN_SAT=90`.
   - The *unlit* red figure reflects enough sunlight to count as red.
   - With 15 red against 15 green, confidence is 0.5, which is below the 0.55 needed. The head reads `unknown`, or `red` at 07.00.
4. **Night bloom** (`night_ped_01.00`).
   - The green man blooms to a white-cyan core, and only the rim stays saturated: 7 green pixels against 36 bright, low-saturation pixels.
   - The US white-walk rule only looks at the upper half of the crop, but the EU green man sits in the lower lamp, so the head reads `unknown`.
5. **The only EU "walk" was a false positive that happened to be right.**
   - On `eu_ped_b_06.00` it came from a side-on, *unlit vehicle* head (box 888 px tall). White sky between its visors matched the white-walk rule.
   - Fixed here, trivially: the white-walk rule now skips `kind == "vehicle"`, with a regression test.
   - The cost is that EU fused go drops from 1/10 to 0/10 (see MUSE_EVAL.md †). That go was never justified.

Improving recall here, for example with a lamp-level detector, a bloom-tolerant figure test, or lower thresholds checked against labelled red frames, is a separate task. Every change needs labelled tests proving it never says `walk` on red.

## Caveats

- The distances have not been validated. There is no ground truth, and the FOV is assumed for every clip. For Most, ~10° comes from assuming a ~0.7 m housing at ~20 m.
- Size priors vary roughly ×2 to ×2.7 between low and high, so the ranges are wide and the nominal value is only a guess.
- The detector boxes sometimes merge two heads or cover one lamp of a multi-lamp head. That breaks the size-based ranking, as at Most. The relative-depth cue helps only when depth is on.
- The detector misses some near heads, such as Most's near red man. A missed head cannot trigger caution. Muse still has to say yes, and the local reader still has to say walk.
- Signs: only COCO "stop sign" is detected. One Way and parking signs are not, so sign distance is implemented but unexercised on these clips.
