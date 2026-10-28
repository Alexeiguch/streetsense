# Local light reader: EU-style and night walk signals

The crossing rule doesn't change: **go only if the local reader says walk AND Muse says yes**. This work only changes when the local reader says walk.

## What changed

1. **Zoom-out pass for close-ups** (`detector.YoloDetector.zoomed_lights`). This runs only when no traffic light reaches conf ≥ 0.35. The frame is shrunk to 0.5×, then 0.3×, and centred on a grey canvas. A separate YOLO instance does the detecting, so ByteTrack state is untouched. Turn it off with `STREETSENSE_LIGHT_ZOOM=0`, or set scales with `STREETSENSE_LIGHT_ZOOM=0.5,0.3`.
2. **Per-lamp walking-figure check** (`traffic_light.lamp_figure`). It builds a green mask and closes it morphologically, so a dotted LED figure joins into one shape, then takes the largest component. The component must:
   - be 18–80 % of the head height, upright (aspect 1.3–4.5) and not solid;
   - not touch the crop edge;
   - have **two feet apart** in its bottom 14 %;
   - sit in the lower half of a vertical head;
   - have little other lit green near it and no lit red.

   Vehicle-kind boxes are never checked.
3. **Glare.** The figure check runs on heads whose colour vote is green or unknown. A pale, washed-out green figure still passes the mask (sat ≥ 50).
4. **Night bloom.** A white core (sat < 70, val ≥ 220) counts as figure only when the ring around it is mostly green (halo ≥ 0.12).
5. **Countdown guard** (`figure_walk_guard`). If lit digits or bars sit right beside the head, a figure walk becomes `green` with `countdown=True`, so the reader does not say walk. This is how the Most "green man + countdown" phase stays non-walk.
6. **Hollow digit fix.** The old 64 px silhouette accepted a tall green "8". A figure walk from that path must now also pass the feet check, or it is downgraded to green.

Turn off parts 2–6 with `STREETSENSE_EU_FIGURE=0`. New tests are in `tests/test_eu_figure.py`.

## Hold-out

The split is by clip:
- **Dev clips (used for tuning):** dc, eu_ped_a, eu_ped_b, night_ped, most11, us_ped, tirana.
- **Held out:** everything else in the DC and broad label sets, plus new licensed frames in `eval/reader_labels_heldout.csv` (see `eval/SOURCES.md`). The new frames are: a close-up dotted LED head, a New Zealand clip, UK and Basel stills, and a Frankfurt vehicle/tram clip with a round green lamp.

Each row runs a fresh Pipeline over 5 frames, 0.25 s apart. Run it with `scripts/eval_local_reader.py --dense`. The combined go uses the cached Muse E@768 answers, so it costs $0.

## Results (before = 6ca4d87, after = this change)

| split / subset | walk recall before → after | reader walk on non-walk before → after | go (cached Muse) before → after | false go |
|---|---|---|---|---|
| dev/dc_orig | 10/10 → 10/10 | 8/20 → 8/20 * | 9/10 → 9/10 | 0 → 0 |
| dev/eu_style | 0/7 → **4/7** | 0/9 → 0/9 | 0/7 → **4/7** | 0 → 0 |
| dev/night | 0/3 → **1/3** | 0/4 → 0/4 | 0/3 → **1/3** | 0 → 0 |
| dev/us_day2 | 2/2 → 2/2 | 0/4 → 0/4 | 2/2 → 2/2 | 0 → 0 |
| heldout/dc_orig | – | 0/10 → 0/10 | – | 0 → 0 |
| heldout/eu_style | 0/3 → 0/3 | 0/5 → 0/5 | 0/3 → 0/3 | 0 → 0 |
| heldout/night | 0/2 → 0/2 | 0/9 → 0/9 | 0/2 → 0/2 | 0 → 0 |
| heldout/new EU (pex, nz) | 0/7 → 0/7 | 0/7 → 0/7 | no Muse cache | – |
| heldout/new stills | 0/2 → 0/2 | 0/1 → 0/1 | no Muse cache | – |
| heldout/vehicle (ffm) | – | 0/3 → 0/3 | – | – |
| **all** | 12/36 → **17/36** | 8/72 → 8/72 | 11/27 → **16/27** | 0 → 0 |

Dense non-walk scan at 4 fps, 1,148 frames over every known non-walk stretch: **0 walk decisions before and after**. This includes the ffm vehicle green, tirana, the Most countdown phases, night_hand and the beacons.

The broad Muse rescore (E@768) went from 2/17 → 6/17 fused go on walk frames, still 0 on non-walk. The DC set is unchanged at 9/10.

\* **The DC false walks were there before this change.** All 8 are US frames inside the walk phase: `walk_symbol_off` ×4 and `countdown_ending` ×4. The US reader cannot read digits, so it holds walk. Muse says no on all of them, so the combined go is 0. The dense DC 48.5–53 s countdown-ending stretch reads walk on 16/19 frames both before and after. "Zero reader walk on every non-walk frame" therefore holds for the broad and held-out sets, but not for those DC frames. This change adds none.

## Caveats

- **No held-out gain.** The held-out walk frames still read green, unknown or nothing:
  - The NZ figure has its legs together, so it fails the feet check.
  - On the pex close-up, the glowing lens joins the figure into one blob that touches the crop edge. A countdown also sits beside it, so the guard would block it anyway.
  - The UK and Basel stills have no confident detection.
  - night_jp has no detection even with zoom.

  The dev gains may not carry over to other head designs. I did not tune on the held-out frames after seeing these results.
- The zoom pass adds one or two extra YOLO runs on frames without a confident light.
- Labels were made by eye. Pex walk frames are medium confidence because the figure is partly out of frame.
- Held-out night coverage is only night_jp (2 walk frames) plus the non-walk night clips.
