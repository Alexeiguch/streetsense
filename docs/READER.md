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

## Signal crop classifier (fd45421)

A small CNN now looks at every traffic-light crop and gives one more input to the local reader. **It can never authorise crossing on its own.** Go still needs local walk AND Muse yes. When the hand rules and the classifier disagree, the reader goes with "not walk".

**Model.** `streetsense/signal_cnn.py` holds a 4-stage CNN (421k params). The ONNX file is `models/signal_cnn/signal_cnn.onnx`, 1.7 MB, and it runs on OpenVINO CPU with 1 thread. It takes 0.59 ms per crop including preprocessing, measured on the box. Input is the tight box crop, letterboxed on black to 64×64.

There are four classes:
- `walk`
- `stop` (red man or hand)
- `countdown` (any digits, including walk plus digits)
- `other` (off or unclear)

A MobileNetV3-small (6.1 MB) was also trained. It came out worse at the zero-false-walk threshold: test walk recall 0.42 with 2 false walks, against 0.51 with 0 for the tiny CNN.

**Data.** Data and licences are listed in `eval/SOURCES.md`: the PTL dataset and Wikimedia Commons, both commercial-use licences, plus dev-clip crops. Labels are in `eval/signal_labels.csv.gz`. To rebuild, run `scripts/get_signal_data.sh` to fetch, then `scripts/train_signal_cnn.py` to train.

Held-out clips and stills are not in the training or validation data. Commons near-duplicates of held-out frames were dropped (`eval/signal_commons_excluded.json`). The one held-out frame that now reads walk, `uk_green`, has no near-duplicate in the Commons crops (best 32×32 correlation 0.57).

**Combining rules** (`traffic_light.apply_signal_cnn`, `cnn_hold_veto`):
- **Veto.** A hand-rule walk becomes countdown or unknown when p(stop|countdown|other) ≥ 0.8 or p(walk) < 0.5.
- **Promote.** A non-vehicle head that the hand rules read as green or unknown becomes a walk figure when p(walk) ≥ `t_walk`. Red, yellow, countdown and vehicle heads are never promoted. `figure_walk_guard` and the 3-of-5 smoother still apply after this.
  - `t_walk` is 0.952, set at the largest validation p(walk) of any non-walk crop plus 0.02, so validation has 0 false walks.
- **Hold veto.** The smoother's history, including its 1 s countdown flicker hold, cannot carry a walk through a frame that the classifier confidently reads as not-walk. This is what fixes the US `countdown_ending` and `walk_symbol_off` frames.
- **Switching off.** `STREETSENSE_SIGNAL_CNN=0` turns all of this off, and the reader falls back to hand rules only. If the model fails to load, the reader also runs hand rules only and never says walk because of the failure.

**External crop test split** (PTL + Commons test, 1,897 crops): accuracy 0.906, walk recall at `t_walk` 0.51, false walks 0. Validation: accuracy 0.910, recall 0.63, false walks 0.

**Results.** Both runs used the same venv. Off = `STREETSENSE_SIGNAL_CNN=0`; on = fd45421.

| split / subset | walk recall off → on | reader walk on non-walk off → on | go (cached Muse) off → on | false go |
|---|---|---|---|---|
| dev/dc_orig | 9/10 → 10/10 | **8/20 → 1/20** | 8/10 → 9/10 | 0 → 0 |
| dev/eu_style | 3/7 → 3/7 | 0/9 → 0/9 | 3/7 → 3/7 | 0 → 0 |
| dev/night | 1/3 → 1/3 | 0/4 → 0/4 | 1/3 → 1/3 | 0 → 0 |
| dev/us_day2 | 2/2 → 2/2 | 0/4 → 0/4 | 2/2 → 2/2 | 0 → 0 |
| heldout/eu_style | 0/3 → 0/3 | 0/5 → 0/5 | 0/3 → 0/3 | 0 → 0 |
| heldout/night | 0/2 → 0/2 | 0/9 → 0/9 | 0/2 → 0/2 | 0 → 0 |
| heldout/new EU (pex, nz) | 0/7 → 0/7 | 0/7 → 0/7 | no Muse cache | – |
| heldout/new stills | 0/2 → **1/2** (uk_green) | 0/1 → 0/1 | no Muse cache | – |
| heldout/dc_orig, vehicle | – | 0/13 → 0/13 | – | – |
| **all** | 15/36 → 17/36 | **8/72 → 1/72** | 14/27 → 15/27 | 0 → 0 |

- **Dense non-walk scan:** 1,148 frames, 0 walk decisions both off and on.
- **US countdown_ending:** the dense DC 48.5–53 s countdown-ending stretch went from 16/19 frames reading walk to **0/19**. One labelled frame, `dc_48.72`, still reads walk, because the classifier is not confident on it. Muse says no on that frame, so go stays 0.

**Environment drift.** The venv was rebuilt (openvino 2026.4.1, ultralytics 8.4.191). With that rebuild, the off run gives 15/22 dev walk recall where 3837a00 reported 17/22. The two changed frames are dc_17.80, where a detection is missing, and eu_ped_b_07.00, where a red read made it unclear. The off → on comparison above is like-for-like.

**Caveats.**
- The held-out gain is one frame (1/14). The NZ, pex and night_jp walks are still missed. Usually the detector finds no head there, or the classifier is under `t_walk`.
- The threshold is strict because the auto-labelled training crops are noisy. A second review pass relabelled 100 lit walk figures, and walk recall at the zero-false-walk threshold went from about 0.2 to about 0.5. Noise is still there, mostly bloom blobs and small heads.
- Most training walks are PTL phone photos, Commons photos and dev crops. Night and US white-figure crops are a small share.
- **Muse rescore from cache ($0, no API calls; the ledger stays at 462 lines).** `eval/muse_local_reader.json` was recomputed with the classifier on.
  - DC set: E, E#1 and E@768 stay at 9/10 fused go on walk frames, with 0 go on non-walk frames. Variant F, which is not shipped, had 3 fused go on non-walk DC frames before because the reader held walk there. It now has 0.
  - Broad set (E@768 and F@768): 6/17 → 5/17 on walk frames, still 0 on non-walk frames. The lost frame is `night_ped@1.00`. **This comes from the local reader not being deterministic, not from the classifier.** The classifier gives p(walk) 0.96 on that head. A rerun of the same 48 broad rows in a fresh process read walk there. Overall, 3 of 48 rows came out differently between runs: `eu_ped_b@7.00`, `night_ped@1.00` and `most04@8.00`. They are near-threshold YOLO detections at night or close up. None of the changes created a walk on a non-walk frame.

## Detectors

The detector benchmark and the opt-in tiled light pass (held-out walk reads 1/14 -> 5/14, 0/1148 dense false walks) are in [DETECTORS.md](DETECTORS.md).

## signal_cnn v1 vs v2 (29 Oct 2026)

v2 (`models/signal_cnn_v2`, `STREETSENSE_SIGNAL_CNN_VERSION=v2`) = v1 data + Commons video train clips.
Reader evals: `scripts/eval_local_reader.py --dense`, logs in `/workspace/datasets/runs/eval/{v1,v2,v1_tiles,v2_tiles}.log`
(tiles = `STREETSENSE_LIGHT_TILES=2x3 STREETSENSE_LIGHT_TILES_MODE=always`). Crop eval: `scripts/compare_signal_cnn.py` → `docs/benchmark/signal_cnn_compare.json`.

| Metric | v1 | v2 | v1 + tiles | v2 + tiles |
|---|---|---|---|---|
| Held-out walk recall (14) | 1/14 | 5/14 | 5/14 | 5/14 |
| Dev walk recall (22) | 16/22 | 17/22 | 16/22 | 16/22 |
| Dev false walk (37) | 1/37 | 1/37 | **0/37** | **1/37** (dc_48.72) |
| All walk recall / misses (36) | 17 / 19 missed | 22 / 14 missed | 21 / 15 | 21 / 15 |
| Held-out false walk (35) | 0 | 0 | 0 | 0 |
| Dense non-walk walk decisions (1073 frames) | 0 | 0 | 0 | 0 |
| US countdown dc 48.5–53 (19 frames) | 0 walk | 0 walk | 0 walk | 0 walk |
| Night walk recall dev / held-out | 1/3, 0/2 | 1/3, 0/2 | 2/3, 0/2 | 2/3, 0/2 |
| Cached-Muse go / false go | 15/27, 0 | 19/27, 0 | 15/27, 0 | 15/27, 0 |

Crop test split (each model at its own t_walk: v1 0.952, v2 0.885):

| Group | v1 acc / walk rec / false walk | v2 acc / walk rec / false walk |
|---|---|---|
| Commons video clips (405) | 0.807 / 6/27 / 0 | 0.807 / 14/27 / 0 |
| cvid:dus_g2r (76, 13 walk) | 0.803 / 0.462 / 0 | 0.921 / 0.846 / 0 |
| cvid:tw_pingtung (37, 14 walk) | 1.000 / 0.000 / 0 | 0.973 / 0.214 / 0 |
| cvid:tw_chaozhou (34) | 0.941 / – / 0 | **0.353** / – / 0 |
| cvid:china / gau / sanjose / us_blinkred | 0.874 / 0.753 / 0.111 / 0.529 | 0.856 / 0.818 / 0.778 / 0.647 |
| Commons stills (356, 45 walk) | 0.862 / 0.289 / 0 | 0.868 / 0.711 / **2** |
| PTL (1541, 80 walk) | 0.916 / 0.650 / 0 | 0.909 / 0.912 / **2** |

Per-class acc on video clips (v1 → v2): walk 1.00→1.00, stop 0.99→0.73, countdown 0.95→0.63, other 0.71→0.84.

**Decision: v1 stays the default; v2 stays opt-in.** v2 improves walk recall without tiles, but it adds false walks:
dc_48.72 (countdown ending) with tiles always (v1+tiles 0/37 → v2+tiles 1/37), plus 2 on Commons stills and 2 on PTL crops.
With tiles always, held-out recall is the same (5/14). Safety rule unchanged: go only if the local reader says walk AND Muse says yes.

## signal_cnn v3 (29 Oct 2026): v2 data + PTL-Crosswalk train crops (licence unconfirmed)

`models/signal_cnn_v3`, `STREETSENSE_SIGNAL_CNN_VERSION=v3`. `trained_on` in its json lists
`ptl_crosswalk_train_LICENCE_UNCONFIRMED`; rollback = v1 (default) or v2. Trained 30 epochs; the epoch is picked on **val**
rec@0fw + acc (never test): epoch 18 (val rec@0fw 0.811, acc 0.898). Val rec@0fw swings 0.03–0.81 between epochs
(ep 17 0.028, ep 18 0.811, ep 23 0.028), so a single-epoch pick is noisy; EMA/averaging of the last epochs is the next
thing to try (not done: only the best state is checkpointed). t_walk 0.965.

PTL-Crosswalk crops (`eval/ptlcw_manifest.csv`, 2,710 labelled frames: day 1713 / night 497 / rain 500):
2,314 no light box, 126 multi-box, 116 vehicle-shaped, 154 single ped box → 94 dropped by the colour check, 1 manual drop
(342NHT, red read as green) → **59 kept** (W 19, S 17, O 23; train 35 / val 5 / test 19). **No rain frame survived.**
Label noise: the label is per image and our box may be a different head; 61% of single-box frames disagreed with our
colour reader. The PTLCW test split is tiny (19 crops, 3 walk), so its numbers are only indicative.

| Metric | v1 | v2 | v3 | v1 + tiles | v2 + tiles | v3 + tiles |
|---|---|---|---|---|---|---|
| Held-out walk recall (14) | 1/14 | 5/14 | 1/14 | 5/14 | 5/14 | 5/14 |
| Dev walk recall (22) | 16/22 | 17/22 | 16/22 | 16/22 | 16/22 | 16/22 |
| Dev false walk (37) | 1/37 | 1/37 | 1/37 (dc_48.72) | **0/37** | 1/37 (dc_48.72) | **1/37** (dc_48.72) |
| All walk recall (36) | 17 | 22 | 17 | 21 | 21 | 21 |
| Held-out false walk (35) | 0 | 0 | 0 | 0 | 0 | 0 |
| Dense non-walk walk decisions | 0 | 0 | 0 | 0 | 0 | 0 |
| US countdown dc 48.5–53 (19 frames) | 0 walk | 0 walk | 0 walk | 0 walk | 0 walk | 0 walk |
| Night walk recall dev / held-out | 1/3, 0/2 | 1/3, 0/2 | 1/3, 0/2 | 2/3, 0/2 | 2/3, 0/2 | 2/3, 0/2 |
| Cached-Muse go / false go | 15/27, 0 | 19/27, 0 | 15/27, 0 | 15/27, 0 | 15/27, 0 | 15/27, 0 |

Crop test split (`signal_ds_v3.pkl`, each at its own t_walk: v1 0.952, v2 0.885, v3 0.965; acc / walk rec / false walk):

| Group | v1 | v2 | v3 |
|---|---|---|---|
| Commons video clips (405, 27 walk) | 0.807 / 6/27 / 0 | 0.807 / 14/27 / 0 | 0.763 / 6/27 / 0 |
| Commons stills (356, 45 walk) | 0.862 / 0.289 / 0 | 0.865 / 0.711 / 2 | 0.857 / 0.267 / 0 |
| PTL (1541, 80 walk) | 0.916 / 0.650 / 0 | 0.909 / 0.912 / 2 | 0.905 / 0.550 / **1** |
| PTLCW test day (13, 2 walk) | 0.692 / 0/2 / 0 | 0.692 / 0/2 / 0 | 0.692 / 0/2 / 0 |
| PTLCW test night (6, 1 walk) | 0.500 / 0/1 / 0 | 0.667 / 0/1 / 0 | 0.833 / 0/1 / 0 |
| PTLCW test rain | 0 crops | 0 crops | 0 crops |

**Decision: v1 stays the default; v3 is opt-in.** v3 fails both parts of the rule. It adds false walks: dc_48.72
with tiles always (v1+tiles 0/37 → v3+tiles 1/37) and 1 PTL test crop (v1 0). Its walk recall is also no better: the same
as v1 or lower (held-out 1/14, PTL crops 0.550 vs 0.650). Logs are in `/workspace/datasets/runs/eval/v3*.log` and `v3.log`. The safety rule is unchanged: go only if the local reader says walk AND Muse says yes.

## PTLD test split (30 Oct 2026): test only, licence unconfirmed

PTLD (PUCRS, Brazil; see DATASETS_CANDIDATES.md #2) is **not** used for training. Labels: go→walk, stop→stop, off→other.
`scripts/get_ptld_crops.py` cuts the GT boxes; `scripts/eval_ptld.py` scores them (STREETSENSE_SAM_FALLBACK=0, no API calls).
Test split: 524 boxes (240 walk / 249 stop / 35 off) in 478 images. Cells = said walk / n.

| Path | v1 (default) | v2 | v3 |
|---|---|---|---|
| Crops, full reader (HSV + CNN + figure guard): walk recall | 31/240 (0.129) | 33/240 (0.138) | 24/240 (0.100) |
| Crops, full reader: false walk on stop / off | 0/249, 0/35 | 0/249, 0/35 | 0/249, 0/35 |
| Crops, CNN alone (p_walk ≥ t_walk): walk recall | 90/240 | 83/240 | 82/240 |
| Crops, CNN alone: false walk on stop / off | 0/249, 0/35 | 0/249, 0/35 | 0/249, 0/35 |
| Full images, Pipeline (detector, 5 repeats, local_signal_state): walk recall | 28/217 | 30/217 | 30/217 |
| Full images: false walk on stop / off images | 0/232, 0/26 | 0/232, 0/26 | 0/232, 0/26 |

HSV rules alone (CNN off, same for all versions) said walk on 11/240 walk, **32/249 stop and 3/35 off** crops; the CNN veto
removed all of them. Image label = walk if any go box, else stop, else off (3 images have no box). Not measured: smoother over
real video, Muse, night/rain split (PTLD has no such tags).

Why recall is low: on walk crops the reader most often says green (v1 139/240) or **yellow (40/240)**. Likely cause (not verified per crop): many Brazilian heads have
yellow housings and green countdown digits above the walk figure; the CNN only promotes green/unknown, so a "yellow" read blocks
it even when the CNN is sure. On full images the detector path mostly ends in unknown (v1 117/217). This is a recall gap, not a
safety gap: no version produced a false walk anywhere on PTLD. Decision unchanged (v1 default). Outputs: `/workspace/datasets/ptld/eval/v*.json`.
