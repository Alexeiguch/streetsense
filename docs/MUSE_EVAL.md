# Muse Spark crossing eval (28 Oct 2026)

Goal: make Muse Spark (`muse-spark-1.3`) say yes on clear WALK frames without ever saying yes on red, flashing, ending-countdown or no-signal frames. All numbers below come from `scripts/eval_muse_crossing.py` runs on 28 Oct 2026. Raw rows are in `docs/benchmark/muse_eval.json` (DC set) and `docs/benchmark/muse_eval_broad.json` (broad set, see [Part 2](#part-2-broader-labelled-set-eu-style-night-second-us-clip)).

**Result:** the new default is variant **E**: the "describe" prompt, one frame, sent at 1024 px. Muse said yes on 9 of 10 walk frames, in both of two runs. It said yes on 0 of 22 non-walk frames. The old default (A: strict prompt, 512 px) said yes on 0 of 10 walk frames.

The safety rule has not changed. The app says go only if the local light reader says `walk` AND Muse says `safe_to_cross=yes`. This eval adds two more blocks to the fusion step (see [Fusion changes](#fusion-changes)).

## Method

- **Labelled set:** `eval/muse_labels.csv` has 40 frames from the sample clips (fetch them with `scripts/get_samples.sh`). I labelled every frame by eye from zoomed crops. The contact sheet is at `docs/benchmark/muse_eval_labels.jpg`.
  - DC frames: I measured the walk-figure and hand pixels in the signal head, frame by frame, and picked times well inside each phase. Each extracted frame was checked again after extraction.
  - On this clip the walk figure blinks: it is lit for about 12 of every 25 video frames, probably LED flicker beating against the camera. So a single still can show only the red countdown digits during the walk phase. Those frames have their own class, `walk_symbol_off`.
- **Local reader:** for each sample, a fresh `Pipeline` (no Muse, no surface model) runs over t−1.0 s to t in 0.25 s steps. Its state at t is the `local` input to fusion. Cached in `eval/muse_local_reader.json`.
- **Muse request:** the system prompt for the variant, the question "Is it safe to cross the street right now?", and 1 or 3 JPEG frames. Settings: `reasoning_effort=minimal`, `max_tokens=512` for the rows in the first table (the default is now 768, see [Part 1](#part-1-max_tokens)).
- **Scoring:** a reply is checked with `validate_crossing_reply`. Malformed replies count as `invalid`, and the app treats them as "I can't tell, please wait". The fused decision comes from `fuse_crossing_decision`.
- **Caching:** each response is stored in `eval/muse_cache/<sha256 of the exact request>.json`, so a rerun costs nothing. `--repeat N` adds a salt to the key so the same requests are paid for again; the E#1 row used this to check repeatability.
- **Logging:** every paid call was appended to `outputs/meta_usage.jsonl`. The runs used `STREETSENSE_META_BUDGET_USD=3.20` (spend before this task plus the $3 cap).

### Label counts

| Label | n | Source | What it is |
|---|---|---|---|
| `walk` | 10 | DC | White walking person lit, countdown 48 down to 13 |
| `walk_symbol_off` | 4 | DC | Walk phase, but the walk figure is dark in this frame; only the countdown (45/33/21/13) shows. Reported separately, not scored. |
| `countdown_ending` | 4 | DC | Walking person lit, countdown 10/9/8/7. Must not be yes. |
| `flashing` | 5 | DC | Flashing hand + countdown 5 to 2 (one frame has the hand dark). Must not be yes. |
| `red` | 7 | DC | Steady hand (one frame shows hand + 0). Must not be yes. |
| `no_signal` | 6 | CDMX ×4, Mazatlán, Tomaszów | No readable pedestrian signal. Must not be yes. |
| `unclear` | 4 | Tyumen | Far two-lamp head with a green countdown; the lower lamp sometimes shows a small green figure or arrow. Reported, not scored. |

The hard requirement covers 22 frames: red, flashing, countdown_ending and no_signal.

**Tyumen label change (please read).** I first labelled Tyumen `no_signal` because I took the head for a vehicle light. When B and E described "a green walking person on the centre pole", I zoomed in. It may be a Russian pedestrian head (countdown on top, green figure below), but I cannot confirm it at this size. I moved all four frames to `unclear` before choosing a variant. Under the first label, A and B would each have had 1 or 2 "false yes" answers there. The local reader called the same head `green` or `unknown`, never `walk`, so fused go stayed at 0.

## Results

Each row covers all 40 frames. "Walk yes" is Muse alone; "walk go" is the fused decision.

| Variant | Prompt | Frames | Image side | Walk yes (Muse) | Walk go (fused) | False yes, non-walk (n=22) | Fused go, non-walk | walk_symbol_off yes (n=4) | unclear yes (n=4) | Unsure + invalid (of 40) | Invalid | Mean / p95 latency | Mean prompt tokens | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A (old default) | strict | 1 | 512 | 0/10 | 0/10 | 0 | 0 | 0/4 | 1/4 | 13/40 | 0 | 2.00 / 2.87 s | 360 | $0.0562 |
| B | describe | 1 | 512 | 0/10 | 0/10 | 0 | 0 | 0/4 | 2/4 | 16/40 | 1 | 2.01 / 3.12 s | 631 | $0.0657 |
| C | describe | 3 (t−1, t−0.5, t) | 512 | 0/10 | 0/10 | 0 | 0 | 0/4 | 3/4 | 9/40 | 1 | 2.75 / 4.48 s | 1018 | $0.0856 |
| G | strict | 1 | 1024 | 1/10 | 1/10 | 0 | 0 | 0/4 | 2/4 | 16/40 | 3 | 2.74 / 4.16 s | 957 | $0.0962 |
| **E (new default)** | describe | 1 | 1024 | **9/10** | **9/10** | **0** | **0** | 0/4 | 3/4 | 9/40 | 1 | 2.48 / 3.72 s | 1228 | $0.0916 |
| E#1 (repeat of E) | describe | 1 | 1024 | 9/10 | 9/10 | 0 | 0 | 0/4 | 3/4 | 10/40 | 2 | 2.22 / 3.33 s | 1228 | $0.0976 |
| F | describe | 3 (t−1, t−0.5, t) | 1024 | 8/10 | 8/10 | 0 | 0 | 3/4 | 3/4 | 7/40 | 3 | 2.68 / 4.19 s | 2809 | $0.1784 |

Each cell below is yes / no / unsure / invalid:

| Variant | walk | walk_symbol_off | countdown_ending | flashing | red | no_signal | unclear |
|---|---|---|---|---|---|---|---|
| A | 0/10/0/0 | 0/4/0/0 | 0/3/1/0 | 0/4/1/0 | 0/4/3/0 | 0/1/5/0 | 1/0/3/0 |
| B | 0/10/0/0 | 0/2/1/1 | 0/3/1/0 | 0/4/1/0 | 0/3/4/0 | 0/0/6/0 | 2/0/2/0 |
| C | 0/10/0/0 | 0/4/0/0 | 0/4/0/0 | 0/5/0/0 | 0/5/2/0 | 0/0/6/0 | 3/0/0/1 |
| G | 1/1/6/2 | 0/4/0/0 | 0/3/1/0 | 0/5/0/0 | 0/7/0/0 | 0/1/5/0 | 2/0/1/1 |
| E | 9/1/0/0 | 0/1/2/1 | 0/4/0/0 | 0/5/0/0 | 0/7/0/0 | 0/1/5/0 | 3/0/1/0 |
| E#1 | 9/1/0/0 | 0/1/2/1 | 0/4/0/0 | 0/4/0/1 | 0/7/0/0 | 0/1/5/0 | 3/0/1/0 |
| F | 8/1/0/1 | 3/0/0/1 | 0/4/0/0 | 0/5/0/0 | 0/7/0/0 | 0/2/4/0 | 3/0/0/1 |

Variant D (multi-frame plus a text summary of the local reader history) is defined in the script but was **not run**. Telling Muse what the local reader saw weakens the independence of the two checks, and F already shows what extra frames do.

### What the runs show

- **Image size is the main factor, not the prompt alone.**
  - At 512 px the DC walk figure is only a few pixels. In A, B and C, Muse called it a "red hand" on all 10 walk frames; it was reading the red countdown digits.
  - At 1024 px the old strict prompt (G) still mostly said unsure (1/10 yes).
  - The describe prompt at 1024 px (E) said "white walking person lit with red countdown N" and answered yes on 9 of 10.
- **The one walk miss is the same frame in both E runs.** On `dc_45.72` the countdown reads 13, but Muse read it as 8 and said no. That is a safe-direction error.
- **Countdown reading is approximate.** Muse read 37 as 31 on one frame and 10 as 0 on another. On all 9 countdown_ending and flashing frames it read 10 or less and said no.
- **Multi-frame (F)** turned 3 of the 4 `walk_symbol_off` frames into yes, which is correct for that phase, with no new false yes. But it costs about 2× (2809 vs 1228 prompt tokens), adds about 0.2 s mean latency, and had 3 invalid (truncated) replies vs 1. Its walk yes-rate was 8/10 vs 9/10.

## Chosen variant and wiring

The selection rule was: zero false yes first, then the highest walk yes-rate, then cost and latency. E won: 0 false yes and 9/10, against 8/10 for F at twice the cost.

| Setting | Default | Env switch |
|---|---|---|
| Crossing prompt | `describe` | `STREETSENSE_MUSE_PROMPT=strict` restores the v0 prompt (unknown names also fall back to strict) |
| Image size for `ask_sync` (on-demand questions) | 1024 px | `STREETSENSE_MUSE_CROSSING_SIDE` |
| Background 10 s summary loop | 512 px (unchanged) | |
| History frames | 0 (single frame) | `STREETSENSE_MUSE_HISTORY_FRAMES=2` sends the frames from about 1.0 s and 0.5 s ago (spacing `history_spacing_s=0.5`) |

How history frames work:
- When history is on, `Pipeline` keeps a short `FrameHistory` buffer (`streetsense/frame_history.py`), and `answers.crossing_answer` passes those frames to `ask_sync(..., history=...)`.
- If no recent frames are buffered, it falls back to one frame.

Model settings: `reasoning_effort=minimal` (unchanged) and `max_tokens=768` (was 512; env `STREETSENSE_MUSE_MAX_TOKENS`, see [Part 1](#part-1-max_tokens)).

### Fusion changes

Both changes make the rule stricter; neither loosens it:
- If Muse reports `countdown_seconds` ≤ 10 (`MIN_WALK_COUNTDOWN_S`), the decision is **stop**, even when the local reader says walk and Muse says yes. The local reader called all 4 countdown_ending frames `walk`, so before this block, only Muse saying no prevented a go on them.
- `pedestrian_signal` values `countdown` and `flashing` now block go.
- `countdown_seconds` is optional. If it is present and is not a number or null, the reply is invalid and the app says "I can't tell, please wait".

## Cost (first run)

- **This task:** $0.6714 across 280 paid calls (A + B + C + E + E#1 + F + G).
- **Project total:** the ledger went from $0.2040 to **$0.8753**, against $150 of credits.
- **Per call:** variant E costs about $0.0023 at 1024 px, vs about $0.0014 for A at 512 px.

## Limitations

- **Small n, one walk clip (first run).** All walk, red, flashing and countdown frames come from one US signal head in one DC clip, in daylight, with the camera pointed up at the signal. There are no EU green-figure, night, rain, distant-head or glasses-height walk examples. 9/10 is not a general accuracy figure. Part 2 adds EU-style, night and a second US clip, but n is still small.
- **Labels are mine, from looking at frames.** There is one labeller and no second check. The Tyumen class is uncertain and was relabelled after Muse's answers (see above).
- **Muse is not deterministic.** E ran twice with the same yes counts in every class (one extra invalid on flashing in the repeat); other variants ran once.
- **Truncation.** With `max_tokens=512`, 1 to 3 of every 40 replies were cut off (completion_tokens hit 512, so the JSON was incomplete). They fall back safely to "can't tell", but they lower the yes-rate. Fixed in [Part 1](#part-1-max_tokens): at 768, E had 0 truncations on 88 frames. The 3-frame variant F still truncated 2 of 48 at 768.
- **Untested combination.** Multi-frame was tested in the eval harness only. In the live app, the history frames come from whatever `/frame` stream the session pipeline has seen, and that path has unit tests but no paid end-to-end run.
- **Countdown misreads.** A misread *upwards* near the threshold (for example 9 read as 19) would remove the countdown block. That did not happen on 9 ending or flashing frames, but n is small.

## Part 1: max_tokens

Muse's reasoning tokens share `max_tokens` with the JSON answer, so a long reasoning trace cuts the JSON off (`finish_reason=length`). The app treats that as invalid: "I can't tell, please wait". I reran E on the same 40 DC frames with `--max-tokens 768`. The cache key includes `max_tokens`, so these were fresh paid calls. Truncated means `finish_reason == "length"`, or completion_tokens ≥ max_tokens for older cache entries that did not record finish_reason.

| Run | max_tokens | Truncated (of 40) | Walk yes (Muse) | Walk go (fused) | False yes, non-walk (n=22) | Fused go, non-walk | Unsure + invalid | Mean / max completion tokens | Mean / p95 latency | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| E | 512 | 1 | 9/10 | 9/10 | 0 | 0 | 9/40 | 273 / 512 | 2.48 / 3.72 s | $0.0916 |
| E#1 | 512 | 2 | 9/10 | 9/10 | 0 | 0 | 10/40 | 304 / 512 | 2.22 / 3.33 s | $0.0976 |
| **E@768** | **768** | **0** | 9/10 | 9/10 | 0 | 0 | 10/40 | 306 / 659 | 2.23 / 3.76 s | $0.0988 |

E@768 per label (yes/no/unsure/invalid): walk 9/1/0/0, walk_symbol_off 0/1/3/0, countdown_ending 0/4/0/0, flashing 0/5/0/0, red 0/7/0/0, no_signal 0/1/5/0, unclear 2/0/2/0. The walk miss is again `dc_45.72` (countdown 13 read as 8, reply no), which is a safe-direction error.

**Decision: 768.** It is the smallest limit tried with zero truncations, so 1024 was not run. Cost per call barely changes because you pay only for the tokens actually generated. On this set the extra headroom did not change the walk yes-rate: the 512 truncations had landed on walk_symbol_off and flashing frames, not on walk frames. The margin is not large: the longest reply was 659 of 768 tokens. On the broad set (Part 2), E@768 again had 0 truncations on 48 frames (longest 622), but F (3 frames) hit 768 twice.

- Default: `VisionLLMConfig.max_tokens = 768`.
- Override: `STREETSENSE_MUSE_MAX_TOKENS=<n>`. Empty, non-numeric or 0 falls back to 768.
- `MetaModelClient` now records `finish_reason` so truncation shows in logs and evals.
- Tests: `test_reasoning_minimal_and_default_max_tokens_768`, `test_max_tokens_env_switch` (5 cases), `test_truncated_reply_is_recorded_and_means_wait`.

## Part 2: broader labelled set (EU-style, night, second US clip)

### Method

- **Clips:** none of the clips already in the repo had a usable EU or night pedestrian head (CDMX, Mazatlán and Tomaszów have no readable head, and the Tyumen head is far and uncertain). I sourced 15 short clips from Wikimedia Commons (CC BY 4.0 / CC0) and Pexels (Pexels License). Every clip's URL, author and licence is in `eval/SOURCES.md`. Videos are not committed. `scripts/get_eval_clips.sh` downloads them into `samples/ext/` and checks their sha256.
- **Labels:** `eval/muse_labels_broad.csv` has 48 frames with id, subset, clip, time, label, confidence and note. I labelled each frame by eye from zoomed crops. The contact sheet is `docs/benchmark/muse_eval_broad_labels.jpg`. Mixed-head frames (one head red, another green with a low countdown) are `unclear` and not scored.
- **Subsets:**
  - `eu_style`: European-style heads (red/green man), daytime. Only the Most (CZ) clip has a confirmed EU location; the Pexels clips do not say where they were filmed.
  - `night`: night clips plus two Most clips filmed at dusk (21:15).
  - `us_day2`: a second US hand/man head with a countdown, to check the DC result on another head.
- **Runs:** E@768 (the new default) and F@768 (3 frames) through the same harness and cache, with `STREETSENSE_META_BUDGET_USD=3.875` (ledger at the start of this task plus $3).
- **Local reader:** same method as above, cached in `eval/muse_local_reader.json`.

### Label counts (broad set)

| Subset | walk | red | flashing | countdown_ending | no_signal | unclear | Total |
|---|---|---|---|---|---|---|---|
| eu_style | 10 | 7 | 0 | 3 | 4 | 0 | 24 |
| night | 5 | 8 | 0 | 0 | 2 | 3 | 18 |
| us_day2 | 2 | 1 | 3 | 0 | 0 | 0 | 6 |

The hard requirement covers 28 non-walk frames: 16 red, 3 flashing, 3 countdown_ending and 6 no_signal.

### Results per subset

"Walk yes" is Muse alone. "Walk go" is the fused decision (local reader `walk` AND Muse yes). DC day is E@768 on the original 40 frames.

| Subset | Variant | n | Walk yes (Muse) | Walk go (fused) | False yes, non-walk | Fused go, non-walk | Unclear yes (fused go) | Truncated | Unsure + invalid | Mean / p95 latency | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DC day | E@768 | 40 | 9/10 | 9/10 | **0/22** | **0** | 2/4 (0) | 0 | 10/40 | 2.23 / 3.76 s | $0.0988 |
| EU-style | E@768 | 24 | 10/10 | 0/10 † | **0/14** | **0** | n/a | 0 | 4/24 | 1.81 / 2.23 s | $0.0539 |
| EU-style | F@768 | 24 | 10/10 | 0/10 † | **0/14** | **0** | n/a | 1 | 3/24 | 1.94 / 3.53 s | $0.0938 |
| Night | E@768 | 18 | 5/5 | 0/5 | **0/10** | **0** | 2/3 (0) | 0 | 2/18 | 1.93 / 2.89 s | $0.0424 |
| Night | F@768 | 18 | 4/5 | 0/5 | **0/10** | **0** | 2/3 (0) | 1 | 3/18 | 2.23 / 4.38 s | $0.0714 |
| US day 2 | E@768 | 6 | 2/2 | 2/2 | **0/4** | **0** | n/a | 0 | 0/6 | 1.58 / 1.85 s | $0.0141 |
| US day 2 | F@768 | 6 | 2/2 | 2/2 | **0/4** | **0** | n/a | 0 | 0/6 | 1.78 / 1.93 s | $0.0266 |

Per label on the broad set (yes/no/unsure/invalid):

| Variant | walk | countdown_ending | flashing | red | no_signal | unclear |
|---|---|---|---|---|---|---|
| E@768 | 17/0/0/0 | 0/3/0/0 | 0/3/0/0 | 0/16/0/0 | 0/0/6/0 | 2/1/0/0 |
| F@768 | 16/0/0/1 | 0/2/0/1 | 0/3/0/0 | 0/16/0/0 | 0/2/4/0 | 2/1/0/0 |

**No false yes.** Across both variants, Muse said yes on 0 of the 28 non-walk frames, and the fused decision was never go on a non-walk or unclear frame. F's two invalid replies were both truncations at 768 (`most11_03.00`, countdown_ending; `night_jp_01.10`, walk). Both fell back to "I can't tell, please wait".

**Muse yes on `unclear` frames (not scored, reported for transparency).** Both are dusk frames from Most with several heads in view, and the fused result was not go in either case:
- `most02_02.00` (E): "Small lower box with green walking person lit". The near head is red and a farther head shows a green man with countdown 4. Local `green` → stop.
- `most03_00.20` (E and F): "green walking person lit on multiple pedestrian heads". The near head may show countdown 1 and turns red within 0.5 s. Local `unknown` → unsure.
- `most04_08.00` (F): "lower green walking person lit with 35 countdown". The near head is red. Local `green` → stop.

These show that Muse can pick the wrong head when several are in view. The local reader requirement is what blocked them.

### Local reader on the broad set

| Subset / label | Local reader states |
|---|---|
| eu_style walk (10) | green 3, red 2, unclear 1, unknown 4 |
| eu_style red (7) | red 4, unclear 1, unknown 2 |
| eu_style countdown_ending (3) | green 3 |
| eu_style no_signal (4) | unknown 4 |
| night walk (5) | unknown 5 |
| night red (8) | red 2, unknown 6 |
| night no_signal (2) | green 1, unknown 1 |
| night unclear (3) | green 1, unclear 1, unknown 1 |
| us_day2 walk (2) | walk 2 |
| us_day2 flashing (3) | red 3 |
| us_day2 red (1) | red 1 |

- The local reader never said `walk` on a non-walk frame. Its one wrong-but-safe call is `eu_ped_b_04.50`: walk labelled, local `red`, Muse yes → stop.
- It does not recognise EU green-man heads as `walk` (0 of 10; it was 1 of 10 before the white-rule fix below), and it recognised none of the 5 night walk frames (all `unknown`). On EU heads it often returns `green`, a lit green head not classified as a pedestrian walk figure, which never allows go.
- So on EU-style and night footage the bottleneck is the local reader, not Muse. Fused go was 0/10 on EU walk frames (1/10 before the fix) and 0/5 at night, against Muse alone at 10/10 and 5/5.
- The fusion is unchanged. The local reader got one trivial tightening (see the next section); otherwise it is unchanged. Improving the reader's EU and night recall is the next step if those settings matter. It needs its own labelled tests so it never says `walk` on red.

† **Re-scored 2026-10-28 after the signal-distance work** (cached Muse replies, no new calls). The one EU walk "go" (`eu_ped_b_06.00`) was right for the wrong reason: the local `walk` came from a side-on, unlit *vehicle* head, where white sky between the visors matched the US white-walk-figure rule. That rule now skips heads classified as vehicle, so the frame reads `red` (from the true walk head's washed-out figure plus ghost red) and fuses to stop. The distance rules also turned some local `green` into `unclear` (`tyumen_06/12/18`, `most04_08.00`, all labelled unclear) and `eu_ped_b_02.80` (red) into `unclear`. DC-day results per label are unchanged. Why the reader misses EU and night walks: [SIGNAL_DISTANCE.md](SIGNAL_DISTANCE.md#why-the-local-reader-rarely-says-walk-on-eu-and-night-frames).

### Part 2 caveats

- Small n (48 frames, 17 walk) and one labeller. The Pexels clips are stock footage, often close-up and centred on the head, so they are easier than glasses-height footage.
- The "EU-style" location is confirmed only for Most (CZ). The other clips show European-style heads but do not state where they were filmed.
- "Night" includes 2 dusk frames from Most. `night_ped_08.50` (red) is medium confidence because the head is at the frame edge.
- Pexels may serve a different rendition later. `get_eval_clips.sh` then fails the sha256 check rather than silently relabelling.

## Cost (this follow-up)

- **This follow-up:** $0.3896 across 131 paid calls (ledger 0.875323 → 1.264950), against the $3 cap. Per-variant costs in the tables are summed from each row's token usage, including rows answered from cache, so they add up to slightly more than the ledger delta.
- **Project total:** $1.2650.

## Reproduce

```bash
scripts/get_samples.sh   # clips are not committed
STREETSENSE_META_BUDGET_USD=<spent+cap> .venv/bin/python scripts/eval_muse_crossing.py --variants A,B,C,E,F,G
# cached: reruns cost $0; add --repeat 2 to pay for a fresh sample
.venv/bin/python scripts/eval_muse_crossing.py --variants E --max-tokens 768
scripts/get_eval_clips.sh   # broad-set clips into samples/ext (sha256-checked)
.venv/bin/python scripts/eval_muse_crossing.py --labels eval/muse_labels_broad.csv --variants E,F --max-tokens 768 --out docs/benchmark/muse_eval_broad.json
```
