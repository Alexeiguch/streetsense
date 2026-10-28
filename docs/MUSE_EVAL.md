# Muse Spark crossing eval (28 Oct 2026)

Goal: make Muse Spark (`muse-spark-1.3`) say yes on clear WALK frames without ever saying yes on red, flashing, ending-countdown or no-signal frames. All numbers below come from `scripts/eval_muse_crossing.py` runs on 28 Oct 2026. Raw rows are in `docs/benchmark/muse_eval.json`.

**Result:** the new default is variant **E**: the "describe" prompt, one frame, sent at 1024 px. Muse said yes on 9 of 10 walk frames, in both of two runs. It said yes on 0 of 22 non-walk frames. The old default (A: strict prompt, 512 px) said yes on 0 of 10 walk frames.

The safety rule has not changed. The app says go only if the local light reader says `walk` AND Muse says `safe_to_cross=yes`. This eval adds two more blocks to the fusion step (see [Fusion changes](#fusion-changes)).

## Method

- **Labelled set:** `eval/muse_labels.csv` has 40 frames from the sample clips (fetch them with `scripts/get_samples.sh`). I labelled every frame by eye from zoomed crops. The contact sheet is at `docs/benchmark/muse_eval_labels.jpg`.
  - DC frames: I measured the walk-figure and hand pixels in the signal head, frame by frame, and picked times well inside each phase. Each extracted frame was checked again after extraction.
  - On this clip the walk figure blinks: it is lit for about 12 of every 25 video frames, probably LED flicker beating against the camera. So a single still can show only the red countdown digits during the walk phase. Those frames have their own class, `walk_symbol_off`.
- **Local reader:** for each sample, a fresh `Pipeline` (no Muse, no surface model) runs over t−1.0 s to t in 0.25 s steps. Its state at t is the `local` input to fusion. Cached in `eval/muse_local_reader.json`.
- **Muse request:** the system prompt for the variant, the question "Is it safe to cross the street right now?", and 1 or 3 JPEG frames. Settings: `reasoning_effort=minimal`, `max_tokens=512`.
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

Model settings are unchanged: `reasoning_effort=minimal` and `max_tokens=512`.

### Fusion changes

Both changes make the rule stricter; neither loosens it:
- If Muse reports `countdown_seconds` ≤ 10 (`MIN_WALK_COUNTDOWN_S`), the decision is **stop**, even when the local reader says walk and Muse says yes. The local reader called all 4 countdown_ending frames `walk`, so before this block, only Muse saying no prevented a go on them.
- `pedestrian_signal` values `countdown` and `flashing` now block go.
- `countdown_seconds` is optional. If it is present and is not a number or null, the reply is invalid and the app says "I can't tell, please wait".

## Cost

- **This task:** $0.6714 across 280 paid calls (A + B + C + E + E#1 + F + G).
- **Project total:** the ledger went from $0.2040 to **$0.8753**, against $150 of credits.
- **Per call:** variant E costs about $0.0023 at 1024 px, vs about $0.0014 for A at 512 px.

## Limitations

- **Small n, one walk clip.** All walk, red, flashing and countdown frames come from one US signal head in one DC clip, in daylight, with the camera pointed up at the signal. There are no EU green-figure, night, rain, distant-head or glasses-height walk examples. 9/10 is not a general accuracy figure.
- **Labels are mine, from looking at frames.** There is one labeller and no second check. The Tyumen class is uncertain and was relabelled after Muse's answers (see above).
- **Muse is not deterministic.** E ran twice with the same yes counts in every class (one extra invalid on flashing in the repeat); other variants ran once.
- **Truncation.** With `max_tokens=512`, 1 to 3 of every 40 replies were cut off (completion_tokens hit 512, so the JSON was incomplete). They fall back safely to "can't tell", but they lower the yes-rate. Raising `max_tokens` was not tested.
- **Untested combination.** Multi-frame was tested in the eval harness only. In the live app, the history frames come from whatever `/frame` stream the session pipeline has seen, and that path has unit tests but no paid end-to-end run.
- **Countdown misreads.** A misread *upwards* near the threshold (for example 9 read as 19) would remove the countdown block. That did not happen on 9 ending or flashing frames, but n is small.

## Reproduce

```bash
scripts/get_samples.sh   # clips are not committed
STREETSENSE_META_BUDGET_USD=<spent+cap> .venv/bin/python scripts/eval_muse_crossing.py --variants A,B,C,E,F,G
# cached: reruns cost $0; add --repeat 2 to pay for a fresh sample
```
