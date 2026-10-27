# StreetSense × Meta Model API

How StreetSense uses Meta’s Muse / SAM / Voice models for the Meta Global AI Developer Hackathon.
Measured on this box with `MODEL_API_KEY` set (2026-10-27, Europe/London). Costs are estimates from [Meta pricing](https://dev.meta.ai/docs/pricing-rate-limits/).

## Auth and base URL

| | |
|---|---|
| Base URL | `https://api.meta.ai/v1` |
| Key env | `MODEL_API_KEY` (never logged, never written to disk) |
| Budget env | `STREETSENSE_META_BUDGET_USD` (default **$20** for dev; hard-refuses further calls) |
| Usage ledger | `outputs/meta_usage.jsonl` (gitignored via `outputs/*`) |

OpenAI-compatible Chat Completions for Muse Spark; Responses API for SAM 3.1; `POST /v1/asr/transcribe` for Muse Voice.

## Models we actually called

From `GET /v1/models` on this key:

`muse-spark-1.3`, `muse-spark-1.2`, `muse-spark-1.1`, contributor variants, `muse-wonder-1.0`, `muse-image-1.0`, `muse-voice-transcribe-1.0`, **`sam-3.1`**.

StreetSense uses:

| Model id | Role in StreetSense |
|---|---|
| `muse-spark-1.3` | Slow vision loop + on-demand **“Can I cross?”** (strict JSON) |
| `sam-3.1` | Text-prompted masks/boxes for `crosswalk` (and on-demand curb / tactile paving / sidewalk) |
| `muse-voice-transcribe-1.0` | Optional spoken “can I cross?” in the phone web demo |

## Muse Spark vision

- Module: `streetsense/meta_api.py` → `MetaVisionAssistant`
- CLI: `python -m streetsense ask --image frame.jpg --question "Can I cross?"`
- Web: `POST /ask` (fuses with the local light reader; see safety rule below)
- Pipeline: background request every ~10 s when the key is present

**Important:** Muse Spark always reasons. Reasoning tokens share `max_tokens` with the visible answer. We set `reasoning_effort="minimal"` and `max_tokens=512`. With the default effort and a small `max_tokens`, `content` comes back empty (`finish_reason=length`, almost all tokens in `reasoning_tokens`).

Images are JPEG-downscaled (`max_image_side=512` by default) and sent only in **user** messages.

### Measured latency / cost (smoke + eval)

| Call | Latency | Tokens (prompt / completion) | Est. cost |
|---|---|---|---|
| Smoke `dc_06.0s` “Can I cross?” | **~1.8 s** | ~334 / ~299 | **~$0.0017** |
| Eval mean over 8 frames | **~1.35 s** | ~330 / ~200–300 | **~$0.0013 / call** |

Pricing used: Spark standard $1.25 / $4.25 per 1M input / output tokens.

## Safety fusion (local light ⊕ Muse)

`fuse_crossing_decision(local_state, muse_json)` in `meta_api.py`:

> **Never say go unless the local pedestrian reader says `walk` AND Muse says `safe_to_cross=yes`.**  
> If either says stop / unsure / red / countdown / none → **stop** or **unsure**.

Vehicle `green` is **not** treated as a walk signal.

### Small-sample agreement (n=8 frames)

Frames: DC red ×2, DC countdown, DC walk, DC green/walk, Tyumen no-light ×2, CDMX night crowd.

| Metric | Result |
|---|---|
| Agreement on go/not-go (comparable) | **3 / 6** |
| Fusion “go” when either source said stop/unsure | **0** (good) |
| Muse said `yes` on a local-`walk` frame in this sample | **0** (Muse was more conservative / sometimes swapped ped vs vehicle) |
| Muse said `yes` with no local light (Tyumen crowd) | 1 → fusion stayed **unsure** |

Honest takeaway: on this tiny sample Muse is a useful second opinion and a scene summariser; the **hard safety gate stays local∩Muse**. Do not ship Muse alone as a walk/don’t-walk oracle.

Raw rows: `docs/benchmark/meta_crossing_eval.json`.

## SAM 3.1

- Module: `MetaSamSegmenter` (Responses API, `metadata.mask_encoding=one_bit`)
- CLI: `python -m streetsense sam --image frame.jpg --prompts crosswalk,curb --out overlay.png`
- Pipeline: optional `--sam` flag; every `SamConfig.interval_s` (default 8 s) runs the configured prompts (default just `crosswalk`) and fills in a classical miss via `sam_boxes_to_crosswalk`

### Measured

| Frame | Prompt | Boxes | Latency | Cost |
|---|---|---|---|---|
| Tyumen crossing | crosswalk | 1 | **1.77 s** | **$0.0025** |
| Tyumen | curb / sidewalk | 0 | 0.7–1.3 s | $0.0025 each |
| DC walk | crosswalk, curb, tactile paving, sidewalk | 0 | 0.3–2.0 s | $0.0025 each |

Overlay: `docs/benchmark/sam_tyumen_before_after.png` (before | after).  
JSON: `docs/benchmark/sam_tyumen.json`, `docs/benchmark/sam_dc.json`.

SAM image price: **$2.50 / 1k images**. Masks decode via optional `meta-sam-parser` when installed.

## Muse Voice Transcribe (optional)

- `POST /v1/asr/transcribe`, model `muse-voice-transcribe-1.0`, mono 16/24 kHz PCM WAV
- Web demo: **Hold to ask (voice)** → `/transcribe` (ffmpeg remux webm→wav) → `/ask?q=...`
- Price: **$0.18 / hour** of audio (~$0.00005 for a 1 s utterance)
- Smoke (gTTS → 24 kHz WAV “Can I cross?”): transcript **`Can I cross?`**, latency **~0.81 s**, cost **~$0.00007**

## Budget discipline

- Ledger line per call: model, tokens / images / audio seconds, estimated USD, latency
- CLI `ask` / `sam` print a running `meta_usage` total
- Hard cap via `STREETSENSE_META_BUDGET_USD` (default $20). Hackathon credits are **$150** total — keep exploratory spend in the low dollars.

## Creative use for judging (25% Meta tooling)

StreetSense is not a thin chat wrapper:

1. **Fast local loop** (YOLO / lights / zebra / surface) for sub-second alerts  
2. **Muse Spark** as a slow, structured second opinion with an explicit dual-veto safety rule  
3. **SAM 3.1** as a text-prompted crosswalk prior when classical stripes fail  
4. **Voice Transcribe** so a blind user can ask “can I cross?” without finding a button  

Together that is a glasses-ready perception stack on the same Meta Model API key.
