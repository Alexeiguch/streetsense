# StreetSense on Ray-Ban Meta glasses

Status, 28 Oct 2026: **not yet run on glasses.** Everything below was built and
tested on the dev box with recorded street clips reshaped to the glasses camera
stream, and with request fixtures copied from Meta's VisionClaw sample. No
Ray-Ban Meta hardware, phone build or Meta AI app pairing has been used.

## What the platform needs

Sources: the Meta Model API cookbook recipe "Smart glasses with OpenClaw"
(VisionClaw) and the Wearables Device Access Toolkit (DAT) docs.

- **A phone companion app is required.** Third-party code does not run on the
  glasses. The DAT 1.0 SDK is native only: iOS 17.2+ (Swift, Xcode 26.4+) or
  Android 10+ (Kotlin). The glasses camera stream reaches our app over
  Bluetooth through the Meta AI app.
- **Pairing and permissions.** Pair the glasses in the Meta AI app, turn on
  Developer Mode (Settings, App Info, tap the version number 5 times), then
  grant the app camera access ("Always allow"). Developer Mode has to be
  turned on again after firmware updates. Release is through a private beta
  channel.
- **Camera stream.** Portrait frames at high 720x1280, medium 504x896 or low
  360x640, at 2, 7, 15, 24 or 30 fps. On weak Bluetooth the toolkit lowers
  resolution first, then frame rate, and keeps at least 15 fps. Frames are
  compressed per frame. Stream states: stopping, stopped, waitingForDevice,
  starting, streaming, paused. There is also a single high-res photo capture.
- **Audio.** Speech out goes to the glasses speakers over A2DP, so normal
  phone text-to-speech works. The glasses mics use HFP (mono, 8 or 16 kHz),
  which cannot run at the same time as A2DP. In-stream audio is beta only.
- **Testing without hardware.** The toolkit ships a Mock Device Kit for the
  phone app. Supported glasses: Ray-Ban Meta Gen 1 and Gen 2, Oakley Meta
  HSTN and Vanguard, Meta Ray-Ban Display.

## Two paths, and what StreetSense built for each

### Path A: VisionClaw (single photo per question). Built, tested with fixtures.

VisionClaw is Meta's sample iOS app. You speak, it grabs one photo from the
glasses, sends one chat turn (text plus a JPEG data URI, model
`openclaw/default`, bearer token) to an OpenClaw gateway at
`/v1/chat/completions`, and speaks the reply. It has no model key and no tools.

StreetSense now speaks that same protocol, so VisionClaw can point straight at
it instead of an OpenClaw gateway:

```bash
export STREETSENSE_GATEWAY_TOKEN="$(openssl rand -hex 24)"   # same value as GATEWAY_TOKEN in VisionClaw
export MODEL_API_KEY=...                                     # server side only
python -m streetsense serve --port 18789
# VisionClaw Secrets.xcconfig: GATEWAY_HOST=<this machine>, GATEWAY_PORT=18789, GATEWAY_TOKEN=<token>
```

- `GET /health` returns `{"ok":true,"status":"live"}`, like OpenClaw.
- `POST /v1/chat/completions` takes the VisionClaw request shape (copied from
  the sample's `ChatDTOsTests`) and returns a `chat.completion` with the
  spoken answer in `choices[0].message.content`. It is off unless
  `STREETSENSE_GATEWAY_TOKEN` is set, checks the token in constant time, and
  shares the `/ask` rate limit and single-flight guard.
- Routing: crossing questions go through the crossing fusion path. Other
  questions go to the Muse Wonder guide when `STREETSENSE_WONDER=1`, else to a
  Muse Spark description that passes through the same no-go filter.
- **A single photo can never produce "go".** The local light reader needs 3
  agreeing frames before it reports walk, one photo is one frame, and the
  gateway resets the light votes for every photo so photos taken seconds
  apart cannot add up. So the
  VisionClaw path answers "Do not cross" or "I can't tell, please wait" for
  crossings, plus hazard and layout guidance. This is by design.

Tested: `tests/test_gateway_glasses_web.py` (offline) and one live run on the
dev box with a real DC street frame: crossing route 4.5 s, "Do not cross."
(Muse saw red); description route 1.9 s. Not tested: an actual VisionClaw
build, a phone, or the glasses.

### Path B: continuous stream (needed for real-time alerts). Server side built; phone app not built.

Real-time alerts (cars, people, obstacles, the walk signal) need the camera
stream, which means a small custom DAT phone app. The app's job is thin:

1. Start a DAT stream, for example `StreamConfiguration(resolution: .medium, frameRate: 15)`.
2. For each frame, JPEG encode at about quality 60 and `POST /frame?mode=glasses`.
   Keep only the newest frame in flight; drop older ones.
3. Play the returned earcon, then speak the short phrase with the phone TTS
   (it comes out of the glasses speakers over A2DP).
4. On `paused` or `stopped`, stop sending and say "StreetSense paused".

What exists in this repo for path B:

- `streetsense/glasses.py`
  - `GlassesStreamConfig`: the toolkit resolutions and frame rates, with validation.
  - `auto_ladder`: the toolkit's bandwidth fallback order.
  - `StreamState`: the toolkit state names.
  - `adapt_frame`: centre crop to 9:16, resize to the stream size, and a JPEG round trip, so any video looks like a glasses frame.
  - `FramePacer`: keeps the newest frame and drops stale ones.
  - `glasses_cue`: turns an alert into at most 5 words plus an earcon. Light and crossing wording is never shortened. INFO notes are not spoken.
- `/frame?mode=glasses` returns `glasses: {text, earcon, priority, key}`.
- Web glasses mode (`/glasses` or `/?mode=glasses&res=medium&fps=15`) runs the
  same loop in a phone browser. It sends portrait frames at toolkit size and
  pace, plays earcons with Web Audio, and speaks short phrases. With the phone
  held at eye level, this is the closest stand-in we have for a glasses
  point of view.
- `python -m streetsense glasses-sim --source <clip> --resolution medium --fps 15`
  replays a clip as a glasses stream and reports processing time against the
  frame budget.

### Measured with `glasses-sim` (dev box, 8 vCPU, OpenVINO, no Meta calls)

| Clip | Profile | Frames | Process p50 / p95 ms | Frame budget ms | Cues |
|---|---|---|---|---|---|
| Tyumen crossing | medium 504x896 @15 | 121 | 33 / 116 | 67 | road ahead, car coming right, traffic light is green |
| DC pedestrian signal | low 360x640 @15 | 101 | 22 / 91 | 67 | none |
| DC pedestrian signal | medium 504x896 @15 | 101 | 26 / 106 | 67 | traffic light is red |
| DC pedestrian signal | high 720x1280 @24 | 201 | 34 / 138 | 42 | none |

What this shows: the server keeps up with 15 fps at the median, and p95
spikes come from the segmenter frames. At low resolution the small distant
signal head in the DC clip is lost, so **medium @15 fps is the recommended
profile**. These clips were filmed with handheld cameras, not glasses, so
real glasses footage (head motion, wide lens, heavier compression) may differ.

## Honest gaps

- No run on glasses, no phone app for path B, no VisionClaw build.
- The glasses camera field of view is not measured. `hfov_deg` is a 60 degree
  placeholder that only affects box-based distance estimates.
- Bluetooth latency from glasses to phone is not measured. The network hop
  from phone to server adds to it; on-phone inference would remove that hop
  but is not built.
- The HFP mic and A2DP speaker conflict means voice questions and spoken
  alerts cannot overlap. VisionClaw handles questions with on-device speech
  recognition.
