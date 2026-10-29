# SAM 3.1 signal fallback: limits and decisions

Code: `streetsense/sam_fallback.py` (`SamFallbackController`, `SamSignalFallback`). It is wired into `Pipeline.process` through `_inject_sam_lights`.
Tests: `tests/test_sam_fallback.py` and `tests/test_sam_fallback_controls.py`. Latency study: [SAM_LATENCY.md](SAM_LATENCY.md).

## What it does

When there is a crossing context and the local detector has no confident, readable signal, the controller sends
the full frame to SAM 3.1 (prompt `traffic light`, 768 px longest side, JPEG q85) in the background. Boxes that come back
fresh (≤ 1.5 s old) and that no local light box overlaps (IoU < 0.3) are added as extra `traffic light` candidates,
tagged `attrs["source"]="sam_fallback"`. These candidates go through the same reader, signal CNN, smoother and
`select_crossing_signal` as YOLO boxes.

**Safety rule (unchanged).** Go is said only when the local reader says walk and a validated Muse Spark reply says yes
(`answers.crossing_answer` → `fuse_crossing_decision`). SAM returns boxes and confidence only, with no state field. It
cannot authorise a crossing. Any doubt means wait. These tests check that: `test_sam_boxes_alone_cannot_produce_go` and
`test_sam_crop_read_as_walk_still_needs_muse_yes`. In the second test, even when the SAM crop is forced to read as walk, a Muse reply of
error, unsure or no never gives go.

## Limits

| Limit | Env var | Default | Why |
|---|---|---|---|
| On/off | `STREETSENSE_SAM_FALLBACK` | on only if `MODEL_API_KEY` is set; `0` forces off; `1` without a key stays off | The caps below bound the cost of a session, and the fallback only adds candidates. With no key there is nothing to call. |
| Confident light | `STREETSENSE_SAM_CONFIDENT_CONF` | 0.35 | Same threshold as the zoom-out pass. A local light at or above it with a readable state means SAM is not needed. |
| Maps crossing radius | `STREETSENSE_SAM_CROSSING_RADIUS_M` | 45 m | Context opens on the approach (the navigator previews at about 40–50 m). |
| "Can I cross?" hold | `STREETSENSE_SAM_CHECK_HOLD_S` | 20 s | A crossing question keeps the context open while the user waits at the kerb. |
| Context grace | `STREETSENSE_SAM_CONTEXT_GRACE_S` | 5 s | The crossing closes, and its counters reset, only after the context has been gone this long. This stops the context flickering. |
| Stop once answered | `STREETSENSE_SAM_STABLE_FRAMES` | 5 frames | A readable smoothed signal from a confident light for 5 frames in a row means this crossing is answered. No more SAM calls until the context ends. |
| Near-duplicate skip | `STREETSENSE_SAM_DUP_THRESH` | 6.0 | Mean absolute difference of 32×32 grey thumbnails against the last SAM frame. Below it, the last result is reused and no call is made. |
| Calls per minute | `STREETSENSE_SAM_MAX_PER_MIN` | 6 | Hard ceiling of $0.90/h even in a pathological session. |
| Calls per crossing | `STREETSENSE_SAM_MAX_PER_CROSSING` | 3 | One frame rarely helps after 2–3 looks. After that, the local reader and Muse decide (or the app waits). |
| Session spend | `STREETSENSE_SAM_SESSION_USD` | $0.50 | About 200 calls. When reached, SAM disables itself for the session. |
| Global budget | `STREETSENSE_META_BUDGET_USD` (existing ledger) | $20 | Checked before every call. When reached, SAM disables itself. |
| Result max age | `STREETSENSE_SAM_MAX_AGE_S` | 1.5 s | p90 latency is about 1.3 s. Older boxes may point at a head the camera has moved off. |
| Minimum interval | `STREETSENSE_SAM_MIN_INTERVAL_S` | 1.0 s | About 1 Hz at most, and never more than one call in flight. |

When a cap is hit:
- **Per-crossing or per-minute cap:** SAM skips the call and logs once per crossing.
- **Session or global budget cap:** SAM disables itself for the rest of the session and logs once (`[sam-fallback] disabled (...)`).
- **API errors:** they give no boxes.

In every case the app carries on with local detection only. The crossing answer is then whatever local and Muse give, and unclear means wait.

## Decisions

- **Fallback only, never primary.** About 0.8 s median and 1.3 s p90 from the box (more on a phone network) is too slow for the 5–10 fps loop.
  On held-out frames, SAM recall (67%) is similar to the local detectors. It is useful as a second look only when the local detector found nothing it can read.
- **Why caps.** At $0.0025/call, a naive 1 Hz loop costs $9/h. The caps make the worst case $0.90/h (per-minute cap), and a session can never spend more than $0.50.
- **Why 768 px full frame.** In SAM_LATENCY.md, 512 px and q70 lost 4–5 of 69 GT signals. 1024 and 1280 px gave no clear recall gain. Latency was flat across sizes because it is server-bound.
  Cropping the upper 60% dropped recall to 48%. The YOLO-region crop was no better than the full frame.
- **Why one call in flight.** Three parallel calls added about 100 ms each and did not make anything faster. Only the newest frame matters, and queued calls would only return stale boxes.
  The OpenAI client keeps its connection open; a new connection costs about 190 ms.
- **Why the default is on with a key.** Spend per session is capped at $0.50 and per hour at $0.90. Measured realistic use is about $0.005 per crossing. SAM cannot change a decision towards go.
  `STREETSENSE_SAM_FALLBACK=0` turns it off.

## Hooks and accounting

- `Pipeline.note_crossing_check()`: called automatically by `answers.crossing_answer` (/ask, glasses, Wonder).
- `Pipeline.note_crossing_distance(m)`: the maps layer should call this with the distance to the next crossing.
  **It is not yet called from the web/maps session.** Until it is, the triggers are crossing questions and detected crosswalks.
- A detected crosswalk ahead (`CrosswalkResult.found`) also opens the context.
- Per frame, `out.timings["sam_fallback"]` holds the controller time in ms.
- `Pipeline.sam_fallback_summary()` returns calls, USD, reused near-duplicates, cap skips (frames), errors, boxes injected, and per-crossing `{calls, usd, answered_t}`.
  It appears in the CLI stats JSON (`"sam_fallback"`), and the web app prints it when a session expires. Every billed call is also in `outputs/meta_usage.jsonl`.

## Measured (scripts/sim_sam_fallback.py, 29 Oct 2026)

The simulation replays real clips through the real Pipeline (YOLO11s OpenVINO, reader, CNN, smoother) at 5 fps in stream time. Setup:
- Each crossing clip is one crossing: maps reports it at 30 m for the whole clip, and the user asks "can I cross?" 3 s in.
- Between crossings there are 60 s of walking with no context.
- A scooter clip with no maps context comes last.

| run | crossings | session (stream time) | SAM calls | calls per crossing | spend | per hour at this density |
|---|---|---|---|---|---|---|
| mock (no boxes back; upper bound), 10 crossings + 1 non-crossing clip | 10 | 953.6 s | 19 | 1,3,1,1,3,3,1,1,3,2 (mean 1.9, max 3) | $0.0475 (estimate, $0 spent) | $0.18/h (38 crossings/h) |
| real SAM 3.1, first 4 crossings + scooter | 4 | 512.4 s | 6 | 1,3,1,1 | $0.015 (ledger) | — |

- The real run measured latency of 1.389, 1.517, 0.959, 0.964, 1.191 and 0.461 s; the median is about 1.08 s.
  In the real run, SAM boxes reached the reader 12 times (`boxes_injected`).
- In both runs, the non-crossing scooter clip made 0 calls.
- The per-crossing cap was hit on 4 of 10 crossings in the mock run (night and dusk clips where the local reader never settles).

**Cost by case**
- Naive SAM at 1 Hz: $9/h.
- Worst case with these caps: $0.90/h (per-minute cap), stopping at $0.50 per session.
- Realistic, from the simulation: about $0.005 per crossing. That is about $0.18/h even at a dense 38 crossings/h. A typical city walk with 6–12 crossings/h would be about $0.03–0.06/h (extrapolated).

## Not measured

- The phone or LTE network path. All latency here is from the dev box, which sits behind a proxy.
- How much SAM boxes improve end-to-end answers. The mock run returns no boxes, and the real run is too small to say.
- How the maps context behaves with real GPS. The maps distance was fixed at 30 m per clip because the web session does not call the hook yet.
- How often the near-duplicate skip fires on a real, steady phone. 38–40 reuses in the mock run came from stream-time replay.
- Rate limits above 3 parallel calls.
