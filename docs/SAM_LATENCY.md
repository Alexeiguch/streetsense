# SAM 3.1 latency (Meta Model API) — measured 2026-10-29

Script: `scripts/sam_latency_bench.py` (rtt / variants / fresh / concurrency / report). Prompt `traffic light`, same
request body and metadata as `MetaSamSegmenter.segment`. 20 held-out frames from `/workspace/det_bench` with ≥1 GT
signal (69 GT objects). Recall = GT objects matched by any SAM box at IoU ≥ 0.3 (no confidence threshold).
Each variant ran once per frame, sequentially, variant by variant, over one keep-alive httpx connection. No interleaving,
so drift over time sits on top of the differences between variants. Cost is the ledger estimate, $0.0025/call. Every call is in
`outputs/meta_usage.jsonl`. Raw results are in `outputs/sam_latency/` (cache, rtt.json, concurrency.json, report.md).

**Caveat:** this box reaches the internet through a sandbox egress proxy, so TCP connect (~0.1 ms) is to the local
proxy, not to api.meta.ai. The network numbers do not stand for the user's phone or LTE path. Single run with n=20 per variant: median gaps under ~150 ms are within noise
(sequential p90 is ~1.4 s).

## 1. Breakdown (median per call)

| variant | resize ms | JPEG ms | b64 ms | json ms | TCP done ms | TLS done ms | body sent ms | headers recv (TTFB) ms | body read ms | parse ms | resp KB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full1280_q85 | 2.1 | 2.0 | 0.10 | 0.23 | 2.1 | 17.1 | 0.7 | 884 | 0.2 | 0.06 | 1.7 |
| full1024_q85 | 2.0 | 1.4 | 0.08 | 0.18 | n/a (reused) | n/a (reused) | 0.5 | 698 | 0.2 | 0.06 | 1.6 |
| full768_q85 | 2.6 | 0.9 | 0.08 | 0.16 | n/a (reused) | n/a (reused) | 0.5 | 781 | 0.2 | 0.06 | 1.2 |
| full512_q85 | 2.7 | 0.5 | 0.05 | 0.12 | n/a (reused) | n/a (reused) | 0.6 | 822 | 0.2 | 0.06 | 1.1 |
| full768_q90 | 1.6 | 1.0 | 0.11 | 0.22 | n/a (reused) | n/a (reused) | 0.6 | 702 | 0.2 | 0.06 | 1.3 |
| full768_q70 | 1.8 | 0.8 | 0.06 | 0.13 | n/a (reused) | n/a (reused) | 0.5 | 645 | 0.2 | 0.06 | 1.2 |
| top60_1024_q85 | 2.0 | 0.9 | 0.05 | 0.13 | n/a (reused) | n/a (reused) | 0.5 | 697 | 0.2 | 0.06 | 1.2 |
| yolo_768_q85 | 1.2 | 0.9 | 0.08 | 0.15 | n/a (reused) | n/a (reused) | 0.5 | 973 | 0.2 | 0.06 | 1.6 |
| full768_q85 (fresh) | 1.6 | 1.0 | 0.09 | 0.21 | 1.6 | 31.6 | 33.1 | 967 | 0.2 | 0.06 | 1.1 |

"TCP/TLS done" and "body sent" are cumulative ms from request start (httpx trace). "n/a (reused)" means the connection was already open.
Measured on the first call (1280) and on 10 fresh-connection calls. Response headers include only `x-request-id` and `x-route`.
The server reports no timing header, so server compute and queueing cannot be split out. TTFB = network + queue + inference.

Network baseline (`rtt.json`, n=10): DNS 1.6 ms. TCP connect 0.1 ms (to the proxy). Fresh connection TLS done after
32 ms median (p90 117). `GET /v1/models` (tiny, unbilled): fresh connection 511 ms median (p90 1477). Reused connection
202 ms median (p90 342). So about 200 ms of every call is fixed API/front-end overhead even for a trivial request. SAM
inference plus queueing is then roughly 450–700 ms (inferred, not measured directly). Client work (resize + JPEG + base64 + JSON) is about 5 ms.
Upload of a payload under 100 KB is under 1 ms here. A fresh connection added about 190 ms median (967 vs 781 ms, at 768 px q85).

## 2. Variants (same 20 frames, same prompt)

| variant | frames | req WxH (median) | payload KB | median ms | p90 ms | recall IoU>=0.3 | dets/frame | $/call |
|---|---|---|---|---|---|---|---|---|
| full1280_q85 | 20 | 720x1280 | 84 | 885 | 1765 | 47/69 (68%) | 2.6 | 0.0025 |
| full1024_q85 | 20 | 576x1024 | 83 | 698 | 1282 | 46/69 (67%) | 2.5 | 0.0025 |
| full768_q85 | 20 | 432x768 | 79 | 781 | 1276 | 46/69 (67%) | 2.5 | 0.0025 |
| full512_q85 | 20 | 288x512 | 43 | 822 | 1436 | 42/69 (61%) | 2.2 | 0.0025 |
| full768_q90 | 20 | 432x768 | 99 | 702 | 1188 | 44/69 (64%) | 2.5 | 0.0025 |
| full768_q70 | 20 | 432x768 | 54 | 646 | 1176 | 42/69 (61%) | 2.3 | 0.0025 |
| top60_1024_q85 | 20 | 1024x345 | 48 | 697 | 961 | 33/69 (48%) | 1.8 | 0.0025 |
| yolo_768_q85 | 14 | 768x509 | 74 | 974 | 1344 | 39/53 (74%) | 3.1 | 0.0025 |
| full768_q85 (fresh conn) | 10 | 432x768 | 81 | 967 | 1644 | 27/41 (66%) | 3.1 | 0.0025 |

- The source frames are mostly 1280×720, so `full1280` is the native size.
- `top60`: rows 0–60 % of the frame. In several clips (e.g. eu_ped_side and night frames) the signal sits lower, so recall drops a lot.
- `yolo`: union of y11s_zoom traffic-light boxes (conf ≥ 0.25) plus a margin. It only exists on 14 of the 20 frames. On those 14 frames: yolo 39/53 at 974 ms median, full768 40/53 at 781 ms, full1280 40/53 at 843 ms. The crop gives no recall gain and is not faster.
- No size or quality change gave a latency gain larger than the noise. Payload size has almost no effect on latency, because the time is server-side.
  512 px and q70 cost 4–5 GT objects of recall.

## 3. Concurrency

6 rounds of 3 parallel 768 px calls (18 calls), all HTTP 200, no 429 and no rate-limit headers. Per-call median was 888 ms
(p90 1260), against 765 ms median (p90 1358) for all 164 sequential calls. Wall time per round of 3 was 1298 ms median. Running 3 calls in parallel
adds about 100 ms per call, which is roughly the noise level, and gives about 2.3 calls/s in total. No rate limit was hit at this low volume.
Higher volume was not tested (budget).

## 4. Recommendation for a background fallback

Use full frame at 768 px longest side, JPEG q85 (the existing `MetaSamSegmenter` default), prompt `traffic light`,
over a persistent client (the OpenAI client already keeps the connection alive; never open a fresh one per call). Keep one call in flight and
send at most about 1 call/s, only when the local detector has no confident light. Expect about 0.8 s median and about 1.3 s p90 from here, plus the phone
network. That is far too slow for anything except a background "look harder" hint. Cost is $0.0025/call, so 1 Hz is about $9/hour.
Gate it hard: for example, only after N frames with no light near a crossing. Cropping (top-60 %, or around a YOLO region) is not worth it.

Prototype: `streetsense/sam_fallback.py` (`SamSignalFallback`). It runs in a background thread, rate-limited, one call in flight,
and drops stale results (>1.5 s). It returns only `SignalBoxes` (boxes plus confidence, no state field). Errors give no boxes. Its
boxes may only feed the local reader, which still needs Muse Spark "yes" before any go. Any doubt means wait. It is not wired into
`pipeline.py`. Tests are in `tests/test_sam_fallback.py` (mocked API).

## Spend

This study cost $0.455 (182 billed calls). The project ledger went from $1.907 to $2.362 total.
