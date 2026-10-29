"""SAM 3.1 latency breakdown + variant benchmark (Meta Model API, POST /v1/responses).

Usage: .venv/bin/python scripts/sam_latency_bench.py {rtt|variants|fresh|concurrency|report}
Every billable call is checked against and recorded in outputs/meta_usage.jsonl (UsageLedger, hard cap via
STREETSENSE_META_BUDGET_USD) and cached under outputs/sam_latency/cache/. Never prints the API key.
Request body mirrors MetaSamSegmenter.segment (same metadata params).
"""
from __future__ import annotations
import base64, hashlib, json, os, socket, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, httpx, numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streetsense.meta_api import parse_sam_boxes
from streetsense.meta_usage import UsageRecord, estimate_cost, get_ledger

B = Path("/workspace/det_bench"); OUT = ROOT / "outputs/sam_latency"; CACHE = OUT / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
BASE = "https://api.meta.ai/v1"; MODEL = "sam-3.1"; PROMPT = "traffic light"; TASK_CAP = 1.0
PER = estimate_cost(MODEL, images=1)
SPENT_F = OUT / "task_spend.json"

def spent() -> float: return json.load(open(SPENT_F))["usd"] if SPENT_F.exists() else 0.0
def add_spent(x):
    json.dump({"usd": round(spent() + x, 6)}, open(SPENT_F, "w"))

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i; return i / u if u > 0 else 0

def gt_objects():
    gt = json.load(open(B / "gt.json")); O = {}
    for k, L in gt.items():
        objs = []
        for g in sorted([g for g in L if g["ok"]], key=lambda g: -(g["box"][2]-g["box"][0])*(g["box"][3]-g["box"][1])):
            if not any(iou(g["box"], o) >= 0.3 for o in objs): objs.append(g["box"])
        O[k] = objs
    return O

def frames(n=20):
    I = json.load(open(B / "index.json")); O = gt_objects()
    ks = sorted({r["cur"] for r in I["rows"] if r["split"] == "heldout" and O.get(r["cur"])})
    step = max(1, len(ks) // n); ks = ks[::step][:n]
    return [(k, I["frames"][k]) for k in ks], O

def yolo_region(k, W, H, margin=0.5):
    d = json.load(open(B / "dets/y11s_zoom.json"))["dets"].get(k, [])
    d = [x for x in d if x[5] == "traffic light" and x[4] >= 0.25]
    if not d: return None
    x1 = min(x[0] for x in d); y1 = min(x[1] for x in d); x2 = max(x[2] for x in d); y2 = max(x[3] for x in d)
    w, h = x2 - x1, y2 - y1; m = max(w, h, 64) * margin + 48
    return [int(max(0, x1 - m)), int(max(0, y1 - m)), int(min(W, x2 + m)), int(min(H, y2 + m))]

# name -> (max_side, quality, crop)
VARIANTS = {
    "full1280_q85": (1280, 85, None), "full1024_q85": (1024, 85, None), "full768_q85": (768, 85, None),
    "full512_q85": (512, 85, None), "full768_q90": (768, 90, None), "full768_q70": (768, 70, None),
    "top60_1024_q85": (1024, 85, "top60"), "yolo_768_q85": (768, 85, "yolo"),
}

def prepare(img, k, side, q, crop):
    H, W = img.shape[:2]; off = (0, 0)
    if crop == "top60": img = img[: int(H * 0.6)]
    elif crop == "yolo":
        r = yolo_region(k, W, H)
        if r is None: return None
        img = img[r[1]:r[3], r[0]:r[2]]; off = (r[0], r[1])
    t0 = time.perf_counter(); h, w = img.shape[:2]; s = min(1.0, side / max(h, w))
    if s < 1: img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    t1 = time.perf_counter(); ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q]); t2 = time.perf_counter()
    url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(); t3 = time.perf_counter()
    body = json.dumps({"model": MODEL, "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": PROMPT}, {"type": "input_image", "image_url": url}]}],
        "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"}}).encode()
    t4 = time.perf_counter()
    return dict(body=body, off=off, scale=(w / img.shape[1], h / img.shape[0]), req_wh=(img.shape[1], img.shape[0]),
                resize_ms=(t1-t0)*1e3, jpeg_ms=(t2-t1)*1e3, b64_ms=(t3-t2)*1e3, json_ms=(t4-t3)*1e3,
                jpeg_bytes=len(buf), payload_bytes=len(body))

def headers():
    return {"Authorization": "Bearer " + os.environ["MODEL_API_KEY"], "Content-Type": "application/json"}

def timed_post(client: httpx.Client, body: bytes, note: str):
    led = get_ledger(); led.check_budget(PER)
    if spent() + PER > TASK_CAP + 1e-9: raise RuntimeError("task cap $1 reached")
    ev = {}; t0 = time.perf_counter()
    def trace(name, info): ev[name] = (time.perf_counter() - t0) * 1e3
    req = client.build_request("POST", BASE + "/responses", content=body, headers=headers(), extensions={"trace": trace})
    r = client.send(req, stream=True); t_hdr = (time.perf_counter() - t0) * 1e3
    raw = r.read(); t_body = (time.perf_counter() - t0) * 1e3
    led.record(UsageRecord(ts=time.time(), model=MODEL, kind="sam", images=1, latency_s=round(t_body / 1e3, 3),
                           estimated_cost_usd=round(PER, 6), note=f"prompt={PROMPT} latency-bench {note}"))
    add_spent(PER)
    tp0 = time.perf_counter(); j = json.loads(raw); text = "".join(c.get("text", "") for o in j.get("output", []) if o.get("type") == "message"
                                                         for c in o.get("content", []) if c.get("type") == "output_text")
    boxes = parse_sam_boxes(text); parse_ms = (time.perf_counter() - tp0) * 1e3
    g = lambda a: ev.get(f"{a}.complete")
    tim = {k: v for k, v in r.headers.items() if any(s in k.lower() for s in ("timing", "time", "duration", "x-envoy", "latency", "processing", "server", "x-request", "cf-ray", "via"))}
    return dict(status=r.status_code, total_ms=t_body, headers_ms=t_hdr, connect_ms=g("connection.connect_tcp"), tls_ms=g("connection.start_tls"),
                sent_ms=g("http11.send_request_body") or g("http2.send_request_body"), resp_bytes=len(raw), parse_ms=parse_ms,
                boxes=boxes, timing_headers=tim, http_version=r.http_version)

def run_one(client, k, path, vname, fresh=False):
    side, q, crop = VARIANTS[vname]; tag = f"{k}__{vname}" + ("__fresh" if fresh else "")
    cf = CACHE / (hashlib.md5(tag.encode()).hexdigest()[:12] + ".json")
    if cf.exists(): return json.load(open(cf))
    img = cv2.imread(path); H, W = img.shape[:2]; p = prepare(img, k, side, q, crop)
    if p is None: return None
    if fresh:
        with httpx.Client(timeout=60, http2=False) as c: res = timed_post(c, p["body"], tag)
    else: res = timed_post(client, p["body"], tag)
    dets = []
    for b in res["boxes"]:
        sx = p["req_wh"][0] / b["w"] if b.get("w") else 1; sy = p["req_wh"][1] / b["h"] if b.get("h") else 1
        f = (p["scale"][0], p["scale"][1])
        dets.append([b["x1"]*sx*f[0] + p["off"][0], b["y1"]*sy*f[1] + p["off"][1], b["x2"]*sx*f[0] + p["off"][0], b["y2"]*sy*f[1] + p["off"][1], float(b.get("c", 1.0))])
    p.pop("body"); out = dict(key=k, variant=vname, fresh=fresh, W=W, H=H, dets=dets, **p, **{x: y for x, y in res.items() if x != "boxes"})
    json.dump(out, open(cf, "w")); return out

def rtt(n=10):
    host = "api.meta.ai"; t0 = time.perf_counter(); ip = socket.getaddrinfo(host, 443)[0][4][0]; dns = (time.perf_counter()-t0)*1e3
    tcp = []
    for _ in range(n):
        t0 = time.perf_counter(); s = socket.create_connection((ip, 443), timeout=5); tcp.append((time.perf_counter()-t0)*1e3); s.close()
    fresh_models, reuse_models, fresh_conn, fresh_tls = [], [], [], []
    for _ in range(n):
        ev = {}; t0 = time.perf_counter()
        with httpx.Client(timeout=20) as c:
            c.get(BASE + "/models", headers=headers(), extensions={"trace": lambda nm, i: ev.__setitem__(nm, (time.perf_counter()-t0)*1e3)})
        fresh_models.append((time.perf_counter()-t0)*1e3); fresh_conn.append(ev.get("connection.connect_tcp.complete")); fresh_tls.append(ev.get("connection.start_tls.complete"))
    with httpx.Client(timeout=20) as c:
        c.get(BASE + "/models", headers=headers())
        for _ in range(n):
            t0 = time.perf_counter(); r = c.get(BASE + "/models", headers=headers()); reuse_models.append((time.perf_counter()-t0)*1e3)
        hdr = {k: v for k, v in r.headers.items() if k.lower() not in ("set-cookie",)}
    res = dict(dns_ms=dns, tcp_connect_ms=tcp, fresh_get_models_ms=fresh_models, fresh_tcp_done_ms=fresh_conn, fresh_tls_done_ms=fresh_tls,
               reuse_get_models_ms=reuse_models, models_response_headers=hdr)
    json.dump(res, open(OUT / "rtt.json", "w"), indent=1); print(json.dumps({k: (statistics.median(v) if isinstance(v, list) and v and v[0] is not None else v) for k, v in res.items() if k != "models_response_headers"}, indent=1))

def variants():
    F, _ = frames(); 
    with httpx.Client(timeout=60) as c:
        for v in VARIANTS:  # variant-major order; interleaving would be fairer but one pass each keeps cost fixed
            for k, p in F:
                r = run_one(c, k, p, v); print(v, k, r and round(r["total_ms"]), r and len(r["dets"]), "spent", round(spent(), 4), flush=True)

def fresh():
    F, _ = frames()
    for k, p in F[:10]:
        r = run_one(None, k, p, "full768_q85", fresh=True); print("fresh", k, round(r["total_ms"]), flush=True)

def concurrency():
    F, _ = frames(); res = []
    with httpx.Client(timeout=60, limits=httpx.Limits(max_connections=4)) as c:
        for k, p in F[:3]: run_one(c, k, p, "full768_q85")  # warm (cached from variants)
        for rnd in range(6):
            ks = F[rnd*3 % len(F): rnd*3 % len(F) + 3]
            def go(kp):
                img = cv2.imread(kp[1]); pr = prepare(img, kp[0], 768, 85, None)
                cf = CACHE / f"conc_{rnd}_{kp[0]}.json"
                if cf.exists(): return json.load(open(cf))
                t = timed_post(c, pr["body"], f"conc r{rnd} {kp[0]}"); t.pop("boxes"); json.dump(t, open(cf, "w")); return t
            t0 = time.perf_counter()
            with ThreadPoolExecutor(3) as ex: out = list(ex.map(go, ks))
            wall = (time.perf_counter()-t0)*1e3
            res.append(dict(round=rnd, wall_ms=wall, per_call_ms=[o["total_ms"] for o in out], status=[o["status"] for o in out]))
            print(res[-1], flush=True)
    json.dump(res, open(OUT / "concurrency.json", "w"), indent=1)

def pct(v, q): return float(np.percentile(v, q)) if v else float("nan")

def report():
    F, O = frames(); keys = [k for k, _ in F]; rows = []; cache = [json.load(open(f)) for f in CACHE.glob("*.json") if not f.name.startswith("conc_")]
    by = {}
    for r in cache: by.setdefault((r["variant"], r["fresh"]), {})[r["key"]] = r
    print("| variant | frames | req WxH (median) | payload KB | median ms | p90 ms | recall IoU>=0.3 | dets/frame | $/call |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (v, fr), d in sorted(by.items(), key=lambda x: (x[0][1], list(VARIANTS).index(x[0][0]))):
        ks = [k for k in keys if k in d]; lat = [d[k]["total_ms"] for k in ks]
        tp = sum(any(iou(x[:4], o) >= 0.3 for x in d[k]["dets"]) for k in ks for o in O[k]); n = sum(len(O[k]) for k in ks)
        wh = sorted(d[k]["req_wh"][0] * d[k]["req_wh"][1] for k in ks); mid = [d[k]["req_wh"] for k in ks if d[k]["req_wh"][0]*d[k]["req_wh"][1] == wh[len(wh)//2]][0]
        print(f"| {v}{' (fresh conn)' if fr else ''} | {len(ks)} | {mid[0]}x{mid[1]} | {statistics.median([d[k]['payload_bytes'] for k in ks])/1024:.0f} | {statistics.median(lat):.0f} | {pct(lat,90):.0f} | {tp}/{n} ({tp/max(n,1):.0%}) | {sum(len(d[k]['dets']) for k in ks)/len(ks):.1f} | {PER:.4f} |")
    print()
    print("| variant | resize ms | JPEG ms | b64 ms | json ms | TCP done ms | TLS done ms | body sent ms | headers recv (TTFB) ms | body read ms | parse ms | resp KB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (v, fr), d in sorted(by.items(), key=lambda x: (x[0][1], list(VARIANTS).index(x[0][0]))):
        L = list(d.values()); m = lambda f: statistics.median([f(x) for x in L if f(x) is not None]) if any(f(x) is not None for x in L) else None
        fm = lambda x: "n/a (reused)" if x is None else f"{x:.1f}"
        print(f"| {v}{' (fresh)' if fr else ''} | {m(lambda x: x['resize_ms']):.1f} | {m(lambda x: x['jpeg_ms']):.1f} | {m(lambda x: x['b64_ms']):.2f} | {m(lambda x: x['json_ms']):.2f} | {fm(m(lambda x: x['connect_ms']))} | {fm(m(lambda x: x['tls_ms']))} | {fm(m(lambda x: x['sent_ms']))} | {m(lambda x: x['headers_ms']):.0f} | {m(lambda x: x['total_ms']-x['headers_ms']):.1f} | {m(lambda x: x['parse_ms']):.2f} | {m(lambda x: x['resp_bytes'])/1024:.1f} |")
    hdrs = {}
    for r in cache:
        for k2 in r["timing_headers"]: hdrs.setdefault(k2, r["timing_headers"][k2])
    print("\ntiming-ish headers seen:", hdrs)
    print("task spend", spent())

if __name__ == "__main__":
    {"rtt": rtt, "variants": variants, "fresh": fresh, "concurrency": concurrency, "report": report}[sys.argv[1]]()
