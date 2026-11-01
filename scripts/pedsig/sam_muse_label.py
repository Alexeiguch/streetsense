#!/usr/bin/env python3
"""Offline labelling helper: SAM 3.1 proposes pedestrian-signal boxes, Muse Spark verifies each crop.

Labels only; SAM and Muse here never touch the crossing decision (the app still says "go" only if the local
reader says walk AND Muse says yes on the live frame). Every paid call goes through streetsense.meta_api /
meta_usage, so it is logged to outputs/meta_usage.jsonl and refused past STREETSENSE_META_BUDGET_USD.
Responses are cached on disk (keyed by model, prompt, file, settings), so no call is paid for twice.
The API key is read from MODEL_API_KEY by the SDK client and never printed or written.

  STREETSENSE_META_BUDGET_USD=<cap> python scripts/pedsig/sam_muse_label.py --select ptld_miss --n 20 \
      --out /workspace/datasets/pedsig/pilot.jsonl
Selections (frames from scripts/pedsig/find_misses.py; train/val sources only, never a test split):
  ptld_miss     PTLD train frames where the default detector misses >= 1 GT box (IoU < 0.3 at conf >= 0.35)
  ptld_val_miss same on PTLD valid
  commons_nolight  commons train frames with no teacher traffic light >= 0.35
  cvid_nolight     commons-video train frames with no teacher traffic light >= 0.35
  cvid_all         every saved commons-video train frame (boxes new vs teacher are the gain)
"""
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from streetsense.meta_api import MetaSamSegmenter, encode_jpeg_data_url  # noqa: E402
from streetsense.meta_usage import BudgetExceeded, get_ledger, usage_from_openai_response  # noqa: E402

P = Path("/workspace/datasets/pedsig")
CACHE = P / "cache"
MUSE_MODEL = "muse-spark-1.3"
VERIFY_SYS = "You check image crops for a training set. Reply with exactly one word: yes or no."
VERIFY_Q = ("Is the object in the centre of this crop a PEDESTRIAN traffic signal head: the box that shows a walking "
            "person, a standing person or a hand (lit or unlit, any country)? Round vehicle traffic lamps, arrows, "
            "street lamps, signs, windows and anything else are 'no'.")


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def tl(f, conf=0.35):
    return [t[:4] for t in f["teacher"] if t[4] == "traffic light" and t[5] >= conf]


def select(frames, how, n, seed):
    if how in ("ptld_miss", "ptld_val_miss"):
        sp = "train" if how == "ptld_miss" else "val"
        c = [f for f in frames if f["src"] == "ptld" and f["split"] == sp and f["gt"]
             and any(max([iou(g[:4], d) for d in tl(f)] or [0]) < 0.3 for g in f["gt"])]
    elif how == "commons_nolight":
        c = [f for f in frames if f["src"] == "commons" and f["split"] == "train" and not tl(f)]
    elif how == "cvid_nolight":
        c = [f for f in frames if f["src"].startswith("cvid") and not tl(f)]
    elif how == "cvid_all":
        c = [f for f in frames if f["src"].startswith("cvid")]
    else:
        raise SystemExit("unknown selection")
    random.Random(seed).shuffle(c)
    return c[:n], len(c)


def cached(kind, key, fn):
    p = CACHE / kind / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if p.exists():
        return json.loads(p.read_text()), True
    out = fn()
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out))
    return out, False


def crop_for_muse(im, b, pad=0.6, side=320):
    H, W = im.shape[:2]; x1, y1, x2, y2 = b; w, h = x2 - x1, y2 - y1
    m = max(w, h) * pad + 8
    X1, Y1, X2, Y2 = int(max(0, x1 - m)), int(max(0, y1 - m)), int(min(W, x2 + m)), int(min(H, y2 + m))
    c = im[Y1:Y2, X1:X2]
    s = side / max(c.shape[:2])
    return cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True); ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="pedestrian traffic signal"); ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--sam-min-conf", type=float, default=0.3); ap.add_argument("--max-boxes", type=int, default=6)
    ap.add_argument("--no-muse", action="store_true"); ap.add_argument("--muse-max-tokens", type=int, default=256); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frames = json.load(open(P / "frames.json"))
    sel, pool = select(frames, a.select, a.n, a.seed)
    print(f"selection {a.select}: {len(sel)} of {pool} candidates; ledger before ${get_ledger().total_usd:.4f}", flush=True)
    sam = MetaSamSegmenter(max_image_side=a.max_side)
    from openai import OpenAI
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ.get("MODEL_API_KEY"), timeout=60)
    spent = {"sam": 0.0, "muse": 0.0, "sam_calls": 0, "muse_calls": 0, "cache_hits": 0}
    with open(a.out, "w") as fo:
        for f in sel:
            im = cv2.imread(f["path"]); H, W = im.shape[:2]

            def do_sam():
                s = sam.segment(im, a.prompt, decode_masks=False)
                return {"boxes": s.boxes, "latency_s": s.latency_s, "cost_usd": s.cost_usd}
            try:
                sr, hit = cached("sam", f"sam-3.1|{a.prompt}|{f['path']}|{a.max_side}", do_sam)
            except BudgetExceeded as e:
                print("STOP budget:", e); break
            spent["cache_hits"] += hit
            if not hit:
                spent["sam"] += sr["cost_usd"]; spent["sam_calls"] += 1
            boxes = []
            for b in sr["boxes"]:
                sx, sy = W / float(b.get("w") or W), H / float(b.get("h") or H)
                boxes.append([b["x1"] * sx, b["y1"] * sy, b["x2"] * sx, b["y2"] * sy, float(b.get("c") or 0.0)])
            boxes = sorted([b for b in boxes if b[4] >= a.sam_min_conf], key=lambda b: -b[4])[: a.max_boxes]
            rec = dict(path=f["path"], src=f["src"], split=f["split"], wh=[W, H], gt=f.get("gt"), sam_raw_n=len(sr["boxes"]),
                       sam_latency_s=sr["latency_s"], boxes=[])
            for b in boxes:
                v = None
                if not a.no_muse:
                    url = encode_jpeg_data_url(crop_for_muse(im, b[:4]), 320, quality=90)

                    def do_muse():
                        get_ledger().check_budget(0.002)
                        t0 = time.perf_counter()
                        r = client.chat.completions.create(
                            model=MUSE_MODEL, max_tokens=a.muse_max_tokens, reasoning_effort="minimal",
                            messages=[{"role": "system", "content": VERIFY_SYS},
                                      {"role": "user", "content": [{"type": "text", "text": VERIFY_Q},
                                                                   {"type": "image_url", "image_url": {"url": url}}]}])
                        lat = time.perf_counter() - t0
                        u = usage_from_openai_response(r, model=MUSE_MODEL, kind="chat", latency_s=lat, note="pedsig_verify")
                        u.images = 1  # one crop sent; usage_from_openai_response leaves images at 0 for chat calls
                        get_ledger().record(u)
                        return {"text": (r.choices[0].message.content or ""), "finish": r.choices[0].finish_reason,
                                "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                                "cost_usd": u.estimated_cost_usd, "latency_s": round(lat, 3)}
                    key = f"{MUSE_MODEL}|{a.muse_max_tokens}|{VERIFY_Q}|{f['path']}|{[round(x) for x in b[:4]]}"
                    try:
                        mr, hit = cached("muse", key, do_muse)
                    except BudgetExceeded as e:
                        print("STOP budget:", e); break
                    spent["cache_hits"] += hit
                    if not hit:
                        spent["muse"] += mr["cost_usd"]; spent["muse_calls"] += 1
                    t = mr["text"].strip().lower()
                    v = "yes" if t.startswith("yes") else ("no" if t.startswith("no") else "unclear")
                    rec["boxes"].append(dict(box=[round(x, 1) for x in b[:4]], sam_conf=round(b[4], 3), muse=v,
                                             muse_tokens=[mr["prompt_tokens"], mr["completion_tokens"]], muse_finish=mr["finish"]))
                else:
                    rec["boxes"].append(dict(box=[round(x, 1) for x in b[:4]], sam_conf=round(b[4], 3), muse=None))
            fo.write(json.dumps(rec) + "\n"); fo.flush()
    print("new spend this run:", {k: round(v, 4) if isinstance(v, float) else v for k, v in spent.items()},
          f"ledger total ${get_ledger().total_usd:.4f} cap ${get_ledger().summary()['budget_usd']:.2f}", flush=True)


if __name__ == "__main__":
    main()
