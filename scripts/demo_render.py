"""Render outputs/demo_before_after.mp4 + stills from outputs/demo_ba/{before,now}.json (see demo_before_after.py).
Muse verdicts are the cached eval rows (no API calls); fusion + speech use the current safety code."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN)); sys.path.insert(0, str(MAIN / "scripts"))
from streetsense.meta_api import fuse_crossing_decision, SAFE_CROSSING_FALLBACK
from demo_before_after import SEGMENTS, WARM, OUT

W, H, FPS = 1280, 720, 30
BG = (14, 16, 21); FG = (236, 238, 242); MUTED = (140, 146, 158)
COL = {"walk": (52, 199, 89), "green": (48, 176, 199), "red": (255, 69, 58), "unclear": (255, 179, 64),
       "countdown": (255, 179, 64), "unknown": (110, 116, 128), "go": (52, 199, 89), "stop": (255, 69, 58), "unsure": (255, 179, 64)}
FD = "/usr/share/fonts/truetype/sand-box/google/Inter/"
def font(sz, bold=False):
    import glob
    fs = sorted(x for x in glob.glob(FD + "*.ttf") if "Italic" not in x) or ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    f = ImageFont.truetype(fs[0], sz)
    try: f.set_variation_by_name("Bold" if bold else "Regular")
    except Exception: pass
    return f
F_T, F_H, F_B, F_S = font(54, True), font(30, True), font(22), font(17)
SPEECH = {"go": "Walk signal looks clear. Still listen and check.", "stop": "Do not cross.", "unsure": SAFE_CROSSING_FALLBACK}

def muse_rows():
    rows = {}
    for f, vs in (("muse_eval.json", ("A", "E")), ("muse_eval_broad.json", ("E@768",))):
        d = json.loads((MAIN / "docs/benchmark" / f).read_text())
        for v in vs:
            for r in d["variants"][v]["rows"]:
                rows[(v.split("@")[0], r["id"])] = r
    return rows
MUSE = muse_rows()
DUMP = {t: json.loads((OUT / f"{t}.json").read_text()) for t in ("before", "now")}
CAPS = {}

def frame_at(seg, t):
    clip = SEGMENTS[seg][0]
    cap = CAPS.setdefault(str(clip), cv2.VideoCapture(str(clip)))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read()
    s = 1280 / f.shape[1]
    return cv2.resize(f, (1280, int(round(f.shape[0] * s))), interpolation=cv2.INTER_AREA)

def pill(d, x, y, text, col, f=F_B):
    w = d.textlength(text, font=f) + 28
    d.rounded_rectangle([x, y, x + w, y + 38], radius=19, fill=col); d.text((x + 14, y + 6), text, font=f, fill=(10, 10, 12))
    return x + w

def panel(seg, rec, tag, pw=616, ph=347, crop=None):
    f = frame_at(seg, rec["t"]); img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(img)
    for det in rec["dets"]:
        st = det["state"] or "unknown"; c = COL.get(st, COL["unknown"]); x1, y1, x2, y2 = det["box"]
        dr.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=c, width=4)
        dr.text((x1, max(0, y1 - 34)), st, font=font(30, True), fill=c, stroke_width=2, stroke_fill=(0, 0, 0))
    if crop: img = img.crop(crop)
    img.thumbnail((pw, ph)); out = Image.new("RGB", (pw, ph), (0, 0, 0)); out.paste(img, ((pw - img.width) // 2, (ph - img.height) // 2))
    return out

def compose(seg, title, sub, i_b, i_n, muse=None, crop=None):
    can = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(can)
    d.text((32, 22), title, font=F_H, fill=FG); d.text((32, 62), sub, font=F_S, fill=MUTED)
    for k, (tag, lab, idx) in enumerate((("before", "BEFORE  ·  commit fee7958", i_b), ("now", "NOW  ·  main", i_n))):
        x = 16 + k * 632; rec = DUMP[tag][seg][idx]
        d.text((x, 100), lab, font=F_S, fill=MUTED if tag == "before" else FG)
        can.paste(panel(seg, rec, tag, crop=crop), (x, 126))
        local = rec["local"]; y = 488
        nx = pill(d, x, y, f"local: {local}", COL.get(local, COL["unknown"]))
        if muse:
            m = muse.get(tag)
            if m is None:
                d.text((nx + 12, y + 8), "Muse: not asked", font=F_S, fill=MUTED); dec = fuse_crossing_decision(local, None)["decision"] if local not in ("walk", "green") else "unsure"
            else:
                pill(d, nx + 10, y, f"Muse: {m['muse_safe']}", COL["go"] if m["muse_safe"] == "yes" else COL["stop"])
                dec = fuse_crossing_decision(local, {"safe_to_cross": m["muse_safe"], "pedestrian_signal": m["muse_signal"], "countdown_seconds": m["countdown"]})["decision"]
            pill(d, x, y + 52, {"go": "GO", "stop": "STOP", "unsure": "WAIT"}[dec], COL[dec], F_H if False else F_B)
            spoken = SPEECH[dec]
        else:
            spoken = rec["last_spoken"] or "(silent)"
        d.text((x, 600), "🔊" if False else "says:", font=F_S, fill=MUTED)
        d.text((x + 52, 596), f"“{spoken}”", font=F_B, fill=FG)
    return can

def idx_of(seg, t): return min(range(len(DUMP["now"][seg])), key=lambda i: abs(DUMP["now"][seg][i]["t"] - t))

def card(lines, title=None):
    can = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(can); y = 150 if title else 260
    if title: d.text((80, 80), title, font=F_T, fill=FG)
    for big, small in lines:
        d.text((80, y), big, font=F_H, fill=FG); d.text((80, y + 40), small, font=F_S, fill=MUTED); y += 100
    return can

def main():
    seq, stills = [], {}
    def add(img, sec): seq.append((img, int(sec * FPS)))
    add(card([("Before vs now", "Same clips, old code (fee7958) vs current main. Muse replies replayed from the eval cache, no new calls.")], "StreetSense"), 3.5)
    plan = [
        ("dc_walk", "Washington DC · walk phase", "Old Muse prompt said no on a clear walk. New prompt says yes; local reader agrees → go.", 13.8, ("A", "E"), "dc_13.80", 1.0, None),
        ("dc_countdown", "US countdown ending · 48.5–53 s", "Old reader kept saying walk while the countdown ran out. Now: unclear → wait.", None, None, None, 1.0, None),
        ("eu_heldout", "Held-out European signal · dotted green man", "Not used in tuning. Old reader: 'green' (not a walk). Now: tiles read the walk figure.", None, None, None, 1.0, None),
        ("night", "Night crossing", "Old reader saw nothing. Now: local walk AND cached Muse yes → go.", 1.0, (None, "E"), "night_ped_01.00", 0.5, None),
    ]
    for seg, title, sub, ft, var, mid, speed, crop in plan:
        clip, t0, t1 = SEGMENTS[seg]; start = t0 + WARM if seg != "night" else t0
        recs = [i for i, r in enumerate(DUMP["now"][seg]) if r["t"] >= start - 1e-6]
        hold = (1 / 10) / speed
        for i in recs: add(compose(seg, title, sub, i, i, crop=crop), hold)
        stills[seg] = compose(seg, title, sub, recs[len(recs) * 2 // 3], recs[len(recs) * 2 // 3], crop=crop)
        if ft is not None:
            i = idx_of(seg, ft); muse = {"before": MUSE.get((var[0], mid)) if var[0] else None, "now": MUSE.get((var[1], mid))}
            fr = compose(seg, title, sub, i, i, muse=muse, crop=crop); add(fr, 4.0); stills[seg] = fr
    add(card([("GO only if local reader says WALK  AND  Muse says yes", "Any doubt, any disagreement, any stale reply → “I can't tell, please wait.”"),
              ("Muse alone saying yes is never enough", "Local reader alone saying walk is never enough")], "Safety rule"), 4.5)
    end = card([("Muse yes on walk frames: 0/10 → 9/10", "0 false yes on 22 non-walk frames · docs/MUSE_EVAL.md"),
                ("US countdown-ending false walks: 16/19 → 0/19", "docs/READER.md"),
                ("Held-out walk signals read: 0/14 → 5/14", "with tiles always on · docs/READER.md, docs/DETECTORS.md"),
                ("SAM fallback: $9/h naive → ≤ $0.90/h capped, ~0.5¢ per crossing", "docs/SAM_FALLBACK.md")], "Measured results")
    add(end, 6.5); stills["results"] = end
    out = MAIN / "outputs/demo_before_after.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    prev = None; XF = 12  # 0.4 s crossfade into each new scene
    for k, (img, n) in enumerate(seq):
        a = np.asarray(img)
        newscene = k == 0 or n >= FPS * 3 or (seq[k - 1][1] >= FPS * 3)
        for j in range(n):
            if prev is not None and newscene and j < XF:
                w = (j + 1) / (XF + 1); p.stdin.write((prev * (1 - w) + a * w).astype(np.uint8).tobytes())
            else: p.stdin.write(a.tobytes())
        prev = a.astype(np.float32)
    p.stdin.close(); p.wait()
    for k, (name, im) in enumerate(stills.items()): im.save(MAIN / f"outputs/demo_still_{k+1}_{name}.png")
    print(out, sum(n for _, n in seq) / FPS, "s", list(stills))

if __name__ == "__main__": main()
