#!/usr/bin/env python
"""Render an annotated demo of signal/sign distance estimates and relevant-signal selection.

Local only: no Meta / Muse calls (cfg.vlm.enabled=False). Shows, per frame:
- every traffic-light / sign box with kind, local state and estimated distance (lo-hi range),
- which head the selector uses for the crossing (thick box) and why,
- the line the app would speak for that signal, as a caption.

Distances are pinhole estimates from assumed housing sizes; they are not validated.

  .venv/bin/python scripts/demo_signal_distance.py \
      --seg samples/dc_ped_light.ogv:7:15:70 --seg samples/ext/most_04_dusk.webm:3:13:10 \
      --out outputs/signal_distance_demo.mp4 --stills outputs/signal_distance_still
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_W = 960
FPS = 10


def read_frames(path: str, start: float, end: float, fps: int):
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=width,height", "-of", "csv=p=0", path], text=True)
    w, h = (int(v) for v in probe.strip().split(",")[:2])
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(end - start), "-i", path,
           "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    i = 0
    while True:
        buf = p.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        yield start + i / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        i += 1
    p.wait()


def fmt_dist(a: dict) -> str:
    m = a.get("signal_distance_m")
    if m is None:
        return "dist ?"
    lo, hi = a.get("signal_distance_lo_m"), a.get("signal_distance_hi_m")
    return f"~{m:.0f} m ({lo:.0f}-{hi:.0f}) {a.get('signal_band', '')}"


def put(img, text, org, scale, color, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


KIND = {"pedestrian": "ped head", "vehicle": "vehicle head", "unknown": "signal head"}
COLORS = {"walk": (80, 255, 80), "green": (80, 200, 80), "red": (60, 60, 255), "yellow": (0, 200, 255),
          "countdown": (0, 160, 255)}
VIEW_H = 720  # every segment is letterboxed into OUT_W x VIEW_H so the video has one size


def label_block(img, lines, colors, box, taken):
    """Draw a small dark label next to the box, avoiding earlier labels."""
    fs, lh = 0.45, 16
    tw = max(cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] for ln in lines) + 8
    th = lh * len(lines) + 4
    x1, y1, x2, y2 = box
    Wi, Hi = img.shape[1], img.shape[0]
    cands = [(x2 + 4, y1), (x1 - tw - 4, y1), (x1, y1 - th - 2), (x1, y2 + 2)]
    for _ in range(12):
        for cx, cy in cands:
            cx, cy = int(min(max(0, cx), Wi - tw)), int(min(max(0, cy), Hi - th))
            r = (cx, cy, cx + tw, cy + th)
            if all(r[2] <= t[0] or r[0] >= t[2] or r[3] <= t[1] or r[1] >= t[3] for t in taken):
                taken.append(r)
                sub = img[r[1]:r[3], r[0]:r[2]]
                sub[:] = (sub * 0.35).astype(np.uint8)
                for i, (ln, c) in enumerate(zip(lines, colors)):
                    cv2.putText(img, ln, (r[0] + 4, r[1] + 14 + i * lh), cv2.FONT_HERSHEY_SIMPLEX, fs, c, 1,
                                cv2.LINE_AA)
                return
        cands = [(cx, cy + th) for cx, cy in cands]


def render(frame, out, sel, spoken, title, hfov, use_depth):
    H, W = frame.shape[:2]
    s = min(OUT_W / W, VIEW_H / H)
    img = cv2.resize(frame, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    taken: list = []
    items = [d for d in out.detections if d.label == "traffic light" or "sign" in d.label]
    items.sort(key=lambda d: not (sel is not None and sel.detection is d))  # selected label first
    for d in items:
        b = d.box
        p1, p2 = (int(b.x1 * s), int(b.y1 * s)), (int(b.x2 * s), int(b.y2 * s))
        is_light = d.label == "traffic light"
        st = d.attrs.get("light_state", "unknown") if is_light else "sign"
        chosen = sel is not None and sel.detection is d
        col = COLORS.get(st, (200, 200, 200))
        cv2.rectangle(img, p1, p2, (255, 255, 255) if chosen else col, 4 if chosen else 1)
        name = KIND.get(d.attrs.get("signal_kind", "unknown"), "signal head") if is_light else d.label
        lines, cols = [f"{name}: {st}", fmt_dist(d.attrs)], [col, (230, 230, 230)]
        if use_depth and d.attrs.get("signal_depth_rel") is not None:
            lines.append(f"rel. depth {d.attrs['signal_depth_rel']:.2f} (higher=nearer)")
            cols.append((200, 200, 120))
        if chosen:
            lines.insert(0, "SELECTED for crossing")
            cols.insert(0, (255, 255, 255))
        label_block(img, lines, cols, (p1[0], p1[1], p2[0], p2[1]), taken)
    view = np.zeros((VIEW_H, OUT_W, 3), np.uint8)
    oy, ox = (VIEW_H - img.shape[0]) // 2, (OUT_W - img.shape[1]) // 2
    view[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    bar = np.zeros((96, OUT_W, 3), np.uint8)
    decision = sel.decision if sel is not None else "no signal"
    why = (sel.reason if sel is not None else "") or ""
    if sel is not None and sel.caution:
        why = f"distance check: {why}"
    put(bar, f"spoken: \"{spoken}\"" if spoken else "spoken: (nothing about signals)", (10, 26), 0.6,
        (255, 255, 255), 1)
    put(bar, f"local signal decision: {decision}   [{why[:70]}]", (10, 52), 0.5,
        (120, 200, 255) if sel is not None and sel.caution else (200, 230, 255))
    put(bar, "distance only adds caution; go needs local walk AND Muse yes (Muse not called here)", (10, 76),
        0.45, (160, 160, 160))
    top = np.zeros((26, OUT_W, 3), np.uint8)
    dep = " + depth" if use_depth else ""
    put(top, f"{title}   HFOV {hfov:.0f} deg{dep}   distances are estimates, not validated", (8, 18), 0.5,
        (255, 255, 0))
    return np.vstack([top, view, bar])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", action="append", required=True, help="path:start:end:hfov_deg[:title[:depth]]")
    ap.add_argument("--out", default="outputs/signal_distance_demo.mp4")
    ap.add_argument("--stills", default="outputs/signal_distance_still")
    ap.add_argument("--still-at", action="append", default=[], help="seg_index:time_s for PNG stills")
    ap.add_argument("--depth", action="store_true", help="run Depth Anything V2 on every segment")
    args = ap.parse_args()

    writer = None
    stills = {(int(a.split(":")[0]), float(a.split(":")[1])) for a in args.still_at}
    stats = []
    for si, seg in enumerate(args.seg):
        parts = seg.split(":")
        path, start, end, hfov = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        title = parts[4] if len(parts) > 4 else Path(path).stem
        os.environ["STREETSENSE_CAMERA_HFOV_DEG"] = str(hfov)  # one source of truth for every caller
        from streetsense.config import Config
        from streetsense.pipeline import Pipeline
        from streetsense.rules import build_alerts, FrameFacts, select_crossing_signal

        cfg = Config()
        cfg.vlm.enabled = False
        cfg.depth.enabled = bool(args.depth) or (len(parts) > 5 and parts[5] == "depth")
        cfg.depth.every_n_frames = 1
        pipe = Pipeline(cfg, use_surface=False)
        for t, frame in read_frames(path, start, end, FPS):
            out = pipe.process(frame, annotate=False, t=t)
            H, W = frame.shape[:2]
            sel = select_crossing_signal(out.detections, W, H, out.crosswalk)
            facts = FrameFacts(W, H, out.detections, out.crosswalk)
            light_alerts = [a for a in build_alerts(facts, cfg.camera.hfov_deg) if a.key.startswith("light:")]
            spoken = light_alerts[0].text if light_alerts else ""
            img = render(frame, out, sel, spoken, title, hfov, cfg.depth.enabled)
            if writer is None:
                hh, ww = img.shape[:2]
                writer = subprocess.Popen(
                    ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ww}x{hh}",
                     "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "26",
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
            writer.stdin.write(img.tobytes())
            stats.append((si, round(t, 1), sel.decision if sel else "none", sel.caution if sel else None, spoken))
            for (ssi, st) in list(stills):
                if ssi == si and abs(st - t) < 0.5 / FPS:
                    cv2.imwrite(f"{args.stills}_{si}_{st:05.1f}.png", img)
    writer.stdin.close()
    writer.wait()
    from collections import Counter
    print("decisions per segment:", {si: dict(Counter(d for s, _, d, _, _ in stats if s == si)) for si in range(len(args.seg))})
    print("cautions:", Counter(c for *_, c, _ in stats if c))
    for row in stats[:: FPS]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
