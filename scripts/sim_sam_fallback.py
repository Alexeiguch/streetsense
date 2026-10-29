"""Simulated walking session for the cost-capped SAM signal fallback (docs/SAM_FALLBACK.md).

Replays real clips through the real Pipeline (YOLO + reader + smoother) in stream time. Each crossing clip is
one crossing: maps reports the crossing within 30 m for the whole clip and the user asks "can I cross?" 3 s in.
Between clips the walker spends `--gap` seconds walking with no crossing context (one blank frame there lets
the context grace expire). A non-crossing clip (scooter) has no maps context: only a detected crosswalk can
trigger SAM there.

--mode mock (default, $0): SAM returns no boxes after --lat s (conservative: never helps the reader settle,
so call counts are an upper bound). --mode real: real SAM 3.1 calls (ledger-logged, capped), run inline and
delivered at their measured latency in stream time.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from streetsense.config import Config
from streetsense.pipeline import Pipeline
from streetsense.sam_fallback import SamControlConfig, SamFallbackController

CROSSINGS = ["samples/dc_ped_light.ogv", "samples/cdmx_crossing_720p.webm", "samples/ext/pexels_30404309_eu_ped.mp4",
             "samples/ext/pexels_5514350_us_ped.mp4", "samples/ext/pexels_17721925_night_ped.mp4",
             "samples/ext/most_04_dusk.webm", "samples/ext/tirana_dusk.webm", "samples/heldout/pexels_6415590.mp4",
             "samples/ext/pexels_7251025_night_square.mp4", "samples/heldout/nz.ogv"]
NON_CROSSING = ["samples/katowice_scooter.webm"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="mock"); ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--gap", type=float, default=60.0); ap.add_argument("--lat", type=float, default=0.8)
    ap.add_argument("--clips", type=int, default=len(CROSSINGS)); ap.add_argument("--out", default="outputs/sam_fallback_sim.json")
    a = ap.parse_args()
    cfg = Config(); cfg.vlm.enabled = False
    pipe = Pipeline(cfg, use_surface=False)
    scfg = SamControlConfig.from_env(); scfg.enabled = True
    if a.mode == "real":
        from streetsense.meta_api import MetaSamSegmenter
        seg = MetaSamSegmenter(max_image_side=768); lat = []
        def fn(img, prompt):
            r = seg.segment(img, prompt, decode_masks=False); lat.append(r.latency_s); ctl.sim_latency_s = r.latency_s; return r
        ctl = SamFallbackController(scfg, fn, sync=True, sim_latency_s=a.lat)
    else:
        lat = []
        ctl = SamFallbackController(scfg, lambda img, p: SimpleNamespace(boxes=[]), budget_check=lambda x: None,
                                    sync=True, sim_latency_s=a.lat)
    pipe.sam_fallback = ctl
    T = 0.0; plan = [(c, True) for c in CROSSINGS[: a.clips]] + [(c, False) for c in NON_CROSSING]
    per_clip = []; t_wall = time.perf_counter(); frames = 0
    for path, crossing in plan:
        cap = cv2.VideoCapture(str(ROOT / path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30; step = max(1, round(fps / a.fps))
        c0 = ctl.stats["calls"]; t0 = T; i = 0; asked = False
        pipe.note_crossing_distance(30.0 if crossing else None)
        while True:
            ok, f = cap.read()
            if not ok: break
            if i % step == 0:
                if f.shape[1] > 1280:
                    s = 1280 / f.shape[1]; f = cv2.resize(f, (1280, int(f.shape[0] * s)))
                t = T + i / fps
                if crossing and not asked and t - t0 >= 3.0:
                    pipe.note_crossing_check(t); asked = True
                pipe.process(f, annotate=False, t=t); frames += 1
            i += 1
        dur = i / fps; T += dur
        per_clip.append({"clip": path, "crossing_ctx": crossing, "dur_s": round(dur, 1), "calls": ctl.stats["calls"] - c0})
        print(per_clip[-1], flush=True)
        pipe.note_crossing_distance(None); T += a.gap
        for k in range(12):  # walking between crossings (lets crosswalk hysteresis and context grace clear)
            pipe.process(np.zeros((360, 640, 3), np.uint8), annotate=False, t=T - 24 + 2 * k)
    s = pipe.sam_fallback_summary()
    res = {"mode": a.mode, "fps": a.fps, "gap_s": a.gap, "session_s": round(T, 1), "frames": frames,
           "wall_s": round(time.perf_counter() - t_wall, 1), "real_latency_s": lat, "per_clip": per_clip, "summary": s,
           "limits": scfg.__dict__}
    Path(ROOT / a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in s.items() if k != "crossings"}, default=str))
    for c in s["crossings"]: print(c)


if __name__ == "__main__":
    main()
