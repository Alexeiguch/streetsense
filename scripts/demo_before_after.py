"""Before/after demo video (outputs/demo_before_after.mp4). No API calls: Muse replies come from eval/muse_cache
via docs/benchmark/muse_eval*.json rows (variant A = old strict prompt, E/E@768 = current default).

  dump:   run a Pipeline (old code in a git worktree at fee7958, or current main) over each segment, write
          per-frame detections + spoken alerts to outputs/demo_ba/<tag>.json
          .venv/bin/python scripts/demo_before_after.py dump --root /workspace/ss_before --tag before
          STREETSENSE_LIGHT_TILES=2x3 STREETSENSE_LIGHT_TILES_MODE=always STREETSENSE_SAM_FALLBACK=0 \
            .venv/bin/python scripts/demo_before_after.py dump --root . --tag now
  render: compose the 1280x720 video + stills.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
S = MAIN / "samples"
SEGMENTS = {  # name: (clip, t0, t1)  -- t0 includes 1.5 s smoother warm-up that is not shown
    "dc_walk": (S / "dc_ped_light.ogv", 8.3, 13.8),
    "dc_countdown": (S / "dc_ped_light.ogv", 47.0, 53.0),
    "eu_heldout": (S / "heldout/pexels_6415590.mp4", 6.0, 10.0),
    "night": (S / "ext/pexels_17721925_night_ped.mp4", 0.0, 1.7),
}
WARM = 1.5
FPS = 10.0
OUT = MAIN / "outputs/demo_ba"


def dump(root: Path, tag: str):
    import cv2
    os.chdir(root); sys.path.insert(0, str(root))
    from streetsense.config import Config
    from streetsense.pipeline import Pipeline
    res = {}
    for name, (clip, t0, t1) in SEGMENTS.items():
        cfg = Config(); cfg.vlm.enabled = False
        if hasattr(cfg, "sam"): cfg.sam.enabled = False
        clock = type("C", (), {"t": 0.0, "__call__": lambda self: self.t})()
        pipe = Pipeline(cfg, use_surface=False, clock=clock)
        cap = cv2.VideoCapture(str(clip)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = []; t = t0; last_spoken = None
        while t <= t1 + 1e-6:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read()
            if not ok: break
            if f.shape[1] != 1280:
                s = 1280 / f.shape[1]; f = cv2.resize(f, (1280, int(round(f.shape[0] * s))), interpolation=cv2.INTER_AREA)
            clock.t = t
            out = pipe.process(f, annotate=False, t=t)
            if out.spoken: last_spoken = out.spoken.text
            dets = [{"box": [d.box.x1, d.box.y1, d.box.x2, d.box.y2], "label": d.label, "conf": round(float(d.conf), 3),
                     "state": d.attrs.get("light_state"), "kind": d.attrs.get("signal_kind"),
                     "src": d.attrs.get("source") or ("tile" if "tile" in d.attrs else None)}
                    for d in out.detections if d.label in ("traffic light",)]
            try:
                from streetsense.rules import select_crossing_signal
                sel = select_crossing_signal(out.detections, f.shape[1], f.shape[0], out.crosswalk)
                local = sel.decision if sel is not None else "unknown"
            except Exception as e:
                local = f"n/a ({type(e).__name__})"
            record = {"t": round(t, 2), "dets": dets, "spoken": out.spoken.text if out.spoken else None,
                      "last_spoken": last_spoken, "local": local, "shape": list(f.shape[:2])}
            try:
                from streetsense.signal_trace import summarise_signals
                record["signal_trace"] = summarise_signals(out, f.shape[1], f.shape[0])
            except ImportError:  # old worktree has no trace module
                pass
            frames.append(record)
            t = round(t + 1 / FPS, 4)
        res[name] = frames
        print(tag, name, len(frames), "local states:", sorted({x["local"] for x in frames if x["t"] >= t0 + WARM}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--root", default=str(MAIN)); ap.add_argument("--tag", default="now")
    a = ap.parse_args()
    if a.cmd == "dump": dump(Path(a.root).resolve(), a.tag)
    else:
        sys.path.insert(0, str(MAIN / "scripts")); import demo_render; demo_render.main()
