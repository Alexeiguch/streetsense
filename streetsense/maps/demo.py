"""walk-sim: replay a GPS walk along a real route and speak the guidance.

At each crossing kerb the walker "looks" at a sample clip: the local pipeline
reads it, the navigator fuses the map's crossing direction with the camera, and
(with --ask-muse) the shared crossing_answer path runs the go rule. Map data
only ever adds context.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .model import choose_safer_route
from .navigator import Cue, Navigator
from .walk_sim import TrackConfig, run_walk, synth_track, to_srt

DC_FROM = (38.9031, -77.0387)  # Farragut Square, north side, Washington DC
DC_TO = (38.8992, -77.0388)  # Renwick Gallery, 17th St and Pennsylvania Ave NW


def _latlon(s: str) -> tuple[float, float]:
    a, b = (float(x) for x in s.split(","))
    return (a, b)


def run(a: Any) -> int:
    from ..config import MapsConfig
    from . import build_session

    sess = build_session(MapsConfig(enabled=True, provider=a.provider))
    prov = sess.provider
    origin, dest = _latlon(a.origin), _latlon(a.dest)
    t0 = time.perf_counter()
    routes = prov.routes(origin, dest)
    route, why = choose_safer_route(routes)
    plan_ms = round((time.perf_counter() - t0) * 1000)
    nav = Navigator(route, a.dest_name, reverse=prov.reverse)
    fixes = synth_track(route, TrackConfig(speed_mps=a.speed, noise_m=a.noise, accuracy_m=a.accuracy,
                                           detour_at_m=a.detour_at if a.detour_at >= 0 else None, seed=a.seed))

    clip_at = [float(x) for x in a.clip_at.split(",")] if a.clip_at else []
    pipe = None
    state = {"crossing_i": 0, "muse_calls": 0}
    checks: list[dict] = []

    def on_crossing(nav: Navigator, fix) -> list[Cue]:
        nonlocal pipe
        i = state["crossing_i"]
        state["crossing_i"] += 1
        if not a.clip or not clip_at:
            return []
        start = clip_at[i % len(clip_at)]
        from ..answers import crossing_answer, local_signal_state
        from ..cli import VideoClock, _local_config, _sample_frames
        from ..observations import summarise_frame
        from ..pipeline import Pipeline

        if pipe is None:
            cfg = _local_config(a)
            cfg.vlm.enabled = bool(a.ask_muse)
            pipe = Pipeline(cfg, use_surface=False, clock=VideoClock())
        from ..traffic_light import LightStateSmoother

        pipe.smoother = LightStateSmoother(clock=pipe.clock)  # votes never carry over from the last crossing
        frame = out = None
        for t, f in _sample_frames(a.clip, start, a.clip_seconds, 4):
            pipe.clock.t = t
            out = pipe.process(f, annotate=False)
            frame = f
        if frame is None:
            return []
        s = summarise_frame(out, frame.shape[1], frame.shape[0])
        local = local_signal_state(out, frame.shape)
        cues: list[Cue] = []
        hint = nav.camera_hint(s.get("crosswalk"), local, fix.t + 1)
        if hint:
            cues.append(hint)
        rec: dict = {"t": fix.t, "crossing": nav.at_crossing.road if nav.at_crossing else None,
                     "clip": a.clip, "clip_start_s": start, "local_light": local, "camera_crosswalk": s.get("crosswalk")}
        if a.ask_muse and state["muse_calls"] < a.muse_max:
            state["muse_calls"] += 1
            ans = crossing_answer(pipe, frame, out=out, clock=lambda: pipe.clock.t)
            rec.update({k: ans[k] for k in ("local_light", "fuse", "spoken", "validated")})
            line = ans["spoken"]
        else:
            from ..meta_api import SAFE_CROSSING_FALLBACK

            line = SAFE_CROSSING_FALLBACK  # without the Muse check the go rule cannot pass
            rec.update({"spoken": line, "note": "Muse check off; go needs local walk AND Muse yes"})
        checks.append(rec)
        cues.append(Cue(fix.t + 3, "crossing_check", line, line[:30], data=rec))
        return cues

    cues = run_walk(nav, fixes, on_crossing)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / a.name
    summary = {
        "provider": a.provider, "origin": origin, "dest": dest, "dest_name": a.dest_name,
        "routes": [r.summary() for r in routes], "chosen": route.summary(), "why": why, "plan_ms": plan_ms,
        "fixes": len(fixes), "walk_s": fixes[-1].t, "cues": len(cues), "crossing_checks": checks,
        "attribution": route.attribution, "http": getattr(prov.http, "calls", [])[-12:], "warnings": prov.warnings,
    }
    with open(f"{stem}.jsonl", "w") as f:
        f.write(json.dumps({"summary": summary}, default=str) + "\n")
        for c in cues:
            f.write(json.dumps(c.to_dict(), default=str) + "\n")
    Path(f"{stem}.srt").write_text(to_srt(cues))
    md = [f"# Simulated walk: {a.dest_name}", "",
          f"Route chosen: {route.label}, {round(route.length_m)} m, {len(route.crossings)} crossings. Why: {why}.",
          f"GPS: synthetic, sigma {a.noise} m, accuracy {a.accuracy} m, {len(fixes)} fixes. {route.attribution}", "",
          "| t (s) | kind | spoken | glasses |", "|---|---|---|---|"]
    md += [f"| {c.t:.0f} | {c.kind} | {c.text} | {c.short} |" for c in cues]
    Path(f"{stem}.md").write_text("\n".join(md) + "\n")
    if a.plot:
        _plot(routes, route, fixes, cues, f"{stem}.png")
    for c in cues:
        print(f"{c.t:6.0f}s  {c.kind:16s} {c.text}")
    print(json.dumps({k: summary[k] for k in ("chosen", "why", "plan_ms", "walk_s", "cues")}, default=str))
    print(f"wrote {stem}.jsonl .srt .md" + (" .png" if a.plot else ""))
    return 0


def _plot(routes, chosen, fixes, cues, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 7), dpi=120)
    for r in routes:
        if r is not chosen:
            ax.plot([p[1] for p in r.points], [p[0] for p in r.points], color="#bbb", lw=2, label=f"{r.label} (not chosen)")
    ax.plot([p[1] for p in chosen.points], [p[0] for p in chosen.points], color="#1f6feb", lw=4, label=f"{chosen.label} (chosen)")
    ax.scatter([f.lon for f in fixes], [f.lat for f in fixes], s=4, color="#f08c00", label="GPS fixes (simulated)")
    for c in chosen.crossings:
        ax.scatter([c.lon], [c.lat], s=90, marker="s", color="#2da44e" if c.signals else "#cf222e", zorder=5)
        ax.annotate((c.road or "").replace(" Northwest", " NW"), (c.lon, c.lat), fontsize=7, xytext=(5, 3), textcoords="offset points")
    ax.set_aspect(1 / 0.778)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("StreetSense walk: green squares are signalised crossings", fontsize=9)
    fig.text(0.01, 0.01, chosen.attribution, fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def add_parser(sub) -> None:
    w = sub.add_parser("walk-sim", help="simulated GPS walk with spoken map guidance and camera checks at crossings")
    w.add_argument("--provider", default="fixture", choices=["fixture", "osm", "record"],
                   help="fixture: recorded OSM data, offline (default); osm: live public servers")
    w.add_argument("--origin", default=f"{DC_FROM[0]},{DC_FROM[1]}")
    w.add_argument("--dest", default=f"{DC_TO[0]},{DC_TO[1]}")
    w.add_argument("--dest-name", default="the Renwick Gallery")
    w.add_argument("--speed", type=float, default=1.3)
    w.add_argument("--noise", type=float, default=4.0, help="GPS noise sigma in metres")
    w.add_argument("--accuracy", type=float, default=8.0, help="reported GPS accuracy in metres")
    w.add_argument("--detour-at", type=float, default=280.0, help="metres along the route to wander off; -1 for none")
    w.add_argument("--seed", type=int, default=7)
    w.add_argument("--clip", default="samples/dc_ped_light.ogv", help="clip the walker looks at at each crossing; '' for none")
    w.add_argument("--clip-at", default="55,44", help="clip start seconds per crossing, cycled (DC clip: red 54-60, walk 41-53)")
    w.add_argument("--clip-seconds", type=float, default=2.5)
    w.add_argument("--ask-muse", action="store_true", help="run the Muse Spark crossing check (costs about $0.003 each)")
    w.add_argument("--muse-max", type=int, default=2)
    w.add_argument("--backend", default="openvino")
    w.add_argument("--plot", action="store_true")
    w.add_argument("--out-dir", default="outputs")
    w.add_argument("--name", default="walk_sim")
    w.set_defaults(fn=run)
