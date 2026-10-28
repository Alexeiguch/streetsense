"""Simulated walk: a GPS track along a real route, replayed through the Navigator.

The track is synthetic but honest about GPS: each fix gets Gaussian noise
(default sigma 4 m, a good open-sky phone) and an accuracy figure, the walker
pauses at every crossing, and an optional detour leaves the route to show
off-route detection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .geo import destination, point_at
from .model import Route
from .navigator import Cue, Fix, Navigator


@dataclass
class TrackConfig:
    speed_mps: float = 1.3
    hz: float = 1.0
    noise_m: float = 4.0
    accuracy_m: float = 8.0
    pause_at_crossing_s: float = 12.0
    detour_at_m: float | None = None  # leave the route here
    detour_m: float = 35.0  # how far off
    detour_side: float = 90.0  # degrees relative to the route
    seed: int = 7


def synth_track(route: Route, cfg: TrackConfig | None = None) -> list[Fix]:
    cfg = cfg or TrackConfig()
    rng = random.Random(cfg.seed)
    dt = 1.0 / cfg.hz
    fixes: list[Fix] = []
    t, along = 0.0, 0.0
    paused: set[int] = set()
    detoured = False

    def noisy(p, acc):
        n = rng.gauss(0.0, cfg.noise_m)
        return destination(p, rng.uniform(0, 360), abs(n)), acc

    while along < route.length_m:
        p, brg = point_at(route.points, route.cum, along)
        (q, acc) = noisy(p, cfg.accuracy_m)
        fixes.append(Fix(t=round(t, 2), lat=q[0], lon=q[1], heading=None, accuracy_m=acc))
        # pause at the kerb of each crossing, as a real walker waits for the light
        for i, c in enumerate(route.crossings):
            if i not in paused and (c.along_m or 0) - 6 <= along < (c.along_m or 0) - 6 + cfg.speed_mps * dt:
                paused.add(i)
                for _ in range(int(cfg.pause_at_crossing_s * cfg.hz)):
                    t += dt
                    (q, acc) = noisy(p, cfg.accuracy_m)
                    fixes.append(Fix(t=round(t, 2), lat=q[0], lon=q[1], accuracy_m=acc))
        if cfg.detour_at_m is not None and not detoured and along >= cfg.detour_at_m:
            detoured = True
            n = max(1, int(cfg.detour_m / cfg.speed_mps))
            out_brg = (brg + cfg.detour_side) % 360
            for k in list(range(1, n + 1)) + list(range(n, 0, -1)):
                t += dt
                (q, acc) = noisy(destination(p, out_brg, k * cfg.speed_mps), cfg.accuracy_m)
                fixes.append(Fix(t=round(t, 2), lat=q[0], lon=q[1], accuracy_m=acc))
        t += dt
        along += cfg.speed_mps * dt
    end = route.points[-1]
    fixes.append(Fix(t=round(t, 2), lat=end[0], lon=end[1], accuracy_m=cfg.accuracy_m))
    return fixes


def run_walk(nav: Navigator, fixes: list[Fix],
             on_crossing: Callable[[Navigator, Fix], list[Cue]] | None = None) -> list[Cue]:
    """Replay fixes; when the walker reaches a crossing kerb, call on_crossing (camera + Muse check)."""
    cues = [nav.start(fixes[0].t if fixes else 0.0)]
    for f in fixes:
        new = nav.update(f)
        cues.extend(new)
        if on_crossing and any(c.kind == "at_crossing" for c in new):
            cues.extend(on_crossing(nav, f))
    return cues


def to_srt(cues: list[Cue], max_len_s: float = 4.0) -> str:
    """Subtitles for the demo video, one block per spoken cue."""
    def ts(x: float) -> str:
        ms = int(round(x * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    spoken = [c for c in cues if c.text]
    blocks = []
    for i, c in enumerate(spoken):
        nxt = spoken[i + 1].t if i + 1 < len(spoken) else c.t + max_len_s
        end = min(c.t + max(1.5, min(max_len_s, len(c.text.split()) / 2.6)), max(c.t + 1.0, nxt - 0.05))
        blocks.append(f"{i + 1}\n{ts(c.t)} --> {ts(end)}\n{c.text}\n")
    return "\n".join(blocks)


def track_stats(fixes: list[Fix]) -> dict:
    return {"fixes": len(fixes), "duration_s": round(fixes[-1].t - fixes[0].t, 1) if fixes else 0,
            "noise_sigma_m": None, "mean_accuracy_m": round(sum(f.accuracy_m for f in fixes) / max(1, len(fixes)), 1)}


