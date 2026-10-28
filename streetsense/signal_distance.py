"""Distance to traffic-light / pedestrian-signal heads and street signs, and the caution rules
that use it when choosing which signal controls the crossing.

Distance can only ever add caution. It never turns a non-walk state into walk and it never
authorises crossing; the go rule stays "local reader says walk AND Muse says yes". What it does:
- a walk head estimated beyond `max_walk_m` (the next junction down the street) is not used;
- a walk head far off to the side (`max_bearing_deg`) is not used (it likely controls another
  crossing);
- with several relevant heads, a walk head whose distance is unknown (box clipped by the frame
  edge) is not used;
- a nearer relevant head showing anything other than walk/green (red, amber, countdown,
  unclear) blocks a farther walk head.
Each of these turns a walk decision into "unclear" (spoken as "wait").

Estimates are pinhole-camera guesses from an assumed physical housing height, d = f * H / h_px,
with f from the camera's horizontal field of view. The housing height of real heads varies
about 2x (single 200 mm lamp module to a two-lamp EU housing), so every estimate carries a
lo/hi range from that spread. They have not been validated against measured distances.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .geometry import focal_length_px, truncation_flags
from .types import Box, Detection

# Housing height priors in metres: (nominal, low, high).
# Pedestrian: a single 200-300 mm module (~0.3 m) up to a two-lamp EU housing (~0.75 m);
# US 16" hand/man modules are ~0.45 m. Vehicle: three 200-300 mm lamps, ~0.75-1.1 m.
SIGNAL_HEIGHT_M = {
    "pedestrian": (0.5, 0.3, 0.8),
    "vehicle": (0.95, 0.75, 1.1),
    "unknown": (0.7, 0.3, 1.1),
}
# Signs: a stop sign is 0.6-0.9 m across (US 30" = 0.76 m; EU 0.6/0.9 m).
SIGN_HEIGHT_M = {
    "stop sign": (0.75, 0.6, 0.9),
}
GENERIC_SIGN_M = (0.6, 0.3, 1.0)

# Camera presets, horizontal FOV in degrees for the frame as delivered.
# phone: main camera in landscape (~26 mm equiv). rayban: Ray-Ban Meta portrait capture,
# HFOV 68 deg per the Ray-Ban Meta camera spec (DFOV 99, VFOV 87).
CAMERA_PRESETS = {"phone": 70.0, "rayban": 68.0, "glasses": 68.0}


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        v = float(raw) if raw else default
    except ValueError:
        return default
    if not math.isfinite(v):
        return default
    return min(hi, max(lo, v))


def camera_hfov_deg(default: float = 70.0) -> float:
    """STREETSENSE_CAMERA_HFOV_DEG wins; else STREETSENSE_CAMERA=phone|rayban; else default."""
    raw = os.environ.get("STREETSENSE_CAMERA_HFOV_DEG", "").strip()
    if raw:
        return _env_float("STREETSENSE_CAMERA_HFOV_DEG", default, 5.0, 170.0)
    preset = os.environ.get("STREETSENSE_CAMERA", "").strip().lower()
    return CAMERA_PRESETS.get(preset, default)


@dataclass(frozen=True)
class SignalDistanceConfig:
    max_walk_m: float = 40.0  # a walk head estimated farther than this is not ours
    near_m: float = 25.0  # label "near" at or below this (display / speech only)
    max_bearing_deg: float = 25.0  # walk head must be within this angle of straight ahead
    nearer_margin: float = 0.8  # another head counts as nearer if d_other < margin * d_walk

    @classmethod
    def from_env(cls) -> "SignalDistanceConfig":
        # Clamped so a typo cannot switch the caution rules off entirely.
        return cls(
            max_walk_m=_env_float("STREETSENSE_SIGNAL_MAX_WALK_M", 40.0, 5.0, 80.0),
            near_m=_env_float("STREETSENSE_SIGNAL_NEAR_M", 25.0, 2.0, 80.0),
            max_bearing_deg=_env_float("STREETSENSE_SIGNAL_MAX_BEARING_DEG", 25.0, 5.0, 50.0),
            nearer_margin=_env_float("STREETSENSE_SIGNAL_NEARER_MARGIN", 0.8, 0.5, 1.0),
        )


@dataclass(frozen=True)
class DistanceEstimate:
    m: float | None  # nominal estimate, None when unknown
    lo: float | None
    hi: float | None
    source: str  # box | box+depth | depth-conflict | unknown
    bearing_deg: float  # negative = left of straight ahead
    band: str  # near | far | unknown
    reason: str = ""

    def as_attrs(self) -> dict:
        return {
            "signal_distance_m": self.m,
            "signal_distance_lo_m": self.lo,
            "signal_distance_hi_m": self.hi,
            "signal_distance_source": self.source,
            "signal_bearing_deg": self.bearing_deg,
            "signal_band": self.band,
        }


def size_prior(label: str, kind: str = "unknown") -> tuple[float, float, float] | None:
    if label == "traffic light":
        return SIGNAL_HEIGHT_M.get(kind, SIGNAL_HEIGHT_M["unknown"])
    if label in SIGN_HEIGHT_M:
        return SIGN_HEIGHT_M[label]
    if "sign" in label:
        return GENERIC_SIGN_M
    return None


def bearing_deg(box: Box, image_width: int, hfov_deg: float) -> float:
    f = focal_length_px(image_width, hfov_deg)
    return math.degrees(math.atan((box.cx - image_width / 2) / f))


def _band(m: float | None, cfg: SignalDistanceConfig) -> str:
    if m is None:
        return "unknown"
    return "near" if m <= cfg.near_m else "far"


def estimate(box: Box, label: str, kind: str, image_width: int, image_height: int,
             hfov_deg: float, cfg: SignalDistanceConfig | None = None) -> DistanceEstimate:
    """Pinhole distance from apparent housing height. Unknown when the box touches a frame edge."""
    cfg = cfg or SignalDistanceConfig()
    brg = bearing_deg(box, image_width, hfov_deg)
    prior = size_prior(label, kind)
    if prior is None or box.h < 3:
        return DistanceEstimate(None, None, None, "unknown", brg, "unknown", "no size prior")
    flags = truncation_flags(box, image_width, image_height)
    if flags["top"] or flags["bottom"] or flags["left"] or flags["right"]:
        return DistanceEstimate(None, None, None, "unknown", brg, "unknown", "clipped by frame edge")
    f = focal_length_px(image_width, hfov_deg)
    # Slant range: account for the off-axis angle so a head at the frame side is not under-read.
    k = f / box.h / max(1e-6, math.cos(math.radians(brg)))
    nom, lo, hi = (k * v for v in prior)
    return DistanceEstimate(nom, lo, hi, "box", brg, _band(nom, cfg))


def fuse_depth(est: DistanceEstimate, depth_m: float | None, cfg: SignalDistanceConfig | None = None) -> DistanceEstimate:
    """Combine with a calibrated monocular-depth reading.

    The depth scale is itself calibrated from person/vehicle box heights, so it is not an
    independent measurement. Agreement (depth inside the box range, with 1.5x slack) gives the
    geometric mean; disagreement widens the range to cover both and is flagged.
    """
    cfg = cfg or SignalDistanceConfig()
    if depth_m is None or not math.isfinite(depth_m) or depth_m <= 0:
        return est
    if est.m is None:
        return DistanceEstimate(depth_m, depth_m * 0.6, depth_m * 1.6, "depth", est.bearing_deg,
                                _band(depth_m, cfg), "depth only")
    if est.lo / 1.5 <= depth_m <= est.hi * 1.5:
        m = math.sqrt(est.m * depth_m)
        return DistanceEstimate(m, min(est.lo, depth_m), max(est.hi, depth_m), "box+depth",
                                est.bearing_deg, _band(m, cfg))
    lo, hi = min(est.lo, depth_m), max(est.hi, depth_m)
    # Keep the nearer of the two as nominal: "nearer" makes the near-head rules stricter.
    m = min(est.m, depth_m)
    return DistanceEstimate(m, lo, hi, "depth-conflict", est.bearing_deg, _band(m, cfg),
                            f"box {est.m:.0f} m vs depth {depth_m:.0f} m")


def get_estimate(d: Detection, image_width: int, image_height: int, hfov_deg: float,
                 cfg: SignalDistanceConfig | None = None) -> DistanceEstimate:
    """Use the pipeline's stored estimate when present, else compute from the box."""
    a = d.attrs
    if "signal_distance_source" in a:
        return DistanceEstimate(a.get("signal_distance_m"), a.get("signal_distance_lo_m"),
                                a.get("signal_distance_hi_m"), a["signal_distance_source"],
                                float(a.get("signal_bearing_deg", 0.0)), a.get("signal_band", "unknown"))
    return estimate(d.box, d.label, a.get("signal_kind", "unknown"), image_width, image_height, hfov_deg, cfg)


def depth_rel(d: Detection) -> float | None:
    """Relative inverse depth at the head centre (larger = nearer), set by the pipeline when depth is on."""
    v = d.attrs.get("signal_depth_rel")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def sample_depth_rel(depth_map, box: Box) -> float | None:
    """Median relative depth over the central half of a box (the lamp face, not the pole)."""
    import numpy as np

    h, w = depth_map.shape[:2]
    bw, bh = box.x2 - box.x1, box.y2 - box.y1
    x1 = int(max(0, min(w - 1, box.x1 + bw / 4)))
    x2 = int(max(x1 + 1, min(w, box.x2 - bw / 4)))
    y1 = int(max(0, min(h - 1, box.y1 + bh / 4)))
    y2 = int(max(y1 + 1, min(h, box.y2 - bh / 4)))
    patch = depth_map[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    v = float(np.median(patch))
    return v if math.isfinite(v) and v > 0 else None


def _nearer(eo: DistanceEstimate, ew: DistanceEstimate, o: Detection, w: Detection,
            cfg: SignalDistanceConfig) -> bool:
    """True when either cue says `o` is clearly nearer than `w` (either cue is enough: caution only)."""
    if eo.m is not None and ew.m is not None and eo.m < cfg.nearer_margin * ew.m:
        return True
    ro, rw = depth_rel(o), depth_rel(w)
    return ro is not None and rw is not None and ro * cfg.nearer_margin > rw


def walk_caution(walk: Detection, relevant: list[Detection], image_width: int, image_height: int,
                 hfov_deg: float, cfg: SignalDistanceConfig | None = None,
                 unreadable: list[Detection] | tuple = ()) -> tuple[str, str] | None:
    """(code, detail) explaining why a walk/green head must not control the crossing, or None.

    Codes: side | far | unknown | nearer. Any code turns the decision into "unclear" (wait).
    `unreadable` = pedestrian/unknown-kind heads whose state could not be read (e.g. bloomed red);
    one that is clearly nearer and roughly ahead also blocks.
    """
    cfg = cfg or SignalDistanceConfig()
    ew = get_estimate(walk, image_width, image_height, hfov_deg, cfg)
    if abs(ew.bearing_deg) > cfg.max_bearing_deg:
        return "side", f"walk signal {abs(ew.bearing_deg):.0f} deg off to the side"
    if ew.m is not None and ew.m > cfg.max_walk_m:
        return "far", f"walk signal about {ew.m:.0f} m away (limit {cfg.max_walk_m:.0f} m)"
    others = [d for d in relevant if d is not walk]
    if ew.m is None and others:
        return "unknown", "walk signal distance unknown with several signals"
    for o in others:
        st = o.attrs.get("light_state")
        if st in ("walk", "green"):
            continue
        eo = get_estimate(o, image_width, image_height, hfov_deg, cfg)
        if _nearer(eo, ew, o, walk, cfg):
            return "nearer", f"nearer signal shows {st}"
        if eo.m is None:
            return "unknown", f"another signal ({st}) at unknown distance"
    for o in unreadable:
        if o is walk:
            continue
        eo = get_estimate(o, image_width, image_height, hfov_deg, cfg)
        if abs(eo.bearing_deg) > cfg.max_bearing_deg:
            continue
        if _nearer(eo, ew, o, walk, cfg):
            return "nearer", "nearer signal unreadable"
    return None


CAUTION_SPEECH = {
    "side": "walk signal is off to the side, not this crossing, wait",
    "far": "walk signal is far away, not this crossing, wait",
    "unknown": "several signals, can't tell which is ours, wait",
    "nearer": "a nearer signal is not showing walk, wait",
}


def spoken_distance(est: DistanceEstimate | None) -> str:
    """'about 15 metres ahead' / '4 metres to your left' / '' when unknown."""
    if est is None or est.m is None:
        return ""
    m = est.m
    if m < 1.5:
        dist = "1 metre"
    elif m < 6:
        dist = f"{round(m)} metres"
    else:
        dist = f"about {int(round(m / 5) * 5)} metres"
    b = est.bearing_deg
    where = "ahead" if abs(b) <= 12 else ("to your left" if b < 0 else "to your right")
    return f"{dist} {where}"
