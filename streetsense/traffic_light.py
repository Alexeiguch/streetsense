"""Traffic-light state from a detection crop, using HSV colour of the lit lamp.

Works for vehicle lights (red/yellow/green) and for pedestrian signals (red/green figure;
US-style white "walk" / orange "hand"; EU-style tall red/green man silhouette). No learned
weights: it is a transparent heuristic that can be unit-tested, and a Meta vision model can
be asked to confirm on demand.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV hue is 0..179.
HUE_RANGES = {
    "red": [(0, 7), (165, 179)],
    "orange": [(8, 19)],  # US "don't walk" hand, some amber lamps
    "yellow": [(20, 34)],
    "green": [(40, 100)],  # LED greens are often cyan-ish (hue 80-95)
}
MIN_SAT = 90
MIN_VAL = 140
MIN_DARK_FRAC = 0.2


@dataclass
class LightReading:
    state: str  # "red" | "yellow" | "green" | "walk" | "countdown" | "unknown"
    confidence: float  # fraction of lit pixels that agree with `state`
    lit_fraction: float  # lit pixels / crop pixels
    lamp_position: float | None  # 0 = top of housing, 1 = bottom, for the dominant colour
    countdown: bool = False  # US-style: walk figure plus a lit (orange) countdown below it
    figure: bool = False  # tall lit silhouette (EU red/green man) -> treat housing as pedestrian


def _hue_mask(hsv: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(hsv.shape[:2], dtype=bool)
    for lo, hi in ranges:
        m |= (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)
    return m



def _figure_silhouette(lit_mask: np.ndarray) -> tuple[bool, float]:
    """True when the lit pixels form a tall standing/walking figure (EU ped man/woman).

    Vehicle circular lamps are roughly round (aspect ~1). Pedestrian stick-figures are
    elongated vertically. Returns (is_figure, aspect_h_over_w).
    """
    if lit_mask is None or not lit_mask.any():
        return False, 0.0
    # Largest connected component of lit pixels
    u8 = lit_mask.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n <= 1:
        return False, 0.0
    # skip background label 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    area = int(stats[idx, cv2.CC_STAT_AREA])
    if w < 2 or h < 4 or area < 8:
        return False, 0.0
    aspect = h / max(1, w)
    fill = area / max(1.0, float(w * h))
    # Tall sparse stick-figure (EU man). Round LED lamps are ~aspect 1 and densely filled.
    is_fig = (
        aspect >= 1.8
        and fill < 0.55
        and h >= 0.35 * lit_mask.shape[0]
        and w <= 0.55 * lit_mask.shape[1]
    )
    return is_fig, float(aspect)


def classify_light(crop_bgr: np.ndarray, kind: str = "unknown") -> LightReading:
    """Classify the lit state of a traffic/pedestrian light crop (BGR uint8).

    `kind` matters for orange: on a vehicle head an orange lamp is amber ("yellow"); on a
    pedestrian head it is the "don't walk" hand ("red"). Unknown kind is treated as pedestrian,
    so the failure mode is "red" (wait), never "go".
    """
    if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 4:
        return LightReading("unknown", 0.0, 0.0, None)
    h, w = crop_bgr.shape[:2]
    scale = 64 / max(h, w)  # normalise size so thresholds behave the same on tiny and large crops
    crop_bgr = cv2.resize(crop_bgr, (max(4, round(w * scale)), max(4, round(h * scale))),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    lit = (sat >= MIN_SAT) & (val >= MIN_VAL)
    H = hsv.shape[0]
    # A real signal head is mostly dark housing around small lamps. Bright, uniformly lit boxes are
    # screens/billboards/signs that the detector mistook for a light: refuse to read them.
    dark_frac = float((val < 100).mean())
    if dark_frac < MIN_DARK_FRAC:
        return LightReading("unknown", 0.0, float(lit.mean()), None)
    total = lit.size
    counts = {name: int((lit & _hue_mask(hsv, r)).sum()) for name, r in HUE_RANGES.items()}
    n_lit = sum(counts.values())

    # US/Canada walk figure: bright, low-saturation (white / bluish-white) in the upper half,
    # often with an orange countdown in the lower half. Vehicle heads never show white, so this
    # runs for every kind. Guard: the white must sit inside the housing, not along the crop border
    # (sky or a white wall leaking into a loose box must not become a "walk" = go signal).
    white = (sat < 100) & (val > 190) & ~lit
    Wd = hsv.shape[1]
    my, mx = max(1, H // 10), max(1, Wd // 6)
    inner_top = white[my:H // 2, mx:Wd - mx]
    border = np.concatenate([white[:my].ravel(), white[:, :mx].ravel(), white[:, Wd - mx:].ravel()])
    white_top = float(inner_top.mean()) if inner_top.size else 0.0
    coloured_top = float(lit[:H // 2].mean())
    if white_top > 0.05 and white_top > coloured_top and float(border.mean()) < 0.15:
        countdown = float(lit[H // 2:].mean()) > 0.03
        return LightReading("walk", min(1.0, white_top * 5), white_top / 2, 0.25, countdown)

    if n_lit < max(3, 0.01 * total):
        return LightReading("unknown", 0.0, n_lit / total, None)

    orange_to = "yellow" if kind == "vehicle" else "red"
    if kind == "unknown" and counts["orange"] > counts["red"]:
        # An orange lamp in the middle third is a vehicle amber; a pedestrian hand sits on top.
        oy = np.nonzero(lit & _hue_mask(hsv, HUE_RANGES["orange"]))[0]
        if oy.size and 0.36 < oy.mean() / max(1, H - 1) < 0.64:
            orange_to = "yellow"
    merged = {"red": counts["red"], "yellow": counts["yellow"], "green": counts["green"]}
    merged[orange_to] += counts["orange"]
    state, n = max(merged.items(), key=lambda kv: kv[1])
    conf = n / n_lit
    if conf < 0.55:
        return LightReading("unknown", conf, n_lit / total, None)

    hue = _hue_mask(hsv, HUE_RANGES[state])
    if state == orange_to:
        hue |= _hue_mask(hsv, HUE_RANGES["orange"])
    ys = np.nonzero(lit & hue)[0]
    pos = float(ys.mean() / max(1, H - 1)) if ys.size else None
    if state == "red" and kind != "vehicle" and pos is not None and pos > 0.62 and coloured_top < 0.01:
        # Stop lamps / hands sit at the top of every standard head. Red-orange light only in the
        # lower part, with the top dark, is a countdown display while the walk figure is between
        # LED refreshes (or the camera missed it). Not a colour vote: the smoother ignores it.
        return LightReading("countdown", float(conf), n_lit / total, pos, True)
    # EU-style pedestrian heads show a standing/walking stick figure rather than a round lamp.
    # Tall lit silhouette + red/green => mark as figure; green figure is spoken as walk.
    fig_mask = lit & hue
    is_fig, _ = _figure_silhouette(fig_mask)
    if is_fig and state == "green" and kind != "vehicle":
        return LightReading("walk", float(conf), n_lit / total, pos, False, True)
    if is_fig and state in ("red", "green") and kind != "vehicle":
        return LightReading(state, float(conf), n_lit / total, pos, False, True)
    return LightReading(state, float(conf), n_lit / total, pos, False, is_fig and state in ("red", "green"))


def signal_kind(box_w: float, box_h: float) -> str:
    """Prior from housing shape only. Real heads overlap a lot (on the DC sample both vehicle and
    pedestrian boxes are h/w 2.1-2.3), so we only commit at the extremes; the pipeline upgrades a
    track to "pedestrian" when a walk figure is seen. Unknown is spoken as "traffic light"."""
    if box_w <= 0:
        return "unknown"
    r = box_h / box_w
    if r > 2.6:
        return "vehicle"
    if r < 1.4:
        return "pedestrian"
    return "unknown"


class LightStateSmoother:
    """Majority vote over the last N readings per track, so one bad frame cannot flip red to green."""

    def __init__(self, window: int = 5, min_votes: int = 3):
        self.window = window
        self.min_votes = min_votes
        self._hist: dict[int, deque[str]] = {}

    def update(self, track_id: int | None, state: str) -> str:
        if track_id is None:
            return state
        q = self._hist.setdefault(track_id, deque(maxlen=self.window))
        # A countdown-only frame says "the crossing phase is still running" but not which colour:
        # hold the current vote instead of diluting it. A steady hand (red, top) still flips it.
        if state != "countdown":
            q.append(state)
        known = [s for s in q if s != "unknown"]
        if not known:
            return "unknown"
        best, votes = Counter(known).most_common(1)[0]
        # Absolute vote count: a brand-new track must agree on several frames before we announce it.
        return best if votes >= self.min_votes else "unknown"
