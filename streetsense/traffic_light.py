"""Traffic-light state from a detection crop, using HSV colour of the lit lamp.

Works for vehicle lights (red/yellow/green) and for pedestrian signals (red/green figure;
US-style white "walk" / orange "hand"; EU-style tall red/green man silhouette). No learned
weights: it is a transparent heuristic that can be unit-tested, and a Meta vision model can
be asked to confirm on demand.
"""

from __future__ import annotations

from typing import Callable

import os
import time
from collections import Counter, deque
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .geometry import iou
from .types import Box

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


def _bottom_gap(mask: np.ndarray) -> float:
    """Widest gap between lit runs along the bottom band of a figure, as a fraction of its width.

    A walking man has two feet apart at the bottom. Seven-segment digits (0-9), arrows and
    round lamps end in one bar, one stroke or one curve, so their bottom band is one run.
    """
    cols = mask.any(axis=0)
    runs: list[tuple[int, int]] = []
    start = None
    for j, v in enumerate(list(cols) + [False]):
        if v and start is None:
            start = j
        elif not v and start is not None:
            runs.append((start, j))
            start = None
    if len(runs) < 2:
        return 0.0
    return max(b[0] - a[1] for a, b in zip(runs, runs[1:])) / max(1, mask.shape[1])


def eu_figure_enabled() -> bool:
    return os.environ.get("STREETSENSE_EU_FIGURE", "1").strip().lower() not in ("0", "false", "no", "off")


def lamp_figure(crop_bgr: np.ndarray, kind: str = "unknown") -> tuple[bool, dict]:
    """Per-lamp check for a lit green walking figure on a European-style head.

    Built for the cases the colour vote misses: a dotted LED outline that fragments at 64 px,
    a sun-washed figure that is pale green or white, and a night figure that blooms into a white
    core with a green halo. Every condition below must hold, so a miss means "not walk":

    * never on a vehicle-shaped head; the box must be dark housing (not a sign or screen);
    * figure pixels = green (relaxed saturation) plus a white core whose surrounding ring is
      green (the halo), joined by a small morphological close;
    * the largest blob is a standing shape: 18-80 % of the head height, aspect 1.3-4.5, not
      solid (fill <= 0.65), not touching the crop edge (sky or a wall leaking into the box);
    * two feet apart at the bottom (rejects digits, arrows and round lamps);
    * in the lower half of a tall head (the walk lamp sits under the stop lamp);
    * no other lit green away from the figure (a countdown or second lamp), no lit red.
    """
    info: dict = {}
    if kind == "vehicle" or crop_bgr is None or crop_bgr.size == 0:
        return False, {"why": "kind"}
    h0, w0 = crop_bgr.shape[:2]
    if h0 < 24 or w0 < 12:
        return False, {"why": "small"}
    H = 128
    W = max(8, round(w0 * H / h0))
    c = cv2.resize(crop_bgr, (W, H), interpolation=cv2.INTER_AREA if h0 > H else cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dark = float((val < 100).mean())
    info["dark"] = round(dark, 2)
    if dark < 0.3:
        return False, {**info, "why": "bright box"}
    lo, hi = HUE_RANGES["green"][0]
    gh = (hue >= lo) & (hue <= hi)
    green = gh & (sat >= 50) & (val >= 120)
    core = (sat < 70) & (val >= 220)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fig = green.copy()
    if core.any():
        ring = cv2.dilate(core.astype(np.uint8), k7, iterations=2).astype(bool) & ~core
        halo = float((ring & gh & (sat >= 30) & (val >= 80)).sum()) / max(1, int(ring.sum()))
        info["halo"] = round(halo, 2)
        if halo >= 0.12:
            fig |= core
    kc = max(3, int(round(H * 0.04)) | 1)
    closed = cv2.morphologyEx(fig.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        return False, {**info, "why": "no figure"}
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = (int(v) for v in st[i])
    comp = lab == i
    aspect, fill, cy = h / max(1, w), a / max(1, w * h), (y + h / 2) / H
    y0 = y + int(h * 0.86)
    gap = _bottom_gap(comp[y0:y + h, x:x + w])
    near = np.zeros_like(comp)
    my, mx = int(0.25 * H), int(0.25 * W)
    near[max(0, y - my):y + h + my, max(0, x - mx):x + w + mx] = True
    other_green = int((gh & (sat >= MIN_SAT) & (val >= MIN_VAL) & ~near).sum())
    red = int((_hue_mask(hsv, HUE_RANGES["red"]) & (sat >= 120) & (val >= 170)).sum())
    info.update(h=round(h / H, 2), aspect=round(aspect, 2), fill=round(fill, 2), cy=round(cy, 2),
                area=a, gap=round(gap, 2), other_green=round(other_green / max(1, a), 2),
                red=round(red / max(1, a), 2))
    why = []
    if not 0.18 <= h / H <= 0.8:
        why.append("height")
    if not 1.3 <= aspect <= 4.5:
        why.append("aspect")
    if fill > 0.65:
        why.append("solid")
    if a < 40:
        why.append("area")
    if x <= 0 or y <= 0 or x + w >= W or y + h >= H:
        why.append("edge")
    if gap < 0.12:
        why.append("feet")
    if h0 / w0 >= 1.4 and cy < 0.5:
        why.append("upper lamp")
    if other_green >= 0.2 * a:
        why.append("other green")
    if red >= 0.3 * a:
        why.append("red lit")
    info["why"] = ",".join(why)
    return not why, info


def classify_light(crop_bgr: np.ndarray, kind: str = "unknown") -> LightReading:
    """Colour vote (`_classify_colour`), then the per-lamp EU figure check for heads that read
    green or unknown. The figure check can only turn green/unknown into walk on a non-vehicle
    head; it never touches red, yellow, countdown or the US white-walk rule. With
    STREETSENSE_EU_FIGURE=0 the old colour-vote reader runs unchanged."""
    r = _classify_colour(crop_bgr, kind)
    if not eu_figure_enabled():
        return r
    if r.state == "walk" and r.figure:
        # The coarse 64 px silhouette also accepts a tall hollow green digit ("8", "0") from a
        # countdown box. A figure walk must pass the stricter per-lamp check (feet apart) too.
        return r if lamp_figure(crop_bgr, kind)[0] else replace(r, state="green")
    if kind == "vehicle" or r.state not in ("green", "unknown"):
        return r
    ok, info = lamp_figure(crop_bgr, kind)
    if not ok:
        return r
    return LightReading("walk", max(0.6, r.confidence), r.lit_fraction, info.get("cy"), False, True)


def _classify_colour(crop_bgr: np.ndarray, kind: str = "unknown") -> LightReading:
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
    # Not on vehicle-shaped housings (h/w > 2.6): a side-on or unlit 3-lamp head lets white sky or
    # wall show through between its visors, which read as a walk figure (seen on eu_ped_b).
    if kind != "vehicle" and white_top > 0.05 and white_top > coloured_top and float(border.mean()) < 0.15:
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


def beside_display(frame_bgr: np.ndarray, box: Box) -> bool:
    """True when a lit display (countdown digits or another lamp) sits right beside the head.

    Some EU heads (Most, CZ) put a green countdown next to the green man. The reader cannot read
    the digits, so a figure walk with a lit display beside it is downgraded to green (no walk):
    a single digit there means the phase is ending. Checks strips 0.9 box widths wide on both
    sides, at the head's height, for a lit coloured blob at least 15 % of the head height.
    """
    Hf, Wf = frame_bgr.shape[:2]
    b = box.clip(Wf, Hf)
    if b.w < 4 or b.h < 4:
        return False
    y1, y2 = int(b.y1), int(b.y2)
    sw = int(round(0.9 * b.w))
    for x1, x2 in ((int(b.x1) - sw, int(b.x1)), (int(b.x2), int(b.x2) + sw)):
        x1, x2 = max(0, x1), min(Wf, x2)
        if x2 - x1 < 3:
            continue
        hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        lit = (hsv[..., 1] >= MIN_SAT) & (hsv[..., 2] >= MIN_VAL)
        lit &= _hue_mask(hsv, HUE_RANGES["red"] + HUE_RANGES["orange"] + HUE_RANGES["yellow"] + HUE_RANGES["green"])
        if not lit.any():
            continue
        k = max(3, int(round(0.04 * b.h)) | 1)
        m = cv2.morphologyEx(lit.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1 and int(st[1:, cv2.CC_STAT_HEIGHT].max()) >= 0.15 * b.h:
            return True
    return False


def figure_walk_guard(reading: LightReading, frame_bgr: np.ndarray, box: Box) -> LightReading:
    """Downgrade an EU figure walk to green when a lit display sits beside the head."""
    if reading.state == "walk" and reading.figure and beside_display(frame_bgr, box):
        return replace(reading, state="green", countdown=True)
    return reading


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
    """Majority vote over the last N readings per track, so one bad frame cannot flip red to green.

    Untracked detections (track_id is None) use an IoU/position key and the same vote rule —
    internal states like `countdown` never leave this layer as-is for untracked heads.
    Countdown frames hold the last positive colour only within `countdown_hold_s`.
    """

    def __init__(
        self,
        window: int = 5,
        min_votes: int = 3,
        countdown_hold_s: float = 1.0,
        clock: Callable[[], float] | None = None,
        iou_match: float = 0.4,
    ):
        self.window = window
        self.min_votes = min_votes
        self.countdown_hold_s = countdown_hold_s
        self.clock = clock or time.monotonic
        self.iou_match = iou_match
        self._hist: dict[int | str, deque[str]] = {}
        self._last_positive_ts: dict[int | str, float] = {}
        self._last_positive_state: dict[int | str, str] = {}
        self._untracked_boxes: dict[str, Box] = {}
        self._next_uid = 0

    def _key_for_untracked(self, box: Box | None) -> str:
        if box is None:
            # No position: never promote a single-frame walk.
            self._next_uid += 1
            return f"anon:{self._next_uid}"
        for k, prev in list(self._untracked_boxes.items()):
            if iou(prev, box) >= self.iou_match:
                self._untracked_boxes[k] = box
                return k
        self._next_uid += 1
        k = f"u:{self._next_uid}"
        self._untracked_boxes[k] = box
        return k

    def update(self, track_id: int | None, state: str, box: Box | None = None) -> str:
        key: int | str
        if track_id is None:
            key = self._key_for_untracked(box)
            # Internal countdown must not reach the alert layer for untracked heads.
            if state == "countdown":
                return "unknown"
        else:
            key = track_id

        now = float(self.clock())
        q = self._hist.setdefault(key, deque(maxlen=self.window))

        if state == "countdown":
            last_ts = self._last_positive_ts.get(key)
            last_st = self._last_positive_state.get(key)
            if last_ts is not None and last_st in ("walk", "red", "yellow", "green"):
                if now - last_ts <= self.countdown_hold_s:
                    return last_st  # short flicker hold
            # Hold expired: report uncertain / countdown, not a stale walk.
            return "countdown" if last_st else "unknown"

        q.append(state)
        if state not in ("unknown", "countdown"):
            self._last_positive_ts[key] = now
            self._last_positive_state[key] = state

        known = [s for s in q if s not in ("unknown", "countdown")]
        if not known:
            return "unknown"
        best, votes = Counter(known).most_common(1)[0]
        return best if votes >= self.min_votes else "unknown"


def apply_signal_cnn(reading: LightReading, probs, kind: str, t_walk: float, t_veto: float = 0.8) -> LightReading:
    """Combine the hand-rule reading with the crop classifier (streetsense/signal_cnn.py).

    ``probs`` = (p_walk, p_stop, p_countdown, p_other) or None (classifier off / failed).
    It can only be one input, and every disagreement resolves towards "not walk":
    * veto: a hand-rule walk becomes countdown/unknown unless the classifier also gives
      p_walk >= 0.5 and no non-walk class reaches ``t_veto``;
    * promote: a non-vehicle head the hand rules read as green/unknown (no red, yellow or
      countdown seen) becomes a figure walk only when p_walk >= ``t_walk`` (calibrated for
      zero false walk on validation). figure_walk_guard and the 3-of-5 smoother still apply.
    """
    if probs is None:
        return reading
    pw, ps, pc, po = (float(v) for v in probs)
    if reading.state == "walk":
        if max(ps, pc, po) >= t_veto or pw < 0.5:
            if pc >= max(ps, po):
                return replace(reading, state="countdown", countdown=True)
            return replace(reading, state="unknown")
        return reading
    if kind == "vehicle" or reading.state not in ("green", "unknown") or reading.countdown:
        return reading
    if pw >= t_walk:
        return replace(reading, state="walk", confidence=pw, figure=True)
    return reading


def cnn_hold_veto(smoothed: str, raw: str, probs, t_veto: float = 0.8) -> str:
    """Stop the smoother's history (3-of-5 vote, 1 s countdown flicker hold) from carrying a
    walk through a frame where the classifier is confident the head is NOT walk right now
    (stop, countdown digits, or off). Never turns anything into walk."""
    if smoothed != "walk" or probs is None:
        return smoothed
    pw, ps, pc, po = (float(v) for v in probs)
    if max(ps, pc, po) >= t_veto and pw < 0.5:
        return "countdown" if (raw == "countdown" or pc >= max(ps, po)) else "unknown"
    return smoothed
