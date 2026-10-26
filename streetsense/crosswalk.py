"""Zebra-crossing detector: a classical-CV heuristic on stripe patterns.

Idea: a zebra crossing is a set of >= 3 bright, low-saturation, elongated, similarly
oriented bars on the ground plane (lower part of the image). We threshold relative to the
local road brightness, keep elongated blobs, and look for a group of them with a shared
orientation. Cheap (~ few ms at 640 px) and testable; misses faded or occluded crossings.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import Box


@dataclass
class CrosswalkResult:
    found: bool
    box: Box | None
    n_stripes: int
    score: float
    stripes: list[np.ndarray]  # rotated-rect corner points, for drawing


def detect_crosswalk(
    frame_bgr: np.ndarray,
    roi_top_frac: float = 0.45,
    work_width: int = 640,
    min_stripes: int = 3,
    ground_mask: np.ndarray | None = None,
    diff_thresh: int = 15,
    blur_k: int = 61,
    min_extent: float = 0.35,
) -> CrosswalkResult:
    """`ground_mask` (HxW bool at frame resolution, e.g. road|sidewalk from SegFormer) restricts
    stripes to the ground plane, which removes most false positives from signs and buildings."""
    h0, w0 = frame_bgr.shape[:2]
    scale = work_width / w0
    small = cv2.resize(frame_bgr, (work_width, int(h0 * scale)), interpolation=cv2.INTER_AREA)
    H, W = small.shape[:2]
    y0 = int(H * roi_top_frac)
    roi = small[y0:]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = hsv[..., 2]

    # Paint is brighter than the asphalt around it: compare to a large local mean (top-hat-like).
    local = cv2.blur(gray, (blur_k, blur_k))
    bright = (gray.astype(np.int16) - local.astype(np.int16) > diff_thresh) & (gray > 100) & (hsv[..., 1] < 70)
    if ground_mask is not None:
        g = cv2.resize(ground_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)[y0:]
        # Paint is classed as road; dilate a little so stripe edges next to people/cars survive.
        g = cv2.dilate(g, np.ones((9, 9), np.uint8))
        bright &= g.astype(bool)
    mask = bright.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = roi.shape[0] * roi.shape[1]
    cands = []  # (angle_deg in [0,180), rect, area)
    for c in contours:
        area = cv2.contourArea(c)
        if area < roi_area * 0.0008 or area > roi_area * 0.2:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        long_side, short_side = max(rw, rh), min(rw, rh)
        if short_side < 2 or long_side / short_side < 2.2:
            continue
        # Solid bars fill their rotated rectangle; text, kerbs and car glare usually do not.
        if area / (rw * rh + 1e-6) < min_extent:
            continue
        orient = ang if rw >= rh else ang + 90
        cands.append((orient % 180, rect, area))

    if len(cands) < min_stripes:
        return CrosswalkResult(False, None, len(cands), 0.0, [])

    # Group by orientation (circular, 180-degree period); keep the largest group.
    best: list = []
    for a, _, _ in cands:
        grp = [c for c in cands if min(abs(c[0] - a), 180 - abs(c[0] - a)) < 15]
        if len(grp) > len(best):
            best = grp
    if len(best) < min_stripes:
        return CrosswalkResult(False, None, len(best), 0.0, [])

    # Stripes of one crossing have similar size; drop strong outliers.
    areas = np.array([g[2] for g in best])
    med = np.median(areas)
    best = [g for g in best if 0.25 * med <= g[2] <= 4 * med]
    if len(best) < min_stripes:
        return CrosswalkResult(False, None, len(best), 0.0, [])

    pts = []
    stripes = []
    for _, rect, _ in best:
        p = cv2.boxPoints(rect)
        p[:, 1] += y0
        p /= scale
        stripes.append(p.astype(np.int32))
        pts.append(p)
    allp = np.concatenate(pts)
    x1, y1 = allp.min(axis=0)
    x2, y2 = allp.max(axis=0)
    score = min(1.0, len(best) / 6) * min(1.0, float(areas.sum()) / (roi_area * 0.02))
    box = Box(float(max(0, x1)), float(max(0, y1)), float(min(w0, x2)), float(min(h0, y2)))
    return CrosswalkResult(True, box, len(best), float(score), stripes)


class CrosswalkHysteresis:
    """Report a crossing only when seen in >= `k` of the last `n` checks (and keep the last box)."""

    def __init__(self, n: int = 5, k: int = 2):
        from collections import deque

        self.hist = deque(maxlen=n)
        self.k = k
        self.last_found: CrosswalkResult | None = None

    def update(self, r: CrosswalkResult) -> CrosswalkResult:
        self.hist.append(r.found)
        if r.found:
            self.last_found = r
        if sum(self.hist) >= self.k and self.last_found is not None:
            return self.last_found if not r.found else r
        return CrosswalkResult(False, None, r.n_stripes, 0.0, [])
