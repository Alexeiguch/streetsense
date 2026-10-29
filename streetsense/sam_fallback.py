"""Asynchronous SAM 3.1 fallback for traffic/pedestrian signal-head boxes (prototype, not wired into pipeline.py).

Safety contract:
- This helper only returns candidate *signal-head boxes* (frame pixel coords). It has no notion of walk/stop and
  exposes no state, colour or crossing decision. Its boxes may only feed the local signal reader, which still needs a
  Muse Spark "yes" before anything is spoken as a go; any doubt = wait.
- Results are advisory and stale-aware: callers must drop results older than `max_age_s`.
- Failures (budget, network, parse) yield an empty result, never an exception into the caller's loop.

Recommended config from docs/SAM_LATENCY.md: full frame (top_fraction=1.0; the upper-60% crop lost recall),
longest side 768, JPEG q85, prompt "traffic light", one call in flight, at most ~1 Hz, and only when the local
detector has no confident light.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class SignalBoxes:
    """Candidate signal-head boxes. Deliberately carries no crossing/walk field."""

    capture_ts: float
    boxes: tuple[tuple[float, float, float, float, float], ...] = ()  # x1, y1, x2, y2, conf in frame pixels
    latency_s: float = 0.0
    error: str | None = None

    def fresh(self, now: float, max_age_s: float) -> bool:
        return self.error is None and (now - self.capture_ts) <= max_age_s


@dataclass
class SamFallbackConfig:
    prompt: str = "traffic light"
    top_fraction: float = 1.0
    max_side: int = 768
    jpeg_quality: int = 85
    min_interval_s: float = 1.0
    min_conf: float = 0.3
    max_age_s: float = 1.5
    max_boxes: int = 8


SegmentFn = Callable[[np.ndarray, str], Any]  # returns object with .boxes like MetaSamSegmenter.segment


class SamSignalFallback:
    """Fire-and-forget background SAM requests; poll `latest()` for the newest result."""

    def __init__(self, segment_fn: SegmentFn | None = None, cfg: SamFallbackConfig | None = None, clock=time.monotonic):
        self.cfg = cfg or SamFallbackConfig()
        self._clock = clock
        if segment_fn is None:
            from streetsense.meta_api import MetaSamSegmenter

            seg = MetaSamSegmenter(max_image_side=self.cfg.max_side)
            seg_q = self.cfg.jpeg_quality

            def segment_fn(img: np.ndarray, prompt: str):  # noqa: E306
                return seg.segment(img, prompt, decode_masks=False, jpeg_quality=seg_q)

        self._segment = segment_fn
        self._lock = threading.Lock()
        self._busy = False
        self._last_start = -1e9
        self._latest: SignalBoxes | None = None

    def maybe_request(self, frame_bgr: np.ndarray, capture_ts: float, *, have_confident_light: bool) -> bool:
        """Start a background call if no confident local light, not busy, and rate limit allows."""
        if have_confident_light:
            return False
        now = self._clock()
        with self._lock:
            if self._busy or now - self._last_start < self.cfg.min_interval_s:
                return False
            self._busy = True
            self._last_start = now
        crop = np.ascontiguousarray(frame_bgr[: max(1, int(frame_bgr.shape[0] * self.cfg.top_fraction))])
        threading.Thread(target=self._run, args=(crop, capture_ts), daemon=True).start()
        return True

    def run_sync(self, frame_bgr: np.ndarray, capture_ts: float) -> SignalBoxes:
        crop = np.ascontiguousarray(frame_bgr[: max(1, int(frame_bgr.shape[0] * self.cfg.top_fraction))])
        return self._compute(crop, capture_ts)

    def _run(self, crop: np.ndarray, capture_ts: float) -> None:
        try:
            res = self._compute(crop, capture_ts)
            with self._lock:
                if self._latest is None or res.capture_ts >= self._latest.capture_ts:
                    self._latest = res
        finally:
            with self._lock:
                self._busy = False

    def _compute(self, crop: np.ndarray, capture_ts: float) -> SignalBoxes:
        t0 = time.perf_counter()
        try:
            seg = self._segment(crop, self.cfg.prompt)
            boxes = to_frame_boxes(getattr(seg, "boxes", []) or [], crop.shape, self.cfg.max_side)
        except Exception as e:  # budget, network, parse: degrade to "no boxes" (= local reader keeps waiting)
            return SignalBoxes(capture_ts=capture_ts, latency_s=time.perf_counter() - t0, error=type(e).__name__)
        boxes = [b for b in boxes if b[4] >= self.cfg.min_conf]
        boxes.sort(key=lambda b: -b[4])
        return SignalBoxes(capture_ts=capture_ts, boxes=tuple(boxes[: self.cfg.max_boxes]), latency_s=time.perf_counter() - t0)

    def latest(self, now: float | None = None) -> SignalBoxes | None:
        """Newest fresh result or None. Never a crossing decision."""
        now = self._clock() if now is None else now
        with self._lock:
            r = self._latest
        return r if r is not None and r.fresh(now, self.cfg.max_age_s) else None


def to_frame_boxes(sam_boxes: list[dict], crop_shape: tuple[int, ...], max_side: int) -> list[tuple[float, float, float, float, float]]:
    """SAM wire boxes are in request-image pixels (w/h = request dims); map back to crop (== top of frame) pixels."""
    H, W = crop_shape[:2]
    out = []
    for b in sam_boxes:
        try:
            rw = float(b.get("w") or 0) or W * min(1.0, max_side / max(H, W))
            rh = float(b.get("h") or 0) or H * min(1.0, max_side / max(H, W))
            sx, sy = W / rw, H / rh
            x1, y1, x2, y2 = (float(b["x1"]) * sx, float(b["y1"]) * sy, float(b["x2"]) * sx, float(b["y2"]) * sy)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append((x1, y1, x2, y2, float(b.get("c", 1.0))))
    return out
