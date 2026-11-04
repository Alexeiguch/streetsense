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
        return self.error is None and 0 <= now - self.capture_ts <= max_age_s


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
        self._generation = 0

    def invalidate(self) -> None:
        with self._lock:
            self._generation += 1
            self._latest = None

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
            generation = self._generation
        try:
            crop = frame_bgr[: max(1, int(frame_bgr.shape[0] * self.cfg.top_fraction))].copy()
            threading.Thread(target=self._run, args=(crop, capture_ts, generation), daemon=True).start()
        except Exception:
            with self._lock:
                self._busy = False
            return False
        return True

    def run_sync(self, frame_bgr: np.ndarray, capture_ts: float) -> SignalBoxes:
        crop = np.ascontiguousarray(frame_bgr[: max(1, int(frame_bgr.shape[0] * self.cfg.top_fraction))])
        return self._compute(crop, capture_ts)

    def _run(self, crop: np.ndarray, capture_ts: float, generation: int) -> None:
        try:
            res = self._compute(crop, capture_ts)
            with self._lock:
                if generation == self._generation and (self._latest is None or res.capture_ts >= self._latest.capture_ts):
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


# ---------------------------------------------------------------------------------------------------------------
# Cost-controlled controller used by Pipeline (docs/SAM_FALLBACK.md).
# ---------------------------------------------------------------------------------------------------------------
import os
from collections import deque

import cv2


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def sam_fallback_enabled_from_env() -> bool:
    """STREETSENSE_SAM_FALLBACK=0/1 forces it; unset = on only when MODEL_API_KEY is present."""
    raw = (os.environ.get("STREETSENSE_SAM_FALLBACK") or "").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    has_key = bool(os.environ.get("MODEL_API_KEY") or os.environ.get("LLAMA_API_KEY"))
    if raw in ("1", "true", "on", "yes"):
        return has_key
    return has_key


@dataclass
class SamControlConfig:
    enabled: bool = False
    confident_conf: float = 0.35  # a local light at/above this conf counts as "confident"
    crossing_radius_m: float = 45.0  # maps: crossing within this distance = crossing context
    check_hold_s: float = 20.0  # "can I cross?" keeps the crossing context alive this long
    context_grace_s: float = 5.0  # context must be gone this long before the crossing is closed
    stable_frames: int = 5  # readable smoothed signal for N consecutive frames = answered
    dup_thresh: float = 6.0  # mean abs diff (0-255) of 32x32 grey thumbnails; below = near-duplicate
    max_per_min: int = 6
    max_per_crossing: int = 3
    session_usd: float = 0.50
    max_age_s: float = 1.5
    min_interval_s: float = 1.0

    @classmethod
    def from_env(cls) -> "SamControlConfig":
        return cls(
            enabled=sam_fallback_enabled_from_env(),
            confident_conf=_env_float("STREETSENSE_SAM_CONFIDENT_CONF", 0.35),
            crossing_radius_m=_env_float("STREETSENSE_SAM_CROSSING_RADIUS_M", 45.0),
            check_hold_s=_env_float("STREETSENSE_SAM_CHECK_HOLD_S", 20.0),
            context_grace_s=_env_float("STREETSENSE_SAM_CONTEXT_GRACE_S", 5.0),
            stable_frames=int(_env_float("STREETSENSE_SAM_STABLE_FRAMES", 5)),
            dup_thresh=_env_float("STREETSENSE_SAM_DUP_THRESH", 6.0),
            max_per_min=int(_env_float("STREETSENSE_SAM_MAX_PER_MIN", 6)),
            max_per_crossing=int(_env_float("STREETSENSE_SAM_MAX_PER_CROSSING", 3)),
            session_usd=_env_float("STREETSENSE_SAM_SESSION_USD", 0.50),
            max_age_s=_env_float("STREETSENSE_SAM_MAX_AGE_S", 1.5),
            min_interval_s=_env_float("STREETSENSE_SAM_MIN_INTERVAL_S", 1.0),
        )


def _thumb(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


class SamFallbackController:
    """Decides when the SAM fallback may run and enforces every cost cap. Output = candidate boxes only.

    step() is called once per frame by Pipeline with the *previous* frame's local evidence. It returns
    fresh SAM signal-head boxes (frame pixels) for the local reader, or []. It never sees or returns a crossing
    decision; the go rule (local walk AND Muse yes) lives in answers.crossing_answer / fuse_crossing_decision.
    """

    def __init__(self, cfg: SamControlConfig | None = None, segment_fn: SegmentFn | None = None,
                 cost_per_call: float | None = None, budget_check: Callable[[float], None] | None = None,
                 sync: bool = False, sim_latency_s: float = 0.0, log: Callable[[str], None] = print):
        self.cfg = cfg or SamControlConfig.from_env()
        self._now = 0.0
        self.sync = sync  # simulation: run inline, deliver at capture_t + sim_latency_s (stream time)
        self.sim_latency_s = sim_latency_s
        self._log = log
        if cost_per_call is None:
            from streetsense.meta_usage import estimate_cost

            cost_per_call = estimate_cost("sam-3.1", images=1)
        self.cost_per_call = float(cost_per_call)
        if budget_check is None:
            def budget_check(x: float) -> None:
                from streetsense.meta_usage import get_ledger

                get_ledger().check_budget(x)
        self._budget_check = budget_check
        self.fb: SamSignalFallback | None = None
        self.disabled_reason: str | None = None if self.cfg.enabled else "off"
        if self.cfg.enabled:
            try:
                self.fb = SamSignalFallback(segment_fn, SamFallbackConfig(min_interval_s=self.cfg.min_interval_s,
                                                                           max_age_s=self.cfg.max_age_s),
                                            clock=lambda: self._now)
            except Exception as e:  # no key / client import failure: local detection only
                self._disable(f"init:{type(e).__name__}")
        # crossing state
        self.crossing_id = 0
        self.in_crossing = False
        self.answered = False
        self._ctx_last = -1e9
        self._check_until = -1e9
        self._maps_m: float | None = None
        self._stable = 0
        self._calls_crossing = 0
        self._call_ts: deque[float] = deque()
        self._last_thumb: np.ndarray | None = None
        self._last_res: SignalBoxes | None = None
        self._pending: list[tuple[float, SignalBoxes]] = []  # sync-sim delivery queue
        self._logged: set[str] = set()
        self.stats = {"calls": 0, "usd": 0.0, "dup_reused": 0, "cap_skips": {}, "crossings": [], "errors": 0,
                      "boxes_injected": 0}

    # -- context inputs -----------------------------------------------------------------------------------
    def note_crossing_check(self, t: float) -> None:
        self._check_until = t + self.cfg.check_hold_s

    def note_crossing_distance(self, meters: float | None) -> None:
        self._maps_m = meters

    def invalidate(self) -> None:
        """Drop old-scene evidence without resetting spend or rate-limit accounting."""
        if self.fb is not None:
            self.fb.invalidate()
        if self.in_crossing:
            self._end_crossing(self._now)
        self._last_thumb = self._last_res = None
        self._pending.clear()
        self._stable = 0
        self._check_until = self._ctx_last = -1e9
        self._maps_m = None

    def _ctx_active(self, t: float, crosswalk_ahead: bool) -> bool:
        maps = self._maps_m is not None and self._maps_m <= self.cfg.crossing_radius_m
        return t <= self._check_until or maps or bool(crosswalk_ahead)

    # -- helpers ------------------------------------------------------------------------------------------
    def _once(self, key: str, msg: str) -> None:
        if key not in self._logged:
            self._logged.add(key)
            self._log(f"[sam-fallback] {msg}")

    def _disable(self, reason: str) -> None:
        if self.disabled_reason is None:
            self.disabled_reason = reason
            self._once("disabled", f"disabled ({reason}); local detection only")

    def _skip(self, why: str) -> None:
        self.stats["cap_skips"][why] = self.stats["cap_skips"].get(why, 0) + 1
        self._once(f"{why}:{self.crossing_id}", f"skip: {why} (crossing {self.crossing_id})")

    def _start_crossing(self, t: float) -> None:
        self.crossing_id += 1
        self.in_crossing, self.answered, self._stable, self._calls_crossing = True, False, 0, 0
        self._last_thumb, self._last_res = None, None
        self.stats["crossings"].append({"id": self.crossing_id, "start_t": round(t, 2), "calls": 0, "usd": 0.0,
                                        "answered_t": None, "end_t": None})

    def _end_crossing(self, t: float) -> None:
        self.in_crossing = False
        self.stats["crossings"][-1]["end_t"] = round(t, 2)

    # -- main ---------------------------------------------------------------------------------------------
    def step(self, frame: np.ndarray, t: float, *, crosswalk_ahead: bool, local_conf: float,
             local_state: str) -> list[tuple[float, float, float, float, float]]:
        self._now = t
        if self._ctx_active(t, crosswalk_ahead):
            self._ctx_last = t
            if not self.in_crossing:
                self._start_crossing(t)
        elif self.in_crossing and t - self._ctx_last > self.cfg.context_grace_s:
            self._end_crossing(t)
        if not self.in_crossing:
            return []
        readable = local_conf >= self.cfg.confident_conf and local_state not in ("unknown", "unclear", "none", "")
        self._stable = self._stable + 1 if readable else 0
        if not self.answered and self._stable >= self.cfg.stable_frames:
            self.answered = True
            self.stats["crossings"][-1]["answered_t"] = round(t, 2)
        if self.disabled_reason is None and not self.answered and not readable:
            self._maybe_call(frame, t)
        return self._fresh_boxes(t)

    def _maybe_call(self, frame: np.ndarray, t: float) -> None:
        c = self.cfg
        while self._call_ts and t - self._call_ts[0] > 60.0:
            self._call_ts.popleft()
        if self.fb is None or self.fb._busy or (self._call_ts and t - self._call_ts[-1] < c.min_interval_s):
            return
        th = _thumb(frame)
        if self._last_thumb is not None and float(np.abs(th - self._last_thumb).mean()) < c.dup_thresh:
            if self._last_res is not None and self._last_res.error is None:
                self._last_res = SignalBoxes(capture_ts=t, boxes=self._last_res.boxes, latency_s=0.0)
                self.stats["dup_reused"] += 1
            return
        if self._calls_crossing >= c.max_per_crossing:
            return self._skip("per_crossing_cap")
        if len(self._call_ts) >= c.max_per_min:
            return self._skip("per_minute_cap")
        if self.stats["usd"] + self.cost_per_call > c.session_usd + 1e-12:
            return self._disable("session_usd_cap")
        try:
            self._budget_check(self.cost_per_call)
        except Exception:
            return self._disable("global_budget_cap")
        self._call_ts.append(t)
        self._calls_crossing += 1
        self._last_thumb = th
        self.stats["calls"] += 1
        self.stats["usd"] = round(self.stats["usd"] + self.cost_per_call, 6)
        cr = self.stats["crossings"][-1]
        cr["calls"] += 1
        cr["usd"] = round(cr["usd"] + self.cost_per_call, 6)
        if self.sync:
            res = self.fb.run_sync(frame, t)
            self._pending.append((t + self.sim_latency_s, res))
        else:
            self.fb.maybe_request(frame, t, have_confident_light=False)

    def _fresh_boxes(self, t: float):
        if self.sync:
            ready = [r for (at, r) in self._pending if at <= t]
            self._pending = [(at, r) for (at, r) in self._pending if at > t]
            for r in ready:
                if r.error:
                    self.stats["errors"] += 1
                self._last_res = r
        elif self.fb is not None:
            r = self.fb.latest(t)
            if r is not None and (self._last_res is None or r.capture_ts >= self._last_res.capture_ts):
                self._last_res = r
        r = self._last_res
        if r is None or not r.fresh(t, self.cfg.max_age_s):
            return []
        self.stats["boxes_injected"] += len(r.boxes)
        return list(r.boxes)

    def summary(self) -> dict:
        s = dict(self.stats)
        s["enabled"] = self.cfg.enabled
        s["disabled_reason"] = self.disabled_reason
        return s
