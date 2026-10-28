"""Ray-Ban Meta "glasses mode": stream profile, frame adapter and short speech cues.

This mirrors the public Meta Wearables Device Access Toolkit (DAT 1.0) camera
stream so StreetSense can be built and tested against glasses-shaped input
before any hardware is in hand. The toolkit itself is native (iOS / Android)
and runs in a phone companion app; this module is the Python side that app
would feed. Nothing here has run on real glasses yet (see docs/RAYBAN.md).

DAT facts mirrored here (from the toolkit docs, Oct 2026):
  * resolutions: high 720x1280, medium 504x896, low 360x640 (portrait)
  * frame rates: 2, 7, 15, 24, 30 fps
  * over Bluetooth Classic the toolkit lowers resolution first, then frame
    rate, and does not go below 15 fps on its own
  * frames are compressed per frame, so quality is lower than a phone camera
  * stream states: stopping, stopped, waitingForDevice, starting, streaming, paused
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from enum import Enum

import cv2
import numpy as np

from .types import Alert, Priority

# (width, height) portrait, as delivered by the toolkit.
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "low": (360, 640),
    "medium": (504, 896),
    "high": (720, 1280),
}
FRAME_RATES: tuple[int, ...] = (2, 7, 15, 24, 30)
_RES_ORDER = ("high", "medium", "low")
AUTO_LADDER_MIN_FPS = 15


class StreamState(str, Enum):
    """Same names as the toolkit's StreamSessionState (iOS) / StreamState (Android)."""

    STOPPING = "stopping"
    STOPPED = "stopped"
    WAITING_FOR_DEVICE = "waitingForDevice"
    STARTING = "starting"
    STREAMING = "streaming"
    PAUSED = "paused"


@dataclass(frozen=True)
class GlassesStreamConfig:
    """Mirror of the toolkit StreamConfiguration (resolution + frameRate) plus our JPEG hop."""

    resolution: str = "medium"
    frame_rate: int = 15
    jpeg_quality: int = 60  # phone app -> StreetSense hop; also mimics toolkit compression
    # Horizontal field of view of the portrait frame. Placeholder until measured on a device.
    hfov_deg: float = 60.0

    def __post_init__(self):
        if self.resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {sorted(RESOLUTIONS)}")
        if self.frame_rate not in FRAME_RATES:
            raise ValueError(f"frame_rate must be one of {FRAME_RATES}")
        if not 10 <= int(self.jpeg_quality) <= 95:
            raise ValueError("jpeg_quality must be 10..95")

    @property
    def size(self) -> tuple[int, int]:
        return RESOLUTIONS[self.resolution]

    @property
    def frame_interval_s(self) -> float:
        return 1.0 / float(self.frame_rate)

    def to_dict(self) -> dict:
        w, h = self.size
        return {"resolution": self.resolution, "width": w, "height": h, "frame_rate": self.frame_rate,
                "jpeg_quality": self.jpeg_quality, "hfov_deg": self.hfov_deg}


def auto_ladder(cfg: GlassesStreamConfig) -> list[GlassesStreamConfig]:
    """Steps the toolkit may take when Bluetooth bandwidth drops: resolution first, then fps >= 15."""
    steps = [cfg]
    cur = cfg
    for res in _RES_ORDER[_RES_ORDER.index(cfg.resolution) + 1:]:
        cur = replace(cur, resolution=res)
        steps.append(cur)
    for fps in sorted((f for f in FRAME_RATES if AUTO_LADDER_MIN_FPS <= f < cur.frame_rate), reverse=True):
        cur = replace(cur, frame_rate=fps)
        steps.append(cur)
    return steps


def adapt_frame(frame_bgr: np.ndarray, cfg: GlassesStreamConfig) -> np.ndarray:
    """Make any camera or video frame look like a toolkit frame.

    Centre-crop to 9:16 portrait, resize to the stream resolution and pass it
    through JPEG at the configured quality.
    """
    W, H = cfg.size
    h, w = frame_bgr.shape[:2]
    target = W / float(H)
    if w / float(h) > target:  # too wide: crop width
        cw = max(1, int(round(h * target)))
        x0 = (w - cw) // 2
        crop = frame_bgr[:, x0:x0 + cw]
    else:  # too tall: crop height
        ch = max(1, int(round(w / target)))
        y0 = (h - ch) // 2
        crop = frame_bgr[y0:y0 + ch, :]
    out = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(cfg.jpeg_quality)])
    if not ok:
        return out
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


class FramePacer:
    """Keep only the newest frame and drop stale ones.

    The toolkit pushes frames at a fixed rate whether or not we keep up. If
    processing is slower than the stream, older frames are skipped rather than
    queued, so alerts always describe the latest view.
    """

    def __init__(self, cfg: GlassesStreamConfig, max_age_s: float = 0.5, clock=time.monotonic):
        self.cfg = cfg
        self.max_age_s = max_age_s
        self.clock = clock
        self._latest: tuple[float, np.ndarray] | None = None
        self.received = 0
        self.dropped = 0

    def push(self, frame: np.ndarray, ts: float | None = None) -> None:
        self.received += 1
        if self._latest is not None:
            self.dropped += 1  # overwritten before anyone took it
        self._latest = (self.clock() if ts is None else ts, frame)

    def take(self) -> np.ndarray | None:
        if self._latest is None:
            return None
        ts, frame = self._latest
        self._latest = None
        if self.clock() - ts > self.max_age_s:
            self.dropped += 1
            return None
        return frame


# Speech cues ------------------------------------------------------------------------------
# Earcons are short tones the web page or phone app plays before (or instead of) speech, so
# the most urgent cue lands in well under a second through the glasses speakers.
EARCONS: dict[str, dict] = {
    "alarm": {"freq_hz": 880, "ms": 180, "repeat": 3},  # vehicle close: stop
    "warn": {"freq_hz": 660, "ms": 120, "repeat": 2},  # person / obstacle in path
    "walk": {"freq_hz": 1046, "ms": 90, "repeat": 2},  # local reader sees a walk signal
    "wait": {"freq_hz": 330, "ms": 250, "repeat": 1},  # red / amber / unclear signal
    "notice": {"freq_hz": 520, "ms": 80, "repeat": 1},  # crossing found, info
}

_SHORTEN = [
    (re.compile(r"\bon your (left|right)\b"), r"\1"),
    (re.compile(r"^careful,\s*"), ""),
    (re.compile(r"\bpedestrian crossing\b"), "crossing"),
    (re.compile(r"\bapproaching\b"), "coming"),
    (re.compile(r"\babout\s+"), ""),
]


def glasses_cue(alert: Alert | None, *, max_words: int = 5) -> dict | None:
    """Short phrase + earcon for one alert, or None when glasses mode stays quiet.

    Light and crossing texts are kept exactly: their wording is part of the safety contract.
    INFO alerts (scene notes, Muse hazard phrases) are not spoken in glasses mode.
    """
    if alert is None or alert.priority <= Priority.INFO:
        return None
    key = alert.key
    state = str((alert.data or {}).get("state", ""))
    if key.startswith("light:"):
        earcon = "walk" if state == "walk" else "wait"
        return {"text": alert.text, "earcon": earcon, "priority": int(alert.priority), "key": key}
    text = alert.text
    for pat, rep in _SHORTEN:
        text = pat.sub(rep, text)
    words = text.replace(",", " ,").split()
    text = " ".join(words[:max_words]).replace(" ,", ",").strip(" ,")
    if alert.priority >= Priority.CRITICAL:
        earcon = "alarm"
    elif alert.priority >= Priority.WARNING:
        earcon = "warn"
    else:
        earcon = "notice"
    return {"text": text, "earcon": earcon, "priority": int(alert.priority), "key": key}
