"""Short rolling buffer of recent frames + local light states for multi-frame Muse checks.

Muse Spark only sees what we send. A single still can land on an LED "off" phase of a
walk signal (seen on the DC clip), so the crossing check can send one or two older
frames with the current one. This buffer never decides anything; fusion still needs
local walk on the current frame AND Muse yes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HistoryEntry:
    t: float
    frame: np.ndarray
    local_state: str


class FrameHistory:
    def __init__(self, max_age_s: float = 3.0, min_gap_s: float = 0.1, max_side: int = 512):
        self.max_age_s = max_age_s
        self.min_gap_s = min_gap_s
        self.max_side = max_side
        self._buf: deque[HistoryEntry] = deque()

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def add(self, t: float, frame: np.ndarray, local_state: str) -> None:
        if self._buf and t - self._buf[-1].t < self.min_gap_s:
            return
        h, w = frame.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame.copy()
        self._buf.append(HistoryEntry(t, small, str(local_state or "unknown")))
        while self._buf and t - self._buf[0].t > self.max_age_s:
            self._buf.popleft()

    def pick(self, now: float, n: int, spacing_s: float) -> list[tuple[float, np.ndarray]]:
        """Up to n older frames near now - k*spacing (k = n..1), as [(age_s, frame)], oldest first.

        A target with no frame within spacing/2 is skipped, so a gap just sends fewer frames.
        Frames newer than spacing/2 (including the current one) are never picked.
        """
        if n <= 0 or spacing_s <= 0:
            return []
        out: list[tuple[float, np.ndarray]] = []
        used: set[int] = set()
        for k in range(n, 0, -1):
            target = now - k * spacing_s
            best, best_d = None, spacing_s / 2
            for i, e in enumerate(self._buf):
                age = now - e.t
                if i in used or age < spacing_s / 2 or age < 0:
                    continue
                d = abs(e.t - target)
                if d <= best_d:
                    best, best_d = i, d
            if best is not None:
                used.add(best)
                e = self._buf[best]
                out.append((round(now - e.t, 2), e.frame))
        return out

    def states(self, now: float, window_s: float = 2.0) -> list[tuple[float, str]]:
        return [(round(now - e.t, 2), e.local_state) for e in self._buf if 0 <= now - e.t <= window_s]


def history_summary_text(states: list[tuple[float, str]]) -> str | None:
    """One line of local reader history for the prompt, oldest first. None when empty."""
    if not states:
        return None
    parts = ", ".join(f"{age:.1f}s ago: {st}" for age, st in sorted(states, key=lambda x: -x[0]))
    return (
        "On-device light reader history (may be wrong; judge the images yourself): " + parts
    )
