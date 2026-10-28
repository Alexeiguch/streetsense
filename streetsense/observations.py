"""A small rolling log of what the fast loop saw, in a form an agent can read.

Each entry is a compact, JSON-safe summary of one processed frame. The log is
cheap (no model calls) and feeds the opt-in Muse Wonder guide, which can then
reason over the last few seconds instead of a single still.
"""

from __future__ import annotations

import threading
from collections import deque

from .geometry import horizontal_zone, in_walking_path
from .rules import select_crossing_signal

MAX_OBJECTS = 12


def summarise_frame(out, width: int, height: int) -> dict:
    """Compact facts for one FrameOutput. Pure function, unit-testable."""
    objs = []
    for d in out.detections:
        if d.label == "traffic light":
            continue
        dist = d.attrs.get("distance_m")
        objs.append({
            "label": d.label,
            "zone": horizontal_zone(d.box, width),
            "in_path": bool(in_walking_path(d.box, width, height)),
            "dist_m": round(float(dist), 1) if isinstance(dist, (int, float)) else None,
            "track": d.track_id,
        })
    objs.sort(key=lambda o: (o["dist_m"] is None, o["dist_m"] or 0.0))
    sel = select_crossing_signal(out.detections, width, height, out.crosswalk)
    cw = out.crosswalk
    surface = out.surface
    return {
        "t": round(float(out.t), 2),
        "objects": objs[:MAX_OBJECTS],
        "signal": sel.decision if sel is not None else "unknown",
        "crosswalk": (
            {"found": True, "zone": horizontal_zone(cw.box, width)}
            if cw is not None and cw.found and cw.box is not None else {"found": False}
        ),
        "underfoot": surface.underfoot if surface is not None else "unknown",
        "spoken": out.spoken.text if out.spoken is not None else None,
    }


class ObservationLog:
    """Thread-safe ring buffer of frame summaries."""

    def __init__(self, maxlen: int = 300):
        self._buf: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, summary: dict) -> None:
        with self._lock:
            self._buf.append(summary)

    def __len__(self) -> int:
        return len(self._buf)

    def latest(self) -> dict | None:
        with self._lock:
            return dict(self._buf[-1]) if self._buf else None

    def window(self, seconds: float, max_entries: int = 15) -> list[dict]:
        """Entries from the last `seconds`, thinned to at most `max_entries` (newest kept)."""
        with self._lock:
            items = list(self._buf)
        if not items:
            return []
        t_end = items[-1]["t"]
        recent = [e for e in items if t_end - e["t"] <= max(0.0, float(seconds))]
        if len(recent) <= max_entries:
            return recent
        step = len(recent) / float(max_entries)
        picked = [recent[int(i * step)] for i in range(max_entries)]
        picked[-1] = recent[-1]
        return picked
