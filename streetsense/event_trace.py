"""Per-frame event trace (JSONL) for demo evaluation and replay."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO


class EventTrace:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._fh: TextIO | None = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def log(self, **event: Any) -> None:
        event.setdefault("ts", time.time())
        line = json.dumps(event, default=str)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def replay_summary(path: str | Path) -> dict:
    """Report critical alerts/min, wrong walk announcements, transition delays from a JSONL trace."""
    critical = 0
    wrong_walk = 0
    transitions: list[float] = []
    t0 = t1 = None
    n = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        n += 1
        ts = ev.get("ts") or ev.get("capture_ts")
        if t0 is None:
            t0 = ts
        t1 = ts
        if ev.get("spoken_priority") == "CRITICAL" or ev.get("priority") == "CRITICAL":
            critical += 1
        if ev.get("wrong_walk"):
            wrong_walk += 1
        if ev.get("transition_delay_s") is not None:
            transitions.append(float(ev["transition_delay_s"]))
    dur_min = max(1e-6, ((t1 or 0) - (t0 or 0)) / 60.0) if t0 is not None else 1.0
    return {
        "events": n,
        "critical_per_min": critical / dur_min,
        "wrong_walk_announcements": wrong_walk,
        "transition_delay_mean_s": (sum(transitions) / len(transitions)) if transitions else None,
    }
