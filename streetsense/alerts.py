"""Alert manager: decides what is worth saying, and when.

Rules:
- One utterance at a time; at most one alert every `global_min_gap_s` (CRITICAL may interrupt).
- The same key is not repeated within its cooldown, unless it escalates in priority.
- Pending alerts are ordered by priority, then recency; stale ones are dropped.
- A key's text can change while still deduped (e.g. "person ahead, 3 metres" -> "2 metres")
  and the newer text wins if it is still pending.
"""

from __future__ import annotations

import time
from typing import Callable

from .config import AlertConfig
from .types import Alert, Priority


class AlertManager:
    def __init__(self, cfg: AlertConfig | None = None, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg or AlertConfig()
        self.clock = clock
        self._pending: dict[str, Alert] = {}
        self._last_spoken_key: dict[str, tuple[float, Priority]] = {}
        self._last_emit_ts: float = -1e9
        self.max_age_s = 2.0  # a pending alert older than this no longer describes the scene

    def _cooldown(self, a: Alert) -> float:
        return self.cfg.critical_cooldown_s if a.priority >= Priority.CRITICAL else self.cfg.default_cooldown_s

    def submit(self, alert: Alert) -> None:
        now = self.clock()
        alert.ts = now
        last = self._last_spoken_key.get(alert.key)
        if last is not None:
            t, prio = last
            if now - t < self._cooldown(alert) and alert.priority <= prio:
                return  # said recently and not more urgent now
        cur = self._pending.get(alert.key)
        if cur is None or alert.priority >= cur.priority:
            self._pending[alert.key] = alert
        if len(self._pending) > self.cfg.max_queue:
            worst = min(self._pending.values(), key=lambda a: (a.priority, a.ts))
            self._pending.pop(worst.key, None)

    def pop_ready(self) -> Alert | None:
        """Return the single alert to speak now, or None."""
        now = self.clock()
        for k in [k for k, a in self._pending.items() if now - a.ts > self.max_age_s]:
            del self._pending[k]
        if not self._pending:
            return None
        best = max(self._pending.values(), key=lambda a: (a.priority, a.ts))
        gap_ok = now - self._last_emit_ts >= self.cfg.global_min_gap_s
        if not gap_ok and best.priority < Priority.CRITICAL:
            return None
        del self._pending[best.key]
        self._last_emit_ts = now
        self._last_spoken_key[best.key] = (now, best.priority)
        return best

    @property
    def pending(self) -> list[Alert]:
        return sorted(self._pending.values(), key=lambda a: (-a.priority, -a.ts))
