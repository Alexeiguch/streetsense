"""Alert manager: decides what is worth saying, and when.

Rules:
- One utterance at a time; at most one alert every `global_min_gap_s` (CRITICAL may interrupt).
- The same *state* key is not repeated within its cooldown, unless it escalates in priority.
- Light alerts share a family pending slot (`light:{kind}`) so walk/red replace each other;
  cooldown is still per `light:{kind}:{state}` so real transitions are not suppressed.
- Pending alerts are ordered by priority, then recency; stale ones are dropped.
- `revalidate(facts)` drops pending speech that no longer matches current perception.
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
        cooldown_key = alert.data.get("cooldown_key") or alert.key
        last = self._last_spoken_key.get(cooldown_key)
        if last is not None:
            t, prio = last
            if now - t < self._cooldown(alert) and alert.priority <= prio:
                return  # said recently and not more urgent now
        family = alert.data.get("family")
        if family:
            # Replace any contradictory pending sibling in this family.
            for k in [k for k in self._pending if self._pending[k].data.get("family") == family]:
                if k != alert.key:
                    del self._pending[k]
        cur = self._pending.get(alert.key)
        if cur is None or alert.priority >= cur.priority or (alert.ts >= cur.ts):
            self._pending[alert.key] = alert
        if len(self._pending) > self.cfg.max_queue:
            worst = min(self._pending.values(), key=lambda a: (a.priority, a.ts))
            self._pending.pop(worst.key, None)

    def submit_light(
        self,
        *,
        kind: str,
        state: str,
        text: str,
        priority: Priority = Priority.NOTICE,
        data: dict | None = None,
    ) -> None:
        """Submit a light-state alert with family replacement + per-state cooldown."""
        family = f"light:{kind}"
        pending_key = family  # one pending slot per signal kind
        cooldown_key = f"{family}:{state}"
        payload = {"state": state, "kind": kind, "family": family, "cooldown_key": cooldown_key}
        if data:
            payload.update(data)
        self.submit(Alert(pending_key, text, priority, data=payload))

    def revalidate(self, facts) -> None:
        """Drop pending alerts that contradict current frame facts (esp. light state)."""
        from .rules import select_crossing_signal

        sel = select_crossing_signal(facts.detections, facts.width, facts.height, facts.crosswalk)
        current_state = None
        current_kind = None
        if sel is not None:
            current_state = getattr(sel, "decision", None) or sel.attrs.get("light_state")
            current_kind = sel.attrs.get("signal_kind") if hasattr(sel, "attrs") else getattr(sel, "kind", None)
            if hasattr(sel, "attrs"):
                current_kind = sel.attrs.get("signal_kind", current_kind)

        drop = []
        for k, a in self._pending.items():
            family = a.data.get("family") or ""
            if not family.startswith("light:"):
                continue
            pending_state = a.data.get("state")
            if current_state in (None, "unclear", "unknown", "countdown"):
                # Scene unclear / no light: don't speak an affirmative walk from the queue.
                if pending_state == "walk":
                    drop.append(k)
                continue
            if pending_state and pending_state != current_state:
                drop.append(k)
            # Affirmative walk must not linger when current is red/yellow/etc.
            if pending_state == "walk" and current_state != "walk":
                drop.append(k)
        for k in drop:
            self._pending.pop(k, None)

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
        cooldown_key = best.data.get("cooldown_key") or best.key
        self._last_spoken_key[cooldown_key] = (now, best.priority)
        return best

    def acknowledge_delivery(self, alert: Alert, delivered: bool) -> None:
        """If speech was dropped, clear the cooldown so we can retry."""
        if delivered:
            return
        cooldown_key = alert.data.get("cooldown_key") or alert.key
        self._last_spoken_key.pop(cooldown_key, None)

    @property
    def pending(self) -> list[Alert]:
        return sorted(self._pending.values(), key=lambda a: (-a.priority, -a.ts))
