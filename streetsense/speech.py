"""Speech output. Structured Alerts with priority eviction; pyttsx3 or print/queue backends."""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

from .types import Alert, Priority


class Speaker:
    """Non-blocking speaker.

    `say` accepts an Alert (preferred) or a plain string. Returns True when the utterance
    was accepted for delivery (or printed). False means it was dropped — callers must not
    treat that as spoken for cooldown purposes.
    """

    def __init__(
        self,
        enabled: bool = True,
        rate: int = 190,
        backend: str | None = None,
        maxsize: int = 4,
        max_age_s: float = 2.5,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.enabled = enabled
        self.maxsize = maxsize
        self.max_age_s = max_age_s
        self.clock = clock
        self._lock = threading.Lock()
        self._pending: list[Alert] = []
        self._interrupt = threading.Event()
        self.backend = backend or "print"
        self._engine = None
        if enabled and backend is None:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", rate)
                self.backend = "pyttsx3"
                threading.Thread(target=self._loop, daemon=True).start()
            except Exception:
                self.backend = "print"
        elif backend == "queue":
            self.backend = "queue"  # test / silent capture backend
        elif backend == "print":
            self.backend = "print"

    def pending_alerts(self) -> list[Alert]:
        with self._lock:
            self._expire()
            return list(self._pending)

    def _expire(self) -> None:
        now = self.clock()
        self._pending = [a for a in self._pending if now - (a.ts or now) <= self.max_age_s]

    def say(self, text_or_alert: str | Alert) -> bool:
        if isinstance(text_or_alert, Alert):
            alert = text_or_alert
            if not alert.ts:
                alert.ts = self.clock()
        else:
            alert = Alert("free", str(text_or_alert), Priority.NOTICE, ts=self.clock())

        if not self.enabled:
            return False

        if self.backend == "print":
            print(f"SAY: {alert.text}", flush=True)
            return True

        with self._lock:
            self._expire()
            # Critical may interrupt / evict lower priority.
            if alert.priority >= Priority.CRITICAL:
                self._pending = [a for a in self._pending if a.priority >= Priority.CRITICAL]
                self._interrupt.set()
                if self._engine is not None:
                    try:
                        self._engine.stop()
                    except Exception:
                        pass
            if len(self._pending) >= self.maxsize:
                # Evict lowest priority, oldest; refuse if new is not better.
                worst_i = min(range(len(self._pending)), key=lambda i: (self._pending[i].priority, self._pending[i].ts))
                worst = self._pending[worst_i]
                if alert.priority > worst.priority:
                    del self._pending[worst_i]
                else:
                    return False
            self._pending.append(alert)
            self._pending.sort(key=lambda a: (-a.priority, a.ts))
            return True

    def _loop(self) -> None:
        while True:
            alert = None
            with self._lock:
                self._expire()
                if self._pending:
                    alert = self._pending.pop(0)
                    self._interrupt.clear()
            if alert is None:
                time.sleep(0.05)
                continue
            try:
                self._engine.say(alert.text)
                self._engine.runAndWait()
            except Exception:
                print(f"SAY: {alert.text}", flush=True)
