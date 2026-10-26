"""Speech output. pyttsx3 (offline) when it works, otherwise print. Non-blocking."""

from __future__ import annotations

import queue
import threading


class Speaker:
    def __init__(self, enabled: bool = True, rate: int = 190):
        self.enabled = enabled
        self._q: queue.Queue[str] = queue.Queue(maxsize=4)
        self.backend = "print"
        if enabled:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", rate)
                self.backend = "pyttsx3"
                threading.Thread(target=self._loop, daemon=True).start()
            except Exception:
                self.backend = "print"

    def say(self, text: str) -> None:
        if self.backend == "print":
            print(f"SAY: {text}", flush=True)
            return
        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass  # the alert manager already prioritises; dropping beats speaking stale news

    def _loop(self) -> None:
        while True:
            t = self._q.get()
            self._engine.say(t)
            self._engine.runAndWait()
