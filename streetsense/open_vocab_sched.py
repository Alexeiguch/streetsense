"""Scheduling for the optional open-vocab hazard detector (slow path)."""

from __future__ import annotations

from typing import Any


class OpenVocabScheduler:
    """Distinguish never-run / empty-success / error; back off on failures; expire cache.

    Empty successful results do NOT force a re-run next frame — only the interval does.
    """

    def __init__(
        self,
        every_n_frames: int = 15,
        error_backoff_frames: int = 30,
        cache_ttl_frames: int = 45,
    ):
        self.every_n_frames = max(1, every_n_frames)
        self.error_backoff_frames = max(1, error_backoff_frames)
        self.cache_ttl_frames = cache_ttl_frames
        self._initialized = False
        self._last_attempt_frame: int | None = None
        self._last_success_frame: int | None = None
        self._backoff_until: int = 0
        self._cached: list[Any] = []

    def should_run(self, frame_i: int) -> bool:
        if frame_i < self._backoff_until:
            return False
        if not self._initialized or self._last_attempt_frame is None:
            return True
        return (frame_i - self._last_attempt_frame) >= self.every_n_frames

    def record_result(self, dets: list[Any], *, failed: bool = False, frame_i: int | None = None) -> None:
        if frame_i is not None:
            self._last_attempt_frame = frame_i
        elif self._last_attempt_frame is None:
            self._last_attempt_frame = 0
        self._initialized = True
        if failed:
            self._backoff_until = self._last_attempt_frame + self.error_backoff_frames
            return
        self._last_success_frame = self._last_attempt_frame
        self._cached = list(dets)

    def mark_attempt(self, frame_i: int) -> None:
        self._last_attempt_frame = frame_i
        self._initialized = True

    def cached(self, frame_i: int) -> list[Any]:
        if self._last_success_frame is None:
            return []
        if frame_i - self._last_success_frame > self.cache_ttl_frames:
            return []
        return list(self._cached)
