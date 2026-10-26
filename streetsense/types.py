"""Core value types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Priority(IntEnum):
    INFO = 0  # scene descriptions, "sidewalk ahead"
    NOTICE = 1  # crossing ahead, light state
    WARNING = 2  # person / obstacle close in path
    CRITICAL = 3  # vehicle close, on the road with red light


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return self.w * self.h

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            max(0.0, min(self.x1, width - 1)),
            max(0.0, min(self.y1, height - 1)),
            max(0.0, min(self.x2, width)),
            max(0.0, min(self.y2, height)),
        )


@dataclass
class Detection:
    label: str  # normalised category, e.g. "person", "car", "traffic light"
    box: Box
    conf: float
    track_id: int | None = None
    attrs: dict = field(default_factory=dict)  # e.g. {"light_state": "red", "signal_kind": "pedestrian"}


@dataclass
class Alert:
    key: str  # dedupe key, e.g. "vehicle:center" or "light:pedestrian:red"
    text: str  # short spoken phrase
    priority: Priority
    ts: float = 0.0
    data: dict = field(default_factory=dict)
