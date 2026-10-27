"""Per-class confidence overrides and default weights."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from streetsense.config import DEFAULT_CONF_BY_CLASS, DetectorConfig
from streetsense.detector import YoloDetector
from streetsense.types import Box, Detection


def test_default_weights_are_yolo11s():
    assert DetectorConfig().weights.endswith("yolo11s.pt")
    assert DetectorConfig().backend == "openvino"
    assert DEFAULT_CONF_BY_CLASS["person"] < DetectorConfig().conf


def test_accept_uses_per_class_floor(monkeypatch):
    monkeypatch.setattr("streetsense.detector.openvino_available", lambda: False)
    monkeypatch.setattr("streetsense.detector.ensure_pt_weights", lambda w: Path("models/yolo11s.pt"))

    class FakeYOLO:
        def __init__(self, path):
            self.names = {0: "person", 1: "car"}

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    d = YoloDetector(DetectorConfig(backend="torch", weights="models/yolo11s.pt", conf=0.35,
                                    conf_by_class={"person": 0.28, "car": 0.35}))
    assert d._accept("person", 0.30) is True
    assert d._accept("person", 0.20) is False
    assert d._accept("car", 0.30) is False
    assert d._accept("car", 0.40) is True
    assert d._conf_floor() == pytest.approx(0.28)
