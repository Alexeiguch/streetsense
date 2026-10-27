"""Detector backend selection, OpenVINO path helpers, and fallback behaviour."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from streetsense.config import DetectorConfig
from streetsense.detector import (
    ApproachTracker,
    normalize_backend,
    openvino_dir_for,
)
from streetsense.types import Box, Detection


def test_normalize_backend_aliases():
    assert normalize_backend("openvino") == "openvino"
    assert normalize_backend("OV") == "openvino"
    assert normalize_backend("torch") == "torch"
    assert normalize_backend("pytorch") == "torch"
    assert normalize_backend("pt") == "torch"
    with pytest.raises(ValueError):
        normalize_backend("tensorrt")


def test_openvino_dir_for_naming():
    assert openvino_dir_for("models/yolo11n.pt") == Path("models/yolo11n_openvino_model")
    assert openvino_dir_for(Path("/tmp/yolo26n.pt")) == Path("/tmp/yolo26n_openvino_model")


def test_default_config_backend_is_openvino():
    assert DetectorConfig().backend == "openvino"


def test_openvino_fallback_when_import_missing(monkeypatch):
    """If openvino isn't importable, YoloDetector must load PyTorch weights instead."""
    import streetsense.detector as det

    monkeypatch.setattr(det, "openvino_available", lambda: False)
    monkeypatch.setattr(det, "ensure_pt_weights", lambda w: Path("models/yolo11n.pt"))

    loads = []

    class FakeYOLO:
        def __init__(self, path):
            loads.append(path)
            self.names = {0: "person"}

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)

    d = det.YoloDetector(DetectorConfig(backend="openvino", weights="models/yolo11n.pt"))
    assert d.backend == "torch"
    assert loads and str(loads[-1]).endswith("yolo11n.pt")


def test_openvino_fallback_when_export_fails(monkeypatch):
    import streetsense.detector as det

    monkeypatch.setattr(det, "openvino_available", lambda: True)
    monkeypatch.setattr(det, "ensure_openvino_model", mock.Mock(side_effect=RuntimeError("export boom")))
    monkeypatch.setattr(det, "ensure_pt_weights", lambda w: Path("models/yolo11n.pt"))

    loads = []

    class FakeYOLO:
        def __init__(self, path):
            loads.append(path)
            self.names = {0: "person"}

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)

    d = det.YoloDetector(DetectorConfig(backend="openvino", weights="models/yolo11n.pt"))
    assert d.backend == "torch"
    assert any(str(p).endswith("yolo11n.pt") for p in loads)


def test_torch_backend_skips_openvino(monkeypatch):
    import streetsense.detector as det

    monkeypatch.setattr(det, "openvino_available", lambda: True)
    called = {"export": False}

    def boom(*a, **k):
        called["export"] = True
        raise AssertionError("should not export for torch backend")

    monkeypatch.setattr(det, "ensure_openvino_model", boom)
    monkeypatch.setattr(det, "ensure_pt_weights", lambda w: Path("models/yolo11n.pt"))

    class FakeYOLO:
        def __init__(self, path):
            self.path = path
            self.names = {0: "person"}

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    d = det.YoloDetector(DetectorConfig(backend="torch", weights="models/yolo11n.pt"))
    assert d.backend == "torch"
    assert called["export"] is False


def test_approach_tracker_flags_growing_box():
    """Regression: approaching-fast logic is independent of the inference backend."""
    tr = ApproachTracker(window_s=1.0, ratio=1.25)
    d0 = Detection("car", Box(0, 0, 40, 40), 0.9, track_id=7)
    d1 = Detection("car", Box(0, 0, 40, 52), 0.9, track_id=7)
    assert tr.update(0.0, [d0]) == set()
    assert 7 in tr.update(1.0, [d1])


@pytest.mark.skipif(
    not Path("models/yolo11n_openvino_model").is_dir() and not Path("models/yolo11n.pt").exists(),
    reason="no local yolo11n weights/export in this checkout",
)
def test_openvino_backend_loads_and_tracks_one_frame():
    """Smoke: real OpenVINO (or auto-export) returns detections with optional track ids."""
    import numpy as np

    from streetsense.detector import YoloDetector

    det = YoloDetector(DetectorConfig(backend="openvino", weights="models/yolo11n.pt", conf=0.25))
    assert det.backend == "openvino"
    # Blank frame: may be empty; still must not crash and track path must run.
    out = det(np.zeros((480, 640, 3), dtype=np.uint8))
    assert isinstance(out, list)
