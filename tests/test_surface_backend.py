"""Surface segmenter backend selection, OpenVINO helpers, and fallback."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from streetsense.config import SurfaceConfig
from streetsense.surface import (
    DEFAULT_MODEL_ID,
    DEFAULT_OV_DIR,
    SurfaceSegmenter,
    normalize_backend,
    openvino_available,
    ov_cache_dir,
    summarise_labels,
)
from streetsense.surface import ROAD, SIDEWALK


def test_normalize_backend_aliases():
    assert normalize_backend("openvino") == "openvino"
    assert normalize_backend("OV") == "openvino"
    assert normalize_backend("torch") == "torch"
    assert normalize_backend("pytorch") == "torch"
    with pytest.raises(ValueError):
        normalize_backend("onnx")


def test_ov_cache_dir_default_and_slug():
    assert ov_cache_dir(DEFAULT_MODEL_ID) == DEFAULT_OV_DIR
    assert ov_cache_dir("org/other-model") == Path("models/org_other-model_openvino")
    assert ov_cache_dir(DEFAULT_MODEL_ID, "tmp/ov") == Path("tmp/ov")


def test_default_surface_config_backend_is_openvino():
    assert SurfaceConfig().backend == "openvino"


def test_summarise_labels_underfoot_road():
    labels = np.full((100, 100), SIDEWALK, dtype=np.uint8)
    labels[80:, 35:65] = ROAD
    r = summarise_labels(labels)
    assert r.underfoot == "road"


def test_openvino_fallback_when_unavailable(monkeypatch):
    import streetsense.surface as surf

    monkeypatch.setattr(surf, "openvino_available", lambda: False)
    monkeypatch.setattr(surf, "_load_torch_segformer", lambda *a, **k: (_Torch(), _Model()))

    class _Torch:
        def from_numpy(self, x):
            return x

        class inference_mode:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

    class _Model:
        def __call__(self, pixel_values=None):
            class R:
                logits = type("T", (), {"detach": lambda self: type("T2", (), {"cpu": lambda self: type("T3", (), {"numpy": lambda self: np.zeros((1, 19, 8, 16), np.float32)})()})()})()

            return R()

    # Patch _init_torch path more simply
    def fake_init_torch(self):
        self._torch = mock.Mock()
        self._torch_model = mock.Mock()

    monkeypatch.setattr(SurfaceSegmenter, "_init_torch", fake_init_torch)
    s = SurfaceSegmenter(backend="openvino")
    assert s.backend == "torch"


def test_openvino_fallback_when_export_fails(monkeypatch):
    import streetsense.surface as surf

    monkeypatch.setattr(surf, "openvino_available", lambda: True)
    monkeypatch.setattr(surf, "ensure_openvino_segformer", mock.Mock(side_effect=RuntimeError("boom")))

    def fake_init_torch(self):
        self._torch = mock.Mock()
        self._torch_model = mock.Mock()

    monkeypatch.setattr(SurfaceSegmenter, "_init_torch", fake_init_torch)
    s = SurfaceSegmenter(backend="openvino")
    assert s.backend == "torch"


def test_torch_backend_skips_openvino_export(monkeypatch):
    import streetsense.surface as surf

    monkeypatch.setattr(surf, "openvino_available", lambda: True)
    called = {"export": False}

    def boom(*a, **k):
        called["export"] = True
        raise AssertionError("should not export")

    monkeypatch.setattr(surf, "ensure_openvino_segformer", boom)

    def fake_init_torch(self):
        self._torch = mock.Mock()
        self._torch_model = mock.Mock()

    monkeypatch.setattr(SurfaceSegmenter, "_init_torch", fake_init_torch)
    s = SurfaceSegmenter(backend="torch")
    assert s.backend == "torch"
    assert called["export"] is False


@pytest.mark.skipif(
    not openvino_available() or not (DEFAULT_OV_DIR / "segformer.xml").is_file(),
    reason="OpenVINO SegFormer cache not present",
)
def test_openvino_surface_smoke_one_frame():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    s = SurfaceSegmenter(backend="openvino", input_width=512)
    assert s.backend == "openvino"
    r = s(frame)
    assert r.labels.shape == (360, 640)
    assert r.underfoot in ("road", "sidewalk", "terrain", "unknown")
