"""The pedsig_ptld detector is opt-in only (STREETSENSE_DETECTOR_VARIANT); default unchanged."""
import pytest

from streetsense.config import Config
from streetsense.pipeline import DEFAULT_DETECTOR_WEIGHTS, DETECTOR_VARIANTS, apply_detector_variant


def test_default_detector_unchanged_without_env(monkeypatch):
    monkeypatch.delenv("STREETSENSE_DETECTOR_VARIANT", raising=False)
    cfg = apply_detector_variant(Config())
    assert cfg.detector.weights == DEFAULT_DETECTOR_WEIGHTS == Config().detector.weights


@pytest.mark.parametrize("v", ["", "default", "yolo11s"])
def test_default_aliases_keep_stock_weights(monkeypatch, v):
    monkeypatch.setenv("STREETSENSE_DETECTOR_VARIANT", v)
    assert apply_detector_variant(Config()).detector.weights == DEFAULT_DETECTOR_WEIGHTS


def test_pedsig_variant_is_opt_in(monkeypatch):
    monkeypatch.setenv("STREETSENSE_DETECTOR_VARIANT", "pedsig_ptld")
    cfg = apply_detector_variant(Config())
    assert cfg.detector.weights == DETECTOR_VARIANTS["pedsig_ptld"] != DEFAULT_DETECTOR_WEIGHTS


def test_explicit_weights_win_over_variant(monkeypatch):
    monkeypatch.setenv("STREETSENSE_DETECTOR_VARIANT", "pedsig_ptld")
    cfg = Config(); cfg.detector.weights = "models/yolo11n.pt"
    assert apply_detector_variant(cfg).detector.weights == "models/yolo11n.pt"


def test_unknown_variant_fails_loudly(monkeypatch):
    monkeypatch.setenv("STREETSENSE_DETECTOR_VARIANT", "nope")
    with pytest.raises(ValueError):
        apply_detector_variant(Config())
