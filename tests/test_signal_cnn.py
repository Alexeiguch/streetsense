"""Crop classifier (streetsense/signal_cnn.py) and how it combines with the hand rules.

Safety: the classifier is one input to the local reader. It may veto a walk at any time,
and may only promote a green/unknown pedestrian head (never red, yellow, countdown or a
vehicle head) when p_walk clears the calibrated threshold.
"""
import numpy as np
import pytest

from streetsense import signal_cnn
from streetsense.traffic_light import LightReading, apply_signal_cnn

T = 0.95


def rd(state, countdown=False):
    return LightReading(state=state, confidence=0.8, lit_fraction=0.1, lamp_position=0.5, countdown=countdown)


def P(w, s=0.0, c=0.0, o=None):
    o = 1.0 - w - s - c if o is None else o
    return (w, s, c, o)


def test_no_probs_is_passthrough():
    r = rd("walk")
    assert apply_signal_cnn(r, None, "pedestrian", T) is r


def test_walk_kept_when_classifier_agrees():
    assert apply_signal_cnn(rd("walk"), P(0.9, 0.05), "pedestrian", T).state == "walk"


@pytest.mark.parametrize("probs,expect", [
    (P(0.1, 0.85), "unknown"),           # classifier sees stop
    (P(0.1, 0.0, 0.85), "countdown"),    # classifier sees digits -> rules turn it into unclear
    (P(0.1, 0.0, 0.0, 0.9), "unknown"),  # off / unclear
    (P(0.45, 0.3, 0.1), "unknown"),      # not confident in walk
])
def test_veto_walk(probs, expect):
    out = apply_signal_cnn(rd("walk"), probs, "pedestrian", T)
    assert out.state == expect
    assert out.state != "walk"


def test_countdown_veto_sets_flag():
    assert apply_signal_cnn(rd("walk"), P(0.05, 0.0, 0.9), "pedestrian", T).countdown


@pytest.mark.parametrize("state", ["green", "unknown"])
def test_promote_only_above_threshold(state):
    up = apply_signal_cnn(rd(state), P(0.97), "pedestrian", T)
    assert up.state == "walk" and up.figure
    assert apply_signal_cnn(rd(state), P(0.94), "pedestrian", T).state == state


@pytest.mark.parametrize("state", ["red", "yellow", "countdown"])
def test_never_promote_red_yellow_countdown(state):
    assert apply_signal_cnn(rd(state), P(0.999), "pedestrian", T).state == state


def test_never_promote_countdown_flag_or_vehicle():
    assert apply_signal_cnn(rd("green", countdown=True), P(0.999), "pedestrian", T).state == "green"
    assert apply_signal_cnn(rd("green"), P(0.999), "vehicle", T).state == "green"


def test_env_switch(monkeypatch):
    monkeypatch.setenv("STREETSENSE_SIGNAL_CNN", "0")
    assert not signal_cnn.enabled()
    assert signal_cnn.get() is None
    monkeypatch.setenv("STREETSENSE_SIGNAL_CNN", "1")
    assert signal_cnn.enabled()


def test_preprocess_letterbox():
    x = signal_cnn.preprocess(np.full((90, 30, 3), 255, np.uint8))
    assert x.shape == (3, signal_cnn.SIZE, signal_cnn.SIZE)
    assert x.dtype == np.float32
    assert x.max() == pytest.approx(0.5) and x.min() == pytest.approx(-0.5)  # white figure, black bars


def test_shipped_model_loads_and_is_calibrated():
    if not (signal_cnn.MODEL_DIR / "signal_cnn.onnx").is_file():
        pytest.skip("model not present")
    m = signal_cnn.SignalCNN()
    assert 0.5 <= m.t_walk <= 0.99
    p = m(np.zeros((60, 30, 3), np.uint8))
    assert len(p.probs) == 4 and abs(sum(p.probs) - 1) < 1e-3
    assert p.p_walk < m.t_walk  # a dark housing is never walk


def test_hold_veto_blocks_smoothed_walk():
    from streetsense.traffic_light import cnn_hold_veto
    assert cnn_hold_veto("walk", "countdown", P(0.05, 0.0, 0.9), 0.8) == "countdown"
    assert cnn_hold_veto("walk", "unknown", P(0.05, 0.9), 0.8) == "unknown"
    assert cnn_hold_veto("walk", "walk", P(0.9), 0.8) == "walk"
    assert cnn_hold_veto("walk", "countdown", None, 0.8) == "walk"
    for st in ("red", "unknown", "countdown", "green"):  # never creates walk
        assert cnn_hold_veto(st, "walk", P(0.99), 0.8) == st


def test_pipeline_smoother_cannot_hold_walk_through_cnn_countdown():
    from streetsense.traffic_light import LightStateSmoother, cnn_hold_veto
    sm = LightStateSmoother(clock=lambda: 0.0)
    for _ in range(4):
        sm.update(1, "walk")
    held = sm.update(1, "countdown")
    assert held == "walk"  # existing flicker hold
    assert cnn_hold_veto(held, "countdown", P(0.02, 0.0, 0.95), 0.8) != "walk"


def test_version_env_selects_model_dir(monkeypatch):
    monkeypatch.delenv("STREETSENSE_SIGNAL_CNN_VERSION", raising=False)
    assert signal_cnn.selected_model_dir() == signal_cnn.MODELS / ("signal_cnn" if signal_cnn.DEFAULT_VERSION == "v1" else f"signal_cnn_{signal_cnn.DEFAULT_VERSION}")
    monkeypatch.setenv("STREETSENSE_SIGNAL_CNN_VERSION", "v1")
    assert signal_cnn.selected_model_dir() == signal_cnn.MODEL_DIR
    monkeypatch.setenv("STREETSENSE_SIGNAL_CNN_VERSION", "2")
    assert signal_cnn.selected_model_dir() == signal_cnn.MODELS / "signal_cnn_v2"
