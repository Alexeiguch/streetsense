"""Opt-in yellow promote (STREETSENSE_READER_YELLOW_PROMOTE): a confident classifier walk may override an
HSV 'yellow' call on a non-vehicle head (yellow-painted housings win the hue vote over a lit walk figure).
Default off; red, countdown and vehicle heads are never promoted; the threshold is unchanged."""
import cv2
import numpy as np
import pytest

from streetsense.traffic_light import LightReading, apply_signal_cnn, classify_light, signal_kind, yellow_promote_enabled

T = 0.95
ENV = "STREETSENSE_READER_YELLOW_PROMOTE"


def rd(state, countdown=False):
    return LightReading(state=state, confidence=0.8, lit_fraction=0.1, lamp_position=0.5, countdown=countdown)


def P(w, s=0.0, c=0.0):
    return (w, s, c, 1.0 - w - s - c)


def test_default_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert not yellow_promote_enabled()
    assert apply_signal_cnn(rd("yellow"), P(0.999), "pedestrian", T).state == "yellow"


@pytest.mark.parametrize("val,on", [("1", True), ("true", True), ("on", True), ("0", False), ("", False), ("no", False)])
def test_env_parsing(monkeypatch, val, on):
    monkeypatch.setenv(ENV, val)
    assert yellow_promote_enabled() is on


@pytest.mark.parametrize("kind", ["pedestrian", "unknown"])
def test_promotes_yellow_only_at_threshold(monkeypatch, kind):
    monkeypatch.setenv(ENV, "1")
    up = apply_signal_cnn(rd("yellow"), P(0.96), kind, T)
    assert up.state == "walk" and up.figure
    assert apply_signal_cnn(rd("yellow"), P(0.949), kind, T).state == "yellow"


def test_explicit_argument_overrides_env(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert apply_signal_cnn(rd("yellow"), P(0.99), "pedestrian", T, promote_yellow=False).state == "yellow"
    monkeypatch.delenv(ENV)
    assert apply_signal_cnn(rd("yellow"), P(0.99), "pedestrian", T, promote_yellow=True).state == "walk"


@pytest.mark.parametrize("state,countdown,kind", [
    ("red", False, "pedestrian"),        # red is never overridden to walk
    ("countdown", True, "pedestrian"),   # countdown stays countdown
    ("yellow", True, "pedestrian"),      # yellow with a countdown flag
    ("yellow", False, "vehicle"),        # vehicle amber
])
def test_never_promotes_red_countdown_or_vehicle(monkeypatch, state, countdown, kind):
    monkeypatch.setenv(ENV, "1")
    out = apply_signal_cnn(rd(state, countdown), P(0.999), kind, T)
    assert out.state == state and out.state != "walk"


def test_veto_unchanged_with_flag(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert apply_signal_cnn(rd("walk"), P(0.1, 0.85), "pedestrian", T).state != "walk"


def test_yellow_housing_crop_needs_flag(monkeypatch):
    """Synthetic US-style head: yellow-painted housing around a lit green walk figure reads HSV 'yellow'."""
    c = np.full((100, 70, 3), (20, 20, 20), np.uint8)
    c[5:95, 5:65] = (0, 190, 230)
    c[55:90, 15:55] = (30, 40, 30)
    cv2.ellipse(c, (35, 72), (6, 13), 0, 0, 360, (170, 230, 40), -1)
    kind = signal_kind(70, 100)
    r = classify_light(c, kind)
    assert r.state == "yellow" and kind != "vehicle"
    monkeypatch.delenv(ENV, raising=False)
    assert apply_signal_cnn(r, P(0.99), kind, T).state == "yellow"
    monkeypatch.setenv(ENV, "1")
    assert apply_signal_cnn(r, P(0.99), kind, T).state == "walk"
    assert apply_signal_cnn(r, P(0.90), kind, T).state == "yellow"
