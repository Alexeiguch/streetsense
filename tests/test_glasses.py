"""Ray-Ban glasses mode: toolkit-shaped stream profile, adapter and cues (no hardware)."""

from __future__ import annotations

import numpy as np
import pytest

from streetsense.glasses import (
    EARCONS,
    FRAME_RATES,
    RESOLUTIONS,
    FramePacer,
    GlassesStreamConfig,
    StreamState,
    adapt_frame,
    auto_ladder,
    glasses_cue,
)
from streetsense.observations import ObservationLog
from streetsense.types import Alert, Priority


def test_toolkit_resolutions_and_rates():
    assert RESOLUTIONS == {"low": (360, 640), "medium": (504, 896), "high": (720, 1280)}
    assert FRAME_RATES == (2, 7, 15, 24, 30)
    assert StreamState.WAITING_FOR_DEVICE.value == "waitingForDevice"


@pytest.mark.parametrize("kw", [{"resolution": "4k"}, {"frame_rate": 60}, {"jpeg_quality": 5}])
def test_config_rejects_values_the_toolkit_does_not_offer(kw):
    with pytest.raises(ValueError):
        GlassesStreamConfig(**kw)


def test_auto_ladder_lowers_resolution_first_then_fps_not_below_15():
    steps = [(c.resolution, c.frame_rate) for c in auto_ladder(GlassesStreamConfig("high", 30))]
    assert steps == [("high", 30), ("medium", 30), ("low", 30), ("low", 24), ("low", 15)]
    assert [(c.resolution, c.frame_rate) for c in auto_ladder(GlassesStreamConfig("low", 7))] == [("low", 7)]


@pytest.mark.parametrize("shape", [(720, 1280, 3), (1080, 1920, 3), (1280, 720, 3), (100, 100, 3)])
def test_adapt_frame_gives_portrait_toolkit_size(shape):
    frame = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
    out = adapt_frame(frame, GlassesStreamConfig("medium"))
    assert out.shape == (896, 504, 3)


def test_adapt_frame_centre_crops_landscape():
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:, 600:680] = 255  # bright centre stripe stays in the crop
    out = adapt_frame(frame, GlassesStreamConfig("low", jpeg_quality=90))
    assert out[:, out.shape[1] // 2].mean() > 200
    assert out[:, :20].mean() < 30


def test_frame_pacer_keeps_newest_and_drops_stale():
    now = [0.0]
    p = FramePacer(GlassesStreamConfig(), max_age_s=0.5, clock=lambda: now[0])
    a, b = np.zeros((2, 2, 3)), np.ones((2, 2, 3))
    p.push(a, ts=0.0)
    p.push(b, ts=0.05)
    assert p.take() is b and p.take() is None
    p.push(a, ts=0.1)
    now[0] = 1.0
    assert p.take() is None
    assert p.received == 3 and p.dropped == 2


def test_cues_are_short_and_keep_light_wording_exact():
    c = glasses_cue(Alert("vehicle:left", "car close on your left", Priority.CRITICAL))
    assert c["text"] == "car close left" and c["earcon"] == "alarm"
    light = Alert("light:pedestrian", "crossing signals unclear, wait", Priority.WARNING,
                  data={"state": "unclear"})
    assert glasses_cue(light)["text"] == "crossing signals unclear, wait"
    assert glasses_cue(light)["earcon"] == "wait"
    walk = Alert("light:pedestrian", "walk signal on", Priority.NOTICE, data={"state": "walk"})
    assert glasses_cue(walk)["earcon"] == "walk"
    assert glasses_cue(Alert("muse_hazard:x", "wet leaves", Priority.INFO)) is None
    assert glasses_cue(None) is None
    long = glasses_cue(Alert("obstacle:bench:ahead", "bench ahead of you right now, about 3 metres",
                             Priority.WARNING))
    assert len(long["text"].split()) <= 5
    assert set(EARCONS) >= {"alarm", "warn", "walk", "wait", "notice"}


def test_observation_log_window_thins_and_keeps_newest():
    log = ObservationLog(maxlen=100)
    for i in range(60):
        log.add({"t": i * 0.5, "objects": []})
    w = log.window(10, max_entries=5)
    assert len(w) == 5 and w[-1]["t"] == 29.5 and w[0]["t"] >= 19.5
    assert log.latest()["t"] == 29.5
    assert ObservationLog().window(5) == []
