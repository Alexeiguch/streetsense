"""Fresh confirmation after a veto, a missing signal or a replay rewind."""
from types import SimpleNamespace

import numpy as np
import pytest

from streetsense.config import Config
from streetsense.signal_cnn import SignalPrediction
from streetsense.signal_trace import summarise_signals
from streetsense.traffic_light import LightReading, LightStateSmoother
from streetsense.types import Box, Detection


@pytest.mark.parametrize("track_id", [1, None])
@pytest.mark.parametrize("veto_state", ["countdown", "unknown", "red"])
def test_veto_requires_fresh_walk_votes(track_id, veto_state):
    sm = LightStateSmoother(reset_walk_on_veto=True)
    box = Box(20, 20, 60, 100)
    for i in range(3):
        state = sm.update(track_id, "walk", box, t=i / 10)
    assert state == "walk"
    assert sm.update(track_id, veto_state, box, t=0.3, walk_blocked=True) != "walk"
    assert sm.update(track_id, "countdown", box, t=0.4) != "walk"  # no stale hold
    assert sm.update(track_id, "walk", box, t=0.5) != "walk"
    assert sm.update(track_id, "walk", box, t=0.6) != "walk"
    assert sm.update(track_id, "walk", box, t=0.7) == "walk"


def test_veto_during_recovery_restarts_confirmation():
    sm = LightStateSmoother(reset_walk_on_veto=True)
    for i, state in enumerate(["walk", "walk", "unknown"] * 4):
        assert sm.update(1, state, t=i / 10, walk_blocked=state == "unknown") != "walk"


def test_countdown_cannot_bypass_walk_confirmation():
    sm = LightStateSmoother(reset_walk_on_veto=True)
    assert sm.update(1, "walk", t=0.0) == "unknown"
    assert sm.update(1, "countdown", t=0.1) != "walk"
    for i in range(3):
        sm.update(1, "walk", t=0.2 + i / 10)
    assert sm.update(1, "unknown", t=0.5, walk_blocked=True) != "walk"
    assert sm.update(1, "walk", t=0.6) == "unknown"
    assert sm.update(1, "countdown", t=0.7) != "walk"


@pytest.mark.parametrize("t,max_gap", [(2.0, 1.0), (-1.0, None)])
@pytest.mark.parametrize("track_id", [1, None])
def test_gap_or_rewind_cannot_reuse_walk_votes(t, max_gap, track_id):
    sm = LightStateSmoother(max_gap_s=max_gap)
    box = Box(20, 20, 60, 100)
    for i in range(3):
        sm.update(track_id, "walk", box, t=i / 10)
    assert sm.update(track_id, "walk", box, t=t) == "unknown"


@pytest.mark.parametrize("track_id", [1, None])
@pytest.mark.parametrize("reset_on_veto", [False, True])
def test_fresh_red_cancels_walk_and_its_countdown_hold(track_id, reset_on_veto):
    sm = LightStateSmoother(reset_walk_on_veto=reset_on_veto)
    box = Box(20, 20, 60, 100)
    for i in range(3):
        state = sm.update(track_id, "walk", box, t=i / 10)
    assert state == "walk"
    assert sm.update(track_id, "red", box, t=0.3) != "walk"
    assert sm.update(track_id, "countdown", box, t=0.4) != "walk"
    assert sm.update(track_id, "walk", box, t=0.5) != "walk"
    assert sm.update(track_id, "walk", box, t=0.6) != "walk"
    assert sm.update(track_id, "walk", box, t=0.7) == "walk"


@pytest.mark.parametrize("track_id", [1, None])
@pytest.mark.parametrize("gap", [1.1, 2.0, 5.0])
def test_default_can_confirm_steady_walk_at_slow_frame_rates(track_id, gap):
    sm = LightStateSmoother(reset_walk_on_veto=False)
    box = Box(20, 20, 60, 100)
    states = [sm.update(track_id, "walk", box, t=i * gap) for i in range(4)]
    assert states == ["unknown", "unknown", "walk", "walk"]


@pytest.mark.parametrize("track_id", [1, None])
@pytest.mark.parametrize("confirmed", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_long_countdown_discards_pre_countdown_walk_votes(track_id, confirmed, sparse):
    sm = LightStateSmoother(reset_walk_on_veto=False)
    box = Box(20, 20, 60, 100)
    for i in range(3 if confirmed else 1):
        sm.update(track_id, "walk", box, t=i / 10)
    # Sparse input expires between the last countdown frame and the next WALK.
    for t in ([0.3] if sparse else [0.3, 0.6, 0.9, 1.2, 1.5]):
        sm.update(track_id, "countdown", box, t=t)
    assert sm.update(track_id, "walk", box, t=1.6) != "walk"
    assert sm.update(track_id, "walk", box, t=1.7) != "walk"
    assert sm.update(track_id, "walk", box, t=1.8) == "walk"


def test_explicit_frame_time_controls_countdown_hold():
    sm = LightStateSmoother(clock=lambda: 999.0, max_gap_s=3.0)
    for i in range(3):
        sm.update(1, "walk", t=i / 10)
    assert sm.update(1, "countdown", t=0.5) == "walk"
    assert sm.update(1, "countdown", t=1.3) != "walk"


def test_pipeline_veto_recovery_and_trace(monkeypatch):
    import streetsense.pipeline as sp

    monkeypatch.setenv("STREETSENSE_SIGNAL_RESET_ON_VETO", "1")

    probs = [(0.99, 0.0, 0.0, 0.01)] * 3 + [(0.01, 0.0, 0.98, 0.01)] + [(0.99, 0.0, 0.0, 0.01)] * 3
    predictions = iter(probs)
    monkeypatch.setattr(sp, "build_detector", lambda cfg: lambda img: [
        Detection("traffic light", Box(140, 30, 180, 110), 0.9, 1)])
    monkeypatch.setattr(sp, "classify_light", lambda crop, kind: LightReading("walk", 0.99, 0.1, None))
    monkeypatch.setattr(sp, "figure_walk_guard", lambda r, *args: r)
    cnn = type("CNN", (), {"t_walk": 0.95, "t_veto": 0.8,
                          "__call__": lambda self, crop: SignalPrediction(next(predictions))})()
    monkeypatch.setattr(sp.signal_cnn, "get", lambda: cnn)
    cfg = Config()
    cfg.vlm.enabled = cfg.sam.enabled = cfg.depth.enabled = cfg.open_vocab.enabled = False
    clock = SimpleNamespace(t=100.0)
    pipe = sp.Pipeline(cfg, use_surface=False, clock=lambda: clock.t)
    assert pipe.smoother.clock() == 100.0
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    records = []
    for i in range(7):
        out = pipe.process(img, annotate=False, t=i / 10)
        records.append(summarise_signals(out, 320, 240))
    assert [r["heads"][0]["state"] for r in records] == [
        "unknown", "unknown", "walk", "unknown", "unknown", "unknown", "walk"]
    head = records[3]["heads"][0]
    assert head["walk_blocked"] and head["cnn_probs"]["countdown"] == 0.98
    assert head["heuristic"] == "walk" and head["raw"] == "countdown"
    assert records[3]["local"] != "walk"
    # The diagnostic replay must reproduce selection without another detector or CNN run.
    from scripts.replay_signal_trace import replay
    replayed = list(replay(records, reset_on_veto=True))
    assert [r["local"] for r in replayed] == [r["local"] for r in records]
    assert [r["heads"][0]["state"] for r in replayed] == [r["heads"][0]["state"] for r in records]


def test_stricter_reset_is_opt_in(monkeypatch):
    monkeypatch.delenv("STREETSENSE_SIGNAL_RESET_ON_VETO", raising=False)
    assert not LightStateSmoother().reset_walk_on_veto
    monkeypatch.setenv("STREETSENSE_SIGNAL_RESET_ON_VETO", "1")
    assert LightStateSmoother().reset_walk_on_veto
    assert not LightStateSmoother(reset_walk_on_veto=False).reset_walk_on_veto


def test_default_countdown_hold_requires_confirmed_walk():
    sm = LightStateSmoother(reset_walk_on_veto=False)
    assert sm.update(1, "walk", t=0) == "unknown"
    assert sm.update(1, "countdown", t=0.1) != "walk"
    sm.update(1, "walk", t=0.2)
    assert sm.update(1, "walk", t=0.3) == "walk"
    assert sm.update(1, "countdown", t=0.4) == "walk"
