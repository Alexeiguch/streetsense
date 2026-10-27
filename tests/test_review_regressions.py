"""Regression tests for the Oct 2026 external code review.

Each test asserts the CORRECT behaviour (not the bug). They must stay offline.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from streetsense.alerts import AlertManager
from streetsense.config import AlertConfig, Config
from streetsense.geometry import estimate_distance_m, truncation_flags
from streetsense.meta_api import (
    SAFE_CROSSING_FALLBACK,
    fuse_crossing_decision,
    parse_json_reply,
    render_crossing_speech,
    validate_crossing_reply,
)
from streetsense.rules import FrameFacts, build_alerts, most_relevant_light, select_crossing_signal
from streetsense.speech import Speaker
from streetsense.traffic_light import LightStateSmoother, classify_light
from streetsense.types import Alert, Box, Detection, Priority


class Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


# ---------- P1-1: stale contradictory light speech ----------

def test_p1_queued_walk_invalidated_by_newer_red():
    """A pending walk must not speak after a newer red for the same signal family."""
    clock = Clock()
    mgr = AlertManager(AlertConfig(global_min_gap_s=1.5, default_cooldown_s=6.0), clock=clock)

    # t=0: unrelated alert occupies the speaker
    mgr.submit(Alert("crosswalk", "crossing ahead", Priority.NOTICE))
    assert mgr.pop_ready() is not None

    clock.t = 1.4
    mgr.submit_light(kind="pedestrian", state="walk", text="walk signal on", priority=Priority.WARNING)
    clock.t = 1.5
    mgr.submit_light(kind="pedestrian", state="red", text="pedestrian light is red", priority=Priority.WARNING)
    spoken_red = mgr.pop_ready()
    assert spoken_red is not None
    assert "red" in spoken_red.text

    # t=3.0: red still observed — revalidate then pop; walk must NOT speak
    clock.t = 3.0
    facts = FrameFacts(
        1280, 720,
        [Detection("traffic light", Box(600, 40, 640, 120), 0.9, track_id=1,
                   attrs={"light_state": "red", "signal_kind": "pedestrian"})],
    )
    mgr.revalidate(facts)
    nxt = mgr.pop_ready()
    assert nxt is None or "walk" not in (nxt.text or "").lower()


def test_p1_light_transition_walk_to_red_not_suppressed_by_shared_cooldown():
    """Sharing one cooldown key must not block announcing a real walk→red transition."""
    clock = Clock()
    mgr = AlertManager(AlertConfig(global_min_gap_s=0, default_cooldown_s=6.0), clock=clock)
    mgr.submit_light(kind="pedestrian", state="walk", text="walk signal on", priority=Priority.WARNING)
    assert mgr.pop_ready().text == "walk signal on"
    clock.t = 1.0
    mgr.submit_light(kind="pedestrian", state="red", text="pedestrian light is red", priority=Priority.WARNING)
    out = mgr.pop_ready()
    assert out is not None and "red" in out.text


# ---------- P1-2: untracked countdown / confirmation ----------

def test_p1_untracked_countdown_maps_to_unknown_not_crash():
    sm = LightStateSmoother()
    assert sm.update(None, "countdown") == "unknown"
    # rules must not KeyError on countdown if it somehow arrives
    dets = [Detection("traffic light", Box(100, 50, 140, 120), 0.9, track_id=None,
                      attrs={"light_state": "countdown", "signal_kind": "pedestrian"})]
    # after smoother policy, build_alerts should tolerate unknown; if countdown slips through, no KeyError
    alerts = build_alerts(FrameFacts(640, 480, dets))
    assert isinstance(alerts, list)


def test_p1_untracked_walk_needs_confirmation():
    """A single untracked walk frame must not announce walk (3-vote / confirmation policy)."""
    sm = LightStateSmoother(min_votes=3)
    assert sm.update(None, "walk") == "unknown"  # first sighting
    # After enough consistent position-keyed updates, walk can confirm
    for _ in range(3):
        state = sm.update(None, "walk", box=Box(100, 50, 140, 120))
    assert state == "walk"


def test_p1_classifier_smoother_rules_countdown_crop_no_crash():
    """Full path: orange countdown-like crop + untracked det → process-like attrs → build_alerts."""
    # Synthetic orange/red-lower crop approximating countdown
    crop = np.zeros((80, 40, 3), np.uint8)
    crop[50:75, 5:35] = (0, 100, 255)  # BGR orange-ish bottom
    crop[50:75, 5:35, 1] = 140
    reading = classify_light(crop, "pedestrian")
    sm = LightStateSmoother()
    state = sm.update(None, reading.state)
    assert state != "countdown"
    dets = [Detection("traffic light", Box(10, 10, 50, 90), 0.9, track_id=None,
                      attrs={"light_state": state, "signal_kind": "pedestrian",
                             "countdown": bool(getattr(reading, "countdown", False))})]
    build_alerts(FrameFacts(200, 200, dets))  # must not raise


# ---------- P1-3: signal selection ----------

def test_p1_central_red_beats_tiny_edge_walk():
    big_red = Detection("traffic light", Box(560, 40, 640, 200), 0.9, track_id=1,
                        attrs={"signal_kind": "pedestrian", "light_state": "red"})
    tiny_walk = Detection("traffic light", Box(20, 10, 40, 40), 0.8, track_id=2,
                          attrs={"signal_kind": "pedestrian", "light_state": "walk"})
    sel = select_crossing_signal([big_red, tiny_walk], frame_w=1280, frame_h=720)
    assert sel is not None
    assert sel.decision in {"red", "unclear", "stop"} or sel.attrs.get("light_state") == "red"
    # Prefer structured result
    if hasattr(sel, "decision"):
        assert sel.decision != "walk"
    else:
        assert sel.attrs["light_state"] == "red"


def test_p1_conflicting_ped_signals_are_unclear():
    left_walk = Detection("traffic light", Box(200, 40, 260, 140), 0.9, track_id=1,
                          attrs={"signal_kind": "pedestrian", "light_state": "walk"})
    right_red = Detection("traffic light", Box(900, 40, 960, 140), 0.9, track_id=2,
                          attrs={"signal_kind": "pedestrian", "light_state": "red"})
    # Both reasonably central-ish / similar size — conflict
    sel = select_crossing_signal([left_walk, right_red], frame_w=1280, frame_h=720)
    assert sel is not None
    assert getattr(sel, "decision", None) == "unclear" or (
        isinstance(sel, Detection) and sel.attrs.get("light_state") == "unclear"
    )
    alerts = build_alerts(FrameFacts(1280, 720, [left_walk, right_red]))
    texts = " ".join(a.text.lower() for a in alerts)
    assert "unclear" in texts or "wait" in texts or "conflict" in texts


# ---------- P1-4: speech urgency ----------

def test_p1_critical_evicts_lower_priority_in_speaker():
    clock = Clock(10.0)
    sp = Speaker(enabled=True, backend="queue", clock=clock, max_age_s=100)  # test backend
    for i in range(4):
        sp.say(Alert(f"n{i}", f"notice {i}", Priority.NOTICE, ts=10.0))
    delivered = sp.say(Alert("vehicle:ahead", "car approaching ahead", Priority.CRITICAL, ts=10.0))
    assert delivered is True
    keys = [a.key for a in sp.pending_alerts()]
    assert "vehicle:ahead" in keys


def test_p1_dropped_alert_not_marked_delivered():
    clock = Clock(10.0)
    sp = Speaker(enabled=True, backend="queue", maxsize=2, clock=clock, max_age_s=100)
    assert sp.say(Alert("a", "a", Priority.NOTICE, ts=10)) is True
    assert sp.say(Alert("b", "b", Priority.NOTICE, ts=10)) is True
    # Same priority, queue full → drop new, report not delivered
    assert sp.say(Alert("c", "c", Priority.NOTICE, ts=10)) is False


# ---------- P1-5: crossing output contract ----------

def test_p1_malformed_muse_reply_safe_fallback():
    raw = parse_json_reply("Yes, cross now.")
    validated = validate_crossing_reply(raw)
    assert validated["ok"] is False
    speech = render_crossing_speech(validated, local_state="unknown")
    assert speech == SAFE_CROSSING_FALLBACK
    assert "cross now" not in speech.lower() or "can't" in speech.lower()


def test_p1_go_requires_local_walk_and_muse_yes():
    good = validate_crossing_reply({
        "summary": "Walk signal visible",
        "hazards": [],
        "crossing_visible": True,
        "pedestrian_signal": "green",
        "safe_to_cross": "yes",
    })
    assert good["ok"] is True
    go_speech = render_crossing_speech(good, local_state="walk")
    assert "clear" in go_speech.lower() or "walk" in go_speech.lower()
    # Free summary must never become the spoken go instruction alone
    speech = render_crossing_speech(good, local_state="red")
    assert speech == "Do not cross."
    fused = fuse_crossing_decision("red", good["data"])
    assert fused["decision"] != "go"


def test_p1_stale_crossing_answer_expires():
    validated = validate_crossing_reply({
        "summary": "ok",
        "hazards": [],
        "crossing_visible": True,
        "pedestrian_signal": "green",
        "safe_to_cross": "yes",
    }, capture_ts=100.0)
    speech = render_crossing_speech(validated, local_state="walk", now=130.0, max_age_s=10.0)
    assert speech == SAFE_CROSSING_FALLBACK


# ---------- P2-8: truncation ----------

def test_p2_bottom_clipped_person_triggers_close_warning():
    # Person box touching bottom edge only, large visible height
    box = Box(500, 200, 700, 719)  # y2 at bottom of 720 frame
    flags = truncation_flags(box, 1280, 720)
    assert flags["bottom"] is True
    dist = estimate_distance_m(box, "person", 1280, 720)
    # Must not claim a precise far distance that suppresses warnings
    assert dist is None or dist < 4.0 or flags.get("uncertain")
    d = Detection("person", box, 0.9, attrs={})
    # rules with conservative proximity
    alerts = build_alerts(FrameFacts(1280, 720, [d]))
    assert any(a.key.startswith("person") for a in alerts)


# ---------- P2-9: countdown hold expiry ----------

def test_p2_countdown_hold_expires_walk():
    clock = Clock(0.0)
    sm = LightStateSmoother(min_votes=3, countdown_hold_s=1.0, clock=clock)
    assert sm.update(7, "walk") == "unknown"
    assert sm.update(7, "walk") == "unknown"
    assert sm.update(7, "walk") == "walk"
    # within hold, countdown preserves walk
    clock.t = 0.5
    assert sm.update(7, "countdown") == "walk"
    # after hold expiry → countdown/uncertain, not walk
    clock.t = 2.0
    out = sm.update(7, "countdown")
    assert out in ("countdown", "unknown", "uncertain")
    assert out != "walk"


# ---------- P2-7: open-vocab scheduling helpers ----------

def test_p2_open_vocab_scheduler_respects_interval_on_empty():
    from streetsense.open_vocab_sched import OpenVocabScheduler

    sched = OpenVocabScheduler(every_n_frames=15)
    calls = 0
    for i in range(30):
        if sched.should_run(i):
            calls += 1
            sched.mark_attempt(i)
            sched.record_result([], failed=False, frame_i=i)
    assert calls == 2  # frames 0 and 15


def test_p2_open_vocab_backs_off_on_error():
    from streetsense.open_vocab_sched import OpenVocabScheduler

    sched = OpenVocabScheduler(every_n_frames=15, error_backoff_frames=30)
    assert sched.should_run(0)
    sched.mark_attempt(0)
    sched.record_result([], failed=True, frame_i=0)
    # next frames within backoff should not run
    assert not sched.should_run(1)
    assert not sched.should_run(15)


def test_camera_health_detects_dark_and_blur():
    from streetsense.web import camera_health

    dark = np.zeros((120, 160, 3), np.uint8)
    assert camera_health(dark)["issue"] == "covered"
    sharp = np.zeros((120, 160, 3), np.uint8)
    sharp[::2, ::2] = 255
    # High-frequency checkerboard should not be "blurred"
    h = camera_health(sharp)
    assert h is None or h["issue"] != "blurred"


def test_event_trace_replay_summary(tmp_path):
    from streetsense.event_trace import EventTrace, replay_summary

    path = tmp_path / "t.jsonl"
    tr = EventTrace(path)
    tr.log(capture_ts=1.0, spoken_priority="CRITICAL", facts={"light": "red"})
    tr.log(capture_ts=2.0, wrong_walk=True, transition_delay_s=0.4)
    tr.close()
    s = replay_summary(path)
    assert s["wrong_walk_announcements"] == 1
    assert s["critical_per_min"] > 0
