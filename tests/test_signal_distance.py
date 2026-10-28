"""Distance to signal heads / signs, and the caution-only rules built on it."""
import math
import random

import numpy as np
import pytest

from streetsense import signal_distance as sd
from streetsense.rules import FrameFacts, build_alerts, select_crossing_signal
from streetsense.traffic_light import classify_light
from streetsense.types import Box, Detection

W, H, HFOV = 1280, 720, 70.0
F = (W / 2) / math.tan(math.radians(HFOV) / 2)  # 914.0 px
CFG = sd.SignalDistanceConfig()


def light(x1, y1, x2, y2, state, kind="pedestrian", tid=None):
    return Detection("traffic light", Box(x1, y1, x2, y2), 0.9, track_id=tid,
                     attrs={"light_state": state, "signal_kind": kind})


def centred(h, state, kind="pedestrian", cx=640, y1=100):
    w = h / 2
    return light(cx - w / 2, y1, cx + w / 2, y1 + h, state, kind)


def pick(dets, **kw):
    return select_crossing_signal(dets, W, H, None, hfov_deg=HFOV, dist_cfg=kw.pop("cfg", CFG))


# ---------- geometry ----------

def test_pinhole_estimate_and_range():
    e = sd.estimate(Box(615, 100, 665, 200), "traffic light", "pedestrian", W, H, HFOV)
    assert e.source == "box" and e.band == "near"
    assert e.m == pytest.approx(0.5 * F / 100, rel=1e-6)
    assert e.lo == pytest.approx(0.3 * F / 100, rel=1e-6) and e.hi == pytest.approx(0.8 * F / 100, rel=1e-6)
    assert abs(e.bearing_deg) < 1e-9


def test_vehicle_and_stop_sign_priors():
    v = sd.estimate(Box(615, 100, 665, 200), "traffic light", "vehicle", W, H, HFOV)
    s = sd.estimate(Box(590, 100, 690, 200), "stop sign", "unknown", W, H, HFOV)
    assert v.m == pytest.approx(0.95 * F / 100) and s.m == pytest.approx(0.75 * F / 100)
    assert sd.estimate(Box(615, 100, 665, 200), "person", "unknown", W, H, HFOV).m is None


def test_edge_clipped_box_is_unknown():
    for b in (Box(600, 0, 640, 80), Box(600, 640, 640, 720), Box(0, 100, 40, 180), Box(1240, 100, 1280, 180)):
        e = sd.estimate(b, "traffic light", "pedestrian", W, H, HFOV)
        assert e.m is None and e.band == "unknown"


def test_bearing_and_slant_range():
    edge = Box(1270 - 20, 100, 1270, 200)
    e = sd.estimate(edge, "traffic light", "pedestrian", W, H, HFOV)
    assert e.bearing_deg == pytest.approx(math.degrees(math.atan((1260 - 640) / F)))
    assert e.m == pytest.approx(0.5 * F / 100 / math.cos(math.radians(e.bearing_deg)))
    assert sd.bearing_deg(Box(0, 0, 2, 2), W, HFOV) < -34


def test_band_threshold():
    near = sd.estimate(Box(630, 100, 650, 100 + 0.5 * F / 24), "traffic light", "pedestrian", W, H, HFOV)
    far = sd.estimate(Box(630, 100, 650, 100 + 0.5 * F / 26), "traffic light", "pedestrian", W, H, HFOV)
    assert near.band == "near" and far.band == "far"


def test_fuse_depth():
    e = sd.estimate(Box(615, 100, 665, 200), "traffic light", "pedestrian", W, H, HFOV)  # 4.57 m (2.7-7.3)
    assert sd.fuse_depth(e, None) == e
    ok = sd.fuse_depth(e, 6.0)
    assert ok.source == "box+depth" and ok.m == pytest.approx(math.sqrt(e.m * 6.0))
    bad = sd.fuse_depth(e, 30.0)
    assert bad.source == "depth-conflict" and bad.m == pytest.approx(e.m) and bad.hi == 30.0
    unk = sd.estimate(Box(600, 0, 640, 80), "traffic light", "pedestrian", W, H, HFOV)
    assert sd.fuse_depth(unk, 12.0).source == "depth"


def test_spoken_distance():
    assert sd.spoken_distance(None) == ""
    e = sd.estimate(Box(630, 100, 650, 100 + 0.5 * F / 15), "traffic light", "pedestrian", W, H, HFOV)
    assert sd.spoken_distance(e) == "about 15 metres ahead"
    left = sd.estimate(Box(100, 100, 120, 160), "traffic light", "pedestrian", W, H, HFOV)
    assert sd.spoken_distance(left).endswith("to your left")


# ---------- config ----------

def test_camera_presets_and_override(monkeypatch):
    monkeypatch.delenv("STREETSENSE_CAMERA_HFOV_DEG", raising=False)
    monkeypatch.delenv("STREETSENSE_CAMERA", raising=False)
    assert sd.camera_hfov_deg() == 70.0
    monkeypatch.setenv("STREETSENSE_CAMERA", "rayban")
    assert sd.camera_hfov_deg() == 68.0
    monkeypatch.setenv("STREETSENSE_CAMERA_HFOV_DEG", "52")
    assert sd.camera_hfov_deg() == 52.0
    monkeypatch.setenv("STREETSENSE_CAMERA_HFOV_DEG", "nan")
    assert sd.camera_hfov_deg() == 70.0
    monkeypatch.setenv("STREETSENSE_CAMERA_HFOV_DEG", "500")
    assert sd.camera_hfov_deg() == 170.0


def test_distance_config_env_is_clamped(monkeypatch):
    monkeypatch.setenv("STREETSENSE_SIGNAL_MAX_WALK_M", "1e9")
    monkeypatch.setenv("STREETSENSE_SIGNAL_MAX_BEARING_DEG", "180")
    monkeypatch.setenv("STREETSENSE_SIGNAL_NEARER_MARGIN", "0")
    c = sd.SignalDistanceConfig.from_env()
    assert c.max_walk_m == 80.0 and c.max_bearing_deg == 50.0 and c.nearer_margin == 0.5
    monkeypatch.setenv("STREETSENSE_SIGNAL_MAX_WALK_M", "junk")
    assert sd.SignalDistanceConfig.from_env().max_walk_m == 40.0


# ---------- selection: distance only adds caution ----------

def test_near_walk_still_walk():
    s = pick([centred(60, "walk")])  # ~7.6 m ahead
    assert s.decision == "walk" and s.caution is None and s.distance.band == "near"


def test_far_walk_is_unclear():
    s = pick([centred(8, "walk")])  # ~57 m
    assert s.decision == "unclear" and s.caution == "far"


def test_side_walk_is_unclear():
    s = pick([light(1100, 100, 1130, 160, "walk")])  # ~27 deg right
    assert s.decision == "unclear" and s.caution == "side"


def test_single_clipped_walk_unchanged_but_several_is_unclear():
    clipped = light(620, 0, 660, 80, "walk")
    assert pick([clipped]).decision == "walk"
    other = light(560, 100, 590, 160, "countdown")
    s = pick([clipped, other])
    assert s.decision == "unclear" and s.caution == "unknown"


def test_nearer_non_walk_head_blocks_farther_walk():
    far_walk = centred(30, "walk", cx=660)  # ~15 m
    near_amber = centred(90, "yellow", kind="unknown", cx=600)  # ~7 m nominal for unknown kind
    s = pick([far_walk, near_amber])
    assert s.decision == "unclear" and s.caution == "nearer"


def test_nearer_green_head_does_not_block_walk():
    # EU green-man heads often read "green" rather than "walk"; they agree with the walk head.
    s = pick([centred(30, "walk", cx=660), centred(90, "green", cx=600)])
    assert s.decision == "walk"


def test_nearest_of_two_walks_is_selected():
    near, far = centred(80, "walk", cx=600), centred(30, "walk", cx=680)
    s = pick([far, near])
    assert s.decision == "walk" and s.detection is near


def test_red_conflict_rule_unchanged():
    s = pick([centred(80, "walk", cx=600), centred(20, "red", cx=680)])
    assert s.decision == "unclear"


def test_alert_speech_includes_distance_and_caution():
    walk = centred(0.5 * F / 15, "walk")
    (a,) = [x for x in build_alerts(FrameFacts(W, H, [walk]), HFOV) if x.key.startswith("light:")]
    assert a.text == "walk signal on, about 15 metres ahead"
    (b,) = [x for x in build_alerts(FrameFacts(W, H, [centred(8, "walk")]), HFOV) if x.key.startswith("light:")]
    assert "far away" in b.text and "wait" in b.text and b.data["caution"] == "far"


def test_fuzz_distance_never_creates_walk():
    """Against a permissive config (distance rules effectively off), the default config may only
    remove walks, never add one, and nothing without a walk head ever becomes walk."""
    loose = sd.SignalDistanceConfig(max_walk_m=1e9, near_m=25, max_bearing_deg=180, nearer_margin=0.0)
    rng = random.Random(7)
    states = ["walk", "red", "green", "yellow", "countdown", "unclear", "unknown"]
    kinds = ["pedestrian", "unknown", "vehicle"]
    n_walk_strict = n_walk_loose = 0
    for _ in range(3000):
        dets = []
        for _ in range(rng.randint(1, 4)):
            h = rng.uniform(3, 300)
            x1, y1 = rng.uniform(-20, W), rng.uniform(-20, H)
            d = light(x1, y1, x1 + h / rng.uniform(0.8, 3), y1 + h, rng.choice(states), rng.choice(kinds))
            if rng.random() < 0.5:
                d.attrs["signal_depth_rel"] = rng.uniform(0.1, 5)
            dets.append(d)
        strict = pick(dets)
        lo = pick(dets, cfg=loose)
        if strict is not None and strict.decision == "walk":
            n_walk_strict += 1
            assert lo is not None and lo.decision == "walk"
            assert strict.detection.attrs["light_state"] == "walk"
        n_walk_loose += bool(lo is not None and lo.decision == "walk")
    assert n_walk_strict < n_walk_loose  # the rules did fire somewhere


def rel(d, v):
    d.attrs["signal_depth_rel"] = v
    return d


def test_nearer_unreadable_head_blocks_walk_by_size():
    # e.g. Most dusk: the near red head blooms to "unknown" while a farther green-man head reads.
    s = pick([centred(30, "walk", cx=660), centred(90, "unknown", cx=600)])
    assert s.decision == "unclear" and s.caution == "nearer"


def test_nearer_unreadable_head_blocks_walk_by_depth_even_if_box_says_farther():
    walk = rel(centred(90, "walk", cx=600), 3.1)  # box: near
    unread = rel(centred(40, "unknown", cx=660), 4.2)  # box: farther, but depth says nearer
    s = pick([walk, unread])
    assert s.decision == "unclear" and s.caution == "nearer"
    # depth on only one head is not a comparison: the walk stands
    walk2, unread2 = centred(90, "walk", cx=600), rel(centred(40, "unknown", cx=660), 4.2)
    assert pick([walk2, unread2]).decision == "walk"


def test_farther_or_side_or_vehicle_unreadable_does_not_block():
    walk = centred(80, "walk", cx=640)
    assert pick([walk, centred(30, "unknown", cx=660)]).decision == "walk"  # farther
    assert pick([walk, centred(200, "unknown", cx=1150, y1=300)]).decision == "walk"  # off to the side
    assert pick([walk, centred(200, "unknown", kind="vehicle", cx=600)]).decision == "walk"


def test_green_gets_the_same_distance_checks():
    assert pick([centred(80, "green", cx=640)]).decision == "green"
    s = pick([centred(30, "green", cx=660), centred(90, "unknown", cx=600)])
    assert s.decision == "unclear" and s.caution == "nearer"
    assert pick([centred(8, "green")]).decision == "unclear"  # far


def test_sample_depth_rel_uses_box_centre():
    dm = np.ones((100, 100), np.float32)
    dm[40:60, 40:60] = 4.0
    assert sd.sample_depth_rel(dm, Box(30, 30, 70, 70)) == pytest.approx(4.0)
    assert sd.sample_depth_rel(np.zeros((10, 10), np.float32), Box(0, 0, 10, 10)) is None


# ---------- reader fix found while diagnosing ----------

def test_white_between_vehicle_visors_is_not_walk():
    crop = np.full((90, 30, 3), 20, np.uint8)
    crop[15:40, 8:22] = 255  # white sky/wall showing through the upper interior
    assert classify_light(crop, "vehicle").state != "walk"
    assert classify_light(crop, "unknown").state == "walk"  # the US walk-figure rule itself is unchanged
