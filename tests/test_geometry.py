import math

import pytest

from streetsense.geometry import (
    clock_direction, estimate_distance_m, focal_length_px, horizontal_zone, in_walking_path, iou, round_distance,
)
from streetsense.types import Box

W, H = 1280, 720


def test_focal_length_90deg():
    assert focal_length_px(1000, 90) == pytest.approx(500)


def test_focal_length_rejects_bad_input():
    with pytest.raises(ValueError):
        focal_length_px(0, 70)


def test_distance_inverse_to_height():
    f = focal_length_px(W, 70)
    # a 1.7 m person 2 m away should be f*1.7/2 px tall
    h = f * 1.7 / 2
    b = Box(600, 300, 680, 300 + h - 1)
    d = estimate_distance_m(b, "person", W, H + 10_000, 70)  # tall canvas so it is not truncated
    assert d == pytest.approx(2.0, rel=0.02)
    b2 = Box(600, 300, 680, 300 + h / 2)
    assert estimate_distance_m(b2, "person", W, H, 70) == pytest.approx(4.0, rel=0.02)


def test_distance_unknown_class_and_truncated():
    assert estimate_distance_m(Box(0, 0, 10, 10), "kite", W, H) is None
    assert estimate_distance_m(Box(500, 0, 700, H), "person", W, H) == 0.5


@pytest.mark.parametrize("cx,zone", [(100, "left"), (640, "ahead"), (500, "ahead"), (1200, "right")])
def test_zones(cx, zone):
    assert horizontal_zone(Box(cx - 10, 0, cx + 10, 10), W) == zone


def test_clock_direction():
    assert clock_direction(Box(630, 0, 650, 10), W) == 12
    assert clock_direction(Box(1270, 0, 1280, 10), W, hfov_deg=120) == 2
    assert clock_direction(Box(0, 0, 10, 10), W, hfov_deg=120) == 10


def test_walking_path():
    assert in_walking_path(Box(600, 300, 700, 700), W, H)
    assert not in_walking_path(Box(0, 300, 100, 700), W, H)  # far left
    assert not in_walking_path(Box(600, 0, 700, 200), W, H)  # high in the frame (far / overhead)


def test_round_distance():
    assert round_distance(0.7) == "1 metre"
    assert round_distance(2.4) == "2 metres"
    assert round_distance(11) == "about 10 metres"


def test_iou():
    a = Box(0, 0, 10, 10)
    assert iou(a, a) == 1
    assert iou(a, Box(20, 20, 30, 30)) == 0
    assert math.isclose(iou(a, Box(5, 0, 15, 10)), 50 / 150)
