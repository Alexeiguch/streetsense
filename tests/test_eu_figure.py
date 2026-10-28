"""EU-style green walking figure: per-lamp check, night bloom, glare, and the safety guards."""

import cv2
import numpy as np
import pytest

from streetsense.detector import light_zoom_scales, zoom_out_canvas
from streetsense.config import DetectorConfig
from streetsense.traffic_light import (
    beside_display, classify_light, figure_walk_guard, lamp_figure,
)
from streetsense.types import Box

GREEN = (150, 230, 0)  # cyan-ish LED green (BGR)


def head(w=120, h=240) -> np.ndarray:
    """Two-lamp EU head: dark housing, two dark lamps (stop on top, walk below)."""
    img = np.full((h, w, 3), 30, np.uint8)
    for cy in (h // 4, 3 * h // 4):
        cv2.circle(img, (w // 2, cy), int(w * 0.42), (22, 22, 22), -1)
    return img


def walking_man(img, cx, cy, r, colour, dotted=True, thick=None):
    """Stick figure with legs apart, optionally made of LED dots like a real head."""
    m = np.zeros(img.shape[:2], np.uint8)
    t = thick or max(2, int(r * 0.14))
    cv2.circle(m, (cx, int(cy - 0.75 * r)), max(2, int(0.15 * r)), 255, -1)
    hip = (cx, int(cy + 0.15 * r))
    cv2.line(m, (cx, int(cy - 0.55 * r)), hip, 255, t)
    cv2.line(m, hip, (int(cx - 0.42 * r), int(cy + 0.85 * r)), 255, t)
    cv2.line(m, hip, (int(cx + 0.42 * r), int(cy + 0.85 * r)), 255, t)
    cv2.line(m, (cx, int(cy - 0.4 * r)), (int(cx - 0.35 * r), int(cy - 0.05 * r)), 255, t)
    cv2.line(m, (cx, int(cy - 0.4 * r)), (int(cx + 0.35 * r), int(cy - 0.1 * r)), 255, t)
    if dotted:
        grid = np.zeros_like(m)
        grid[::3, ::3] = 255
        grid = cv2.dilate(grid, np.ones((2, 2), np.uint8))
        m &= grid
    img[m > 0] = colour
    return m


def green_man_head(**kw):
    img = head()
    walking_man(img, 60, 180, 42, kw.get("colour", GREEN), dotted=kw.get("dotted", True))
    return img


def test_dotted_green_man_in_lower_lamp_is_walk():
    r = classify_light(green_man_head(), kind="unknown")
    assert r.state == "walk" and r.figure


def test_solid_green_man_also_walk_on_pedestrian_kind():
    assert classify_light(green_man_head(dotted=False), kind="pedestrian").state == "walk"


def test_vehicle_kind_is_never_walk():
    assert classify_light(green_man_head(), kind="vehicle").state != "walk"
    assert lamp_figure(green_man_head(), kind="vehicle")[0] is False


def test_round_green_vehicle_lamp_is_not_walk():
    img = head()
    cv2.circle(img, (60, 180), 40, GREEN, -1)
    for kind in ("unknown", "pedestrian", "vehicle"):
        assert classify_light(img, kind=kind).state != "walk"


def test_green_arrow_is_not_walk():
    img = head()
    pts = np.array([[60, 145], [90, 185], [70, 185], [70, 215], [50, 215], [50, 185], [30, 185]], np.int32)
    cv2.fillPoly(img, [pts], GREEN)
    assert classify_light(img, kind="unknown").state != "walk"


def seven_seg_8(img, x, y, w, h, colour):
    t = max(3, w // 6)
    for (x1, y1, x2, y2) in [(x, y, x + w, y), (x, y + h // 2, x + w, y + h // 2), (x, y + h, x + w, y + h),
                             (x, y, x, y + h), (x + w, y, x + w, y + h)]:
        cv2.line(img, (x1, y1), (x2, y2), colour, t)


def test_countdown_digit_is_not_a_figure():
    # Most (CZ): a green "8" in its own square box beside the head. Tall, hollow, centred:
    # it passed the old silhouette test's shape rules; feet-apart rejects it.
    img = np.full((112, 100, 3), 30, np.uint8)
    cv2.circle(img, (50, 56), 46, (22, 22, 22), -1)
    seven_seg_8(img, 40, 25, 26, 60, GREEN)
    assert classify_light(img, kind="pedestrian").state != "walk"
    assert lamp_figure(img, "pedestrian")[0] is False


def test_figure_in_upper_lamp_is_not_walk():
    img = head()
    walking_man(img, 60, 60, 42, GREEN)
    assert classify_light(img, kind="unknown").state != "walk"


def test_lit_red_lamp_blocks_figure_walk():
    img = green_man_head()
    cv2.circle(img, (60, 60), 40, (0, 0, 255), -1)
    assert classify_light(img, kind="unknown").state != "walk"


def test_green_countdown_in_upper_lamp_blocks_walk():
    img = green_man_head()
    seven_seg_8(img, 45, 30, 26, 55, GREEN)
    assert classify_light(img, kind="unknown").state != "walk"


def test_sun_washed_pale_green_figure_is_walk():
    # Glare: saturation ~70 instead of ~255. The old lit mask (sat >= 90) sees nothing.
    pale = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[80, 70, 230]]]), cv2.COLOR_HSV2BGR)[0, 0])
    r = classify_light(green_man_head(colour=pale), kind="unknown")
    assert r.state == "walk"


def test_night_bloom_white_core_with_green_halo_is_walk():
    img = head()
    m = walking_man(img, 60, 180, 42, (40, 160, 0), dotted=False, thick=12)  # halo
    core = cv2.erode(m, np.ones((7, 7), np.uint8))
    img[core > 0] = (250, 255, 250)
    assert classify_light(img, kind="unknown").state == "walk"


def test_white_figure_without_green_halo_is_not_eu_walk():
    img = head()
    walking_man(img, 60, 180, 42, (255, 255, 255), dotted=False, thick=8)
    ok, info = lamp_figure(img, "unknown")
    assert ok is False


def test_figure_touching_crop_edge_is_not_walk():
    img = head()
    walking_man(img, 60, 200, 60, GREEN)  # runs off the bottom of the box
    assert lamp_figure(img, "unknown")[0] is False


def test_bright_box_is_refused():
    img = np.full((240, 120, 3), 200, np.uint8)
    walking_man(img, 60, 180, 42, GREEN)
    assert lamp_figure(img, "unknown")[0] is False


def test_env_switch_turns_figure_check_off(monkeypatch):
    monkeypatch.setenv("STREETSENSE_EU_FIGURE", "0")
    assert classify_light(green_man_head(), kind="unknown").state != "walk"


def frame_with_head(display: bool):
    frame = np.full((400, 600, 3), 90, np.uint8)
    frame[100:340, 300:420] = green_man_head()
    if display:  # green countdown digits in a separate box to the left
        frame[150:262, 190:290] = 30
        seven_seg_8(frame, 225, 175, 26, 60, GREEN)
    return frame, Box(300, 100, 420, 340)


def test_display_beside_head_downgrades_figure_walk():
    frame, box = frame_with_head(display=True)
    r = classify_light(frame[100:340, 300:420], kind="unknown")
    assert r.state == "walk"
    assert beside_display(frame, box)
    g = figure_walk_guard(r, frame, box)
    assert g.state == "green" and g.countdown


def test_no_display_keeps_figure_walk():
    frame, box = frame_with_head(display=False)
    r = classify_light(frame[100:340, 300:420], kind="unknown")
    assert not beside_display(frame, box)
    assert figure_walk_guard(r, frame, box).state == "walk"


def test_guard_leaves_us_white_walk_alone():
    from streetsense.traffic_light import LightReading

    frame, box = frame_with_head(display=True)
    us = LightReading("walk", 0.9, 0.1, 0.25, False, False)
    assert figure_walk_guard(us, frame, box).state == "walk"


def test_light_zoom_env(monkeypatch):
    cfg = DetectorConfig()
    monkeypatch.delenv("STREETSENSE_LIGHT_ZOOM", raising=False)
    assert light_zoom_scales(cfg) == (0.5, 0.3)
    monkeypatch.setenv("STREETSENSE_LIGHT_ZOOM", "0")
    assert light_zoom_scales(cfg) == ()
    monkeypatch.setenv("STREETSENSE_LIGHT_ZOOM", "0.4")
    assert light_zoom_scales(cfg) == (0.4,)
    monkeypatch.setenv("STREETSENSE_LIGHT_ZOOM", "junk")
    assert light_zoom_scales(cfg) == ()


def test_zoom_out_canvas_maps_back():
    frame = np.zeros((200, 300, 3), np.uint8)
    frame[50:100, 60:120] = 255
    canvas, ox, oy = zoom_out_canvas(frame, 0.5)
    assert canvas.shape == frame.shape and canvas[0, 0, 0] == 114
    ys, xs = np.nonzero(canvas[..., 0] == 255)
    x1, y1 = (xs.min() - ox) / 0.5, (ys.min() - oy) / 0.5
    assert abs(x1 - 60) <= 2 and abs(y1 - 50) <= 2


class _FakeDet:
    def __init__(self, dets, zoomed):
        self.dets, self.zoomed, self.calls = dets, zoomed, 0

    def __call__(self, frame):
        return list(self.dets)

    def zoomed_lights(self, frame, scales, conf):
        self.calls += 1
        return list(self.zoomed)




@pytest.mark.parametrize("conf,expect_calls", [(0.9, 0), (0.2, 1)])
def test_pipeline_zoom_runs_only_without_confident_light(monkeypatch, conf, expect_calls):
    import streetsense.pipeline as pl
    from streetsense.config import Config
    from streetsense.types import Detection

    light = Detection("traffic light", Box(10, 10, 40, 100), conf)
    zoomed = [Detection("traffic light", Box(300, 100, 420, 340), 0.8, None, {"zoom": 0.5})]
    fake = _FakeDet([light], zoomed)
    monkeypatch.setattr(pl, "build_detector", lambda cfg: fake)
    cfg = Config()
    cfg.vlm.enabled = False
    cfg.depth.enabled = False
    pipe = pl.Pipeline(cfg, use_surface=False)
    frame, _ = frame_with_head(display=False)
    out = pipe.process(frame, annotate=False, t=0.0)
    assert fake.calls == expect_calls
    lights = [d for d in out.detections if d.label == "traffic light"]
    if expect_calls:
        assert [d.box for d in lights] == [zoomed[0].box]
        assert lights[0].attrs["light_raw"] == "walk"
    else:
        assert lights[0].box == light.box


def test_pipeline_never_walks_on_single_frame():
    """The zoomed figure walk still goes through the 3-of-5 smoother."""
    import streetsense.pipeline as pl
    from streetsense.config import Config
    from streetsense.types import Detection

    fake = _FakeDet([], [Detection("traffic light", Box(300, 100, 420, 340), 0.8)])
    cfg = Config()
    cfg.vlm.enabled = False
    cfg.depth.enabled = False
    orig = pl.build_detector
    pl.build_detector = lambda c: fake
    try:
        pipe = pl.Pipeline(cfg, use_surface=False)
    finally:
        pl.build_detector = orig
    frame, _ = frame_with_head(display=False)
    states = [pipe.process(frame, annotate=False, t=0.25 * i).detections[0].attrs["light_state"] for i in range(3)]
    assert states[:2] == ["unknown", "unknown"] and states[2] == "walk"
