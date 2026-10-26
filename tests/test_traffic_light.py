from pathlib import Path

import cv2
import numpy as np
import pytest

from streetsense.traffic_light import LightStateSmoother, classify_light, signal_kind

FIX = Path(__file__).parent / "fixtures"


def housing(lit_row: int | None, bgr=(0, 0, 255), h=90, w=36) -> np.ndarray:
    """Synthetic 3-lamp head: dark grey housing, one lit lamp at row 0/1/2."""
    img = np.full((h, w, 3), 40, np.uint8)
    for row in range(3):
        cy = int(h * (row + 0.5) / 3)
        col = bgr if row == lit_row else (25, 25, 25)
        cv2.circle(img, (w // 2, cy), w // 3, col, -1)
    return img


@pytest.mark.parametrize("row,bgr,expected", [
    (0, (0, 0, 255), "red"),
    (1, (0, 210, 255), "yellow"),
    (2, (120, 230, 0), "green"),   # cyan-ish LED green
    (2, (0, 255, 0), "green"),
])
def test_synthetic_lamps(row, bgr, expected):
    r = classify_light(housing(row, bgr), kind="vehicle")
    assert r.state == expected
    assert r.lamp_position is not None and abs(r.lamp_position - (row + 0.5) / 3) < 0.15


def test_all_dark_is_unknown():
    assert classify_light(housing(None), kind="vehicle").state == "unknown"


def test_empty_crop_is_unknown():
    assert classify_light(np.zeros((0, 0, 3), np.uint8)).state == "unknown"


def test_orange_depends_on_kind():
    orange = (0, 110, 255)  # hue ~13
    assert classify_light(housing(1, orange), kind="vehicle").state == "yellow"
    # on a pedestrian head, orange is the "don't walk" hand: must read as red (wait), never go
    assert classify_light(housing(0, orange), kind="pedestrian").state == "red"
    # unknown kind: amber in the middle lamp is a vehicle amber
    assert classify_light(housing(1, orange), kind="unknown").state == "yellow"


def test_white_sky_on_border_is_not_walk():
    img = housing(2, (0, 255, 0))
    img[:, :8] = 255  # sky leaking down the left side of a loose box
    assert classify_light(img, kind="unknown").state != "walk"


@pytest.mark.parametrize("path", sorted(FIX.glob("*.png")), ids=lambda p: p.stem)
def test_real_crops(path):
    """Real crops (DC: CC BY 3.0; Tyumen: CC BY-SA 4.0, Wikimedia), labelled by eye; suffix = expected."""
    expected = path.stem.split("_")[-1]
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    r = classify_light(img, kind=signal_kind(w, h))
    assert r.state == expected


def test_walk_with_countdown_flag():
    img = cv2.imread(str(FIX / "dc_t10_pedestrian_walk.png"))
    assert classify_light(img).countdown is True


def test_signal_kind_extremes():
    assert signal_kind(30, 90) == "vehicle"
    assert signal_kind(40, 45) == "pedestrian"
    assert signal_kind(40, 88) == "unknown"


def test_smoother_needs_votes_and_resists_flicker():
    s = LightStateSmoother(window=5, min_votes=3)
    assert s.update(1, "green") == "unknown"  # brand-new track: not announced yet
    assert s.update(1, "green") == "unknown"
    assert s.update(1, "green") == "green"
    assert s.update(1, "red") == "green"  # one odd frame does not flip it
    assert s.update(1, "red") == "green"
    assert s.update(1, "red") == "red"
    assert s.update(None, "red") == "red"  # untracked: pass-through


def test_bright_screen_is_not_a_light():
    """Regression: an advertising screen detected as 'traffic light' (Tyumen clip) read as green/red."""
    screen = np.full((90, 90, 3), (60, 170, 90), np.uint8)  # bright green-ish panel, no dark housing
    screen[:6] = 250  # white frame
    assert classify_light(screen).state == "unknown"


def _ped_head(top=None, bottom=None):
    img = np.full((120, 60, 3), 25, np.uint8)
    if top == "white":
        img[15:50, 18:42] = (235, 235, 235)
    if top == "orange":
        img[15:50, 18:42] = (0, 110, 255)
    if bottom == "orange":
        img[75:105, 15:45] = (0, 110, 255)
    return img


def test_countdown_only_is_not_red():
    r = classify_light(_ped_head(bottom="orange"), "pedestrian")
    assert r.state == "countdown" and r.countdown
    assert classify_light(_ped_head(top="orange"), "pedestrian").state == "red"
    assert classify_light(_ped_head(top="white", bottom="orange"), "pedestrian").state == "walk"


def test_countdown_frames_hold_walk_and_hand_still_flips():
    s = LightStateSmoother()
    for st in ["walk", "walk", "walk", "countdown", "countdown", "countdown", "countdown", "walk", "countdown"]:
        out = s.update(1, st)
    assert out == "walk"
    for st in ["red", "countdown", "red", "red"]:
        out = s.update(1, st)
    assert out == "red"
