"""Fusion safety rule + SAM helpers (no network)."""

import numpy as np

from streetsense.meta_api import (
    fuse_crossing_decision,
    parse_sam_result,
    sam_boxes_to_crosswalk,
)


def test_fuse_never_go_without_dual_yes():
    assert fuse_crossing_decision("walk", {"safe_to_cross": "yes", "pedestrian_signal": "green"})["decision"] == "go"
    assert fuse_crossing_decision("red", {"safe_to_cross": "yes", "pedestrian_signal": "green"})["decision"] != "go"
    assert fuse_crossing_decision("walk", {"safe_to_cross": "unsure", "pedestrian_signal": "green"})["decision"] != "go"
    assert fuse_crossing_decision("walk", {"safe_to_cross": "no", "pedestrian_signal": "red"})["decision"] == "stop"
    # vehicle green is not a ped walk
    assert fuse_crossing_decision("green", {"safe_to_cross": "yes", "pedestrian_signal": "green"})["decision"] != "go"


def test_sam_boxes_to_crosswalk_scales():
    boxes = [{"x1": 0, "y1": 100, "x2": 200, "y2": 180, "w": 640, "h": 360, "c": 0.8}]
    # frame is 1280x720 -> 2x
    cw = sam_boxes_to_crosswalk(boxes, (720, 1280, 3))
    assert cw is not None and cw.found
    assert cw.box.x2 == 400 and cw.box.y1 == 200
    assert cw.score == 0.8


def test_parse_sam_result_boxes_without_parser_crash():
    line = "<0f>0<|box;x1=10;y1=20;x2=30;y2=40;w=100;h=80|><|mask;x=0;y=0;data=1,1,!00000|>"
    out = parse_sam_result(line)
    assert out["boxes"][0]["x1"] == 10
