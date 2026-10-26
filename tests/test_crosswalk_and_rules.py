import cv2
import numpy as np

from streetsense.crosswalk import CrosswalkHysteresis, CrosswalkResult, detect_crosswalk
from streetsense.detector import ApproachTracker
from streetsense.meta_api import parse_json_reply, parse_sam_boxes
from streetsense.rules import FrameFacts, build_alerts
from streetsense.surface import POLE, ROAD, SIDEWALK, summarise_labels
from streetsense.types import Box, Detection, Priority


def road(h=720, w=1280):
    img = np.full((h, w, 3), 70, np.uint8)  # grey asphalt
    noise = np.random.default_rng(0).integers(-8, 8, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def zebra(n=6, angle_deg=0.0):
    img = road()
    h, w = img.shape[:2]
    for i in range(n):  # horizontal bars stacked in the lower half, like a crossing seen ahead
        y = int(h * 0.58 + i * h * 0.065)
        cv2.rectangle(img, (int(w * 0.2), y), (int(w * 0.8), y + int(h * 0.03)), (235, 235, 235), -1)
    if angle_deg:
        M = cv2.getRotationMatrix2D((w / 2, h * 0.75), angle_deg, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(70, 70, 70))
    return img


def test_detects_synthetic_zebra():
    r = detect_crosswalk(zebra())
    assert r.found and r.n_stripes >= 4
    assert r.box.y1 > 720 * 0.45


def test_detects_rotated_zebra():
    assert detect_crosswalk(zebra(angle_deg=20)).found


def test_plain_road_and_single_line_not_crossing():
    assert not detect_crosswalk(road()).found
    img = road()
    cv2.line(img, (100, 650), (1200, 650), (240, 240, 240), 12)  # one lane line
    assert not detect_crosswalk(img).found


def test_ground_mask_suppresses_off_ground_stripes():
    img = zebra()
    sky = np.zeros(img.shape[:2], bool)  # nothing is ground -> stripes are ignored
    assert not detect_crosswalk(img, ground_mask=sky).found
    assert detect_crosswalk(img, ground_mask=~sky).found


def test_hysteresis():
    hy = CrosswalkHysteresis(n=5, k=2)
    yes = CrosswalkResult(True, Box(0, 0, 1, 1), 5, 1.0, [])
    no = CrosswalkResult(False, None, 0, 0.0, [])
    assert not hy.update(yes).found  # one sighting is not enough
    assert hy.update(no).found is False
    assert hy.update(yes).found  # 2 of last 3
    assert hy.update(no).found  # still within window, keeps last box


def test_surface_summary():
    lab = np.full((100, 100), SIDEWALK, np.uint8)
    assert summarise_labels(lab).underfoot == "sidewalk"
    lab[80:, :] = ROAD
    assert summarise_labels(lab).underfoot == "road"
    lab2 = np.full((100, 100), SIDEWALK, np.uint8)
    lab2[30:, 48:52] = POLE
    assert summarise_labels(lab2).pole_ahead


def test_rules_vehicle_close_is_critical():
    W, H = 1280, 720
    car = Detection("car", Box(500, 300, 900, 700), 0.9, track_id=3)
    alerts = build_alerts(FrameFacts(W, H, [car], approaching={3}))
    assert alerts and alerts[0].priority == Priority.CRITICAL
    assert "approaching" in alerts[0].text


def test_rules_far_person_off_path_is_silent():
    p = Detection("person", Box(10, 300, 40, 380), 0.9)
    assert build_alerts(FrameFacts(1280, 720, [p])) == []


def test_rules_person_ahead_distance_text():
    p = Detection("person", Box(560, 150, 720, 700), 0.9)
    (a,) = build_alerts(FrameFacts(1280, 720, [p]))
    assert a.key == "person:ahead" and "metre" in a.text


def test_rules_light_and_crossing():
    tl = Detection("traffic light", Box(600, 50, 630, 120), 0.8, attrs={"light_state": "red", "signal_kind": "pedestrian"})
    cw = CrosswalkResult(True, Box(300, 500, 1000, 700), 5, 1.0, [])
    alerts = {a.key: a for a in build_alerts(FrameFacts(1280, 720, [tl], crosswalk=cw))}
    assert alerts["light:pedestrian:red"].priority == Priority.WARNING
    assert alerts["light:pedestrian:red"].text == "pedestrian light is red"
    assert "crossing" in alerts["crosswalk"].text


def test_unknown_light_not_announced():
    tl = Detection("traffic light", Box(600, 50, 630, 120), 0.8, attrs={"light_state": "unknown"})
    assert build_alerts(FrameFacts(1280, 720, [tl])) == []


def test_approach_tracker():
    tr = ApproachTracker(window_s=1.0, ratio=1.25)
    for i in range(11):
        t = i / 10
        h = 100 * (1 + 0.5 * t)  # grows 50% in 1 s
        got = tr.update(t, [Detection("car", Box(0, 0, 50, h), 0.9, track_id=7)])
    assert 7 in got
    tr2 = ApproachTracker()
    for i in range(11):
        got = tr2.update(i / 10, [Detection("car", Box(0, 0, 50, 100), 0.9, track_id=8)])
    assert 8 not in got


def test_parse_sam_boxes_doc_example():
    # Example line from Meta's "Read SAM segmentation output" docs (mask payload truncated).
    line = ("<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91|>,"
            "1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334;c=0.96|><|mask;x=0;y=0;data=27,48,~!!!!J!0c|>")
    recs = parse_sam_boxes(line)
    assert [r["id"] for r in recs] == [0, 1]
    assert recs[0]["x1"] == 211 and recs[1]["c"] == 0.96 and recs[1]["w"] == 320


def test_parse_json_reply_tolerates_fences():
    assert parse_json_reply('```json\n{"safe_to_cross": "no"}\n```')["safe_to_cross"] == "no"
    assert parse_json_reply("just prose")["summary"] == "just prose"
