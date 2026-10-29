import threading
import time
from types import SimpleNamespace

import numpy as np

from streetsense.sam_fallback import SamFallbackConfig, SamSignalFallback, SignalBoxes, to_frame_boxes


def _frame(h=1000, w=800):
    return np.zeros((h, w, 3), np.uint8)


def test_to_frame_boxes_rescales_request_coords():
    # crop 600x800 (HxW) sent at 768 longest side -> 576x768 request
    b = [{"x1": 76.8, "y1": 57.6, "x2": 153.6, "y2": 115.2, "w": 768, "h": 576, "c": 0.9}]
    (x1, y1, x2, y2, c), = to_frame_boxes(b, (600, 800, 3), 768)
    assert abs(x1 - 80) < 1e-6 and abs(y1 - 60) < 1e-6 and abs(x2 - 160) < 1e-6 and abs(y2 - 120) < 1e-6 and c == 0.9


def test_to_frame_boxes_drops_garbage_and_clamps():
    b = [{"x1": "bad"}, {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "w": 800, "h": 600}, {"x1": -5, "y1": 2, "x2": 900, "y2": 50, "w": 800, "h": 600}]
    out = to_frame_boxes(b, (600, 800, 3), 768)
    assert out == [(0.0, 2.0, 800.0, 50.0, 1.0)]


def test_run_sync_sends_top_crop_and_filters_conf():
    seen = {}

    def seg(img, prompt):
        seen["shape"] = img.shape
        seen["prompt"] = prompt
        return SimpleNamespace(boxes=[{"x1": 10, "y1": 10, "x2": 20, "y2": 30, "w": 800, "h": 600, "c": 0.8},
                                      {"x1": 40, "y1": 10, "x2": 50, "y2": 30, "w": 800, "h": 600, "c": 0.1}])

    fb = SamSignalFallback(seg, SamFallbackConfig(top_fraction=0.6))
    r = fb.run_sync(_frame(), capture_ts=5.0)
    assert seen["shape"][0] == 600 and seen["prompt"] == "traffic light"
    assert len(r.boxes) == 1 and r.error is None


def test_result_has_no_crossing_authority():
    fields = set(SignalBoxes.__dataclass_fields__)
    assert not fields & {"walk", "go", "cross", "crossing", "state", "decision", "safe"}
    assert not any(hasattr(SamSignalFallback, a) for a in ("decide", "can_cross", "crossing_state"))


def test_api_error_degrades_to_no_boxes():
    def seg(img, prompt):
        raise RuntimeError("budget")

    r = SamSignalFallback(seg).run_sync(_frame(), 1.0)
    assert r.boxes == () and r.error == "RuntimeError" and not r.fresh(1.0, 10)


def test_async_rate_limit_busy_and_skip_when_confident():
    gate = threading.Event()
    calls = []

    def seg(img, prompt):
        calls.append(1)
        gate.wait(2)
        return SimpleNamespace(boxes=[{"x1": 1, "y1": 1, "x2": 9, "y2": 9, "w": 800, "h": 600, "c": 0.9}])

    t = [100.0]
    fb = SamSignalFallback(seg, SamFallbackConfig(min_interval_s=1.0, max_age_s=1.5), clock=lambda: t[0])
    assert not fb.maybe_request(_frame(), 100.0, have_confident_light=True)
    assert fb.maybe_request(_frame(), 100.0, have_confident_light=False)
    assert not fb.maybe_request(_frame(), 100.0, have_confident_light=False)  # busy
    gate.set()
    for _ in range(100):
        if fb.latest() is not None:
            break
        time.sleep(0.01)
    assert fb.latest() is not None and len(fb.latest().boxes) == 1
    t[0] = 100.5
    assert not fb.maybe_request(_frame(), 100.5, have_confident_light=False)  # rate limited
    t[0] = 102.0
    assert fb.latest() is None  # stale -> dropped
    assert len(calls) == 1
