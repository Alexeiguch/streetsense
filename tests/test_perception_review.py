"""November review regressions: temporal freshness, tracker recovery and segmentation cache."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from streetsense.config import Config, DetectorConfig
from streetsense.detector import ApproachTracker, YoloDetector, nms_lights
from streetsense.meta_api import SceneReport
from streetsense.sam_fallback import SamFallbackConfig, SamSignalFallback, SignalBoxes
from streetsense.surface import summarise_labels
from streetsense.types import Box, Detection


@pytest.fixture
def fake_pipeline(monkeypatch):
    from streetsense import pipeline as mod

    monkeypatch.setenv('STREETSENSE_SIGNAL_CNN', '0')
    monkeypatch.setenv('STREETSENSE_LIGHT_ZOOM', '0')
    monkeypatch.delenv('MODEL_API_KEY', raising=False)
    monkeypatch.delenv('LLAMA_API_KEY', raising=False)

    class Detector:
        def __call__(self, frame):
            return []

    class Surface:
        calls = 0

        def __call__(self, frame):
            self.calls += 1
            return summarise_labels(np.zeros(frame.shape[:2], np.uint8))

    monkeypatch.setattr(mod, 'build_detector', lambda cfg: Detector())
    monkeypatch.setattr('streetsense.surface.SurfaceSegmenter', lambda *a, **k: Surface())
    clock = [0.0]
    pipe = mod.Pipeline(Config(), clock=lambda: clock[0])
    return pipe, clock


@pytest.mark.parametrize('tracked,track_conf,expected', [(True, .1, .1), (False, .1, .28), (True, None, .28)])
def test_bytetrack_sees_weak_detections_without_lowering_alert_thresholds(monkeypatch, tracked, track_conf, expected):
    calls = []

    class Model:
        names = {0: 'person'}

        def __init__(self, path):
            pass

        def track(self, frame, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=None)]

        predict = track

    monkeypatch.setattr('ultralytics.YOLO', Model)
    monkeypatch.setattr('streetsense.detector.ensure_pt_weights', lambda p: p)
    detector = YoloDetector(DetectorConfig(backend='torch', track=tracked, track_conf=track_conf))
    detector(np.zeros((40, 60, 3), np.uint8))
    assert calls[0]['conf'] == pytest.approx(expected)
    assert not detector._accept('person', .15)
    assert not detector._accept('car', .3)


def test_stronger_tiled_box_retains_matching_track_identity():
    local = Detection('traffic light', Box(10, 10, 40, 80), .6, 42)
    tile = Detection('traffic light', Box(11, 10, 40, 79), .9, attrs={'tile': 1})
    result, = nms_lights([local, tile])
    assert result.box == tile.box and result.conf == tile.conf
    assert result.track_id == 42
    assert tile.track_id is None  # input detections remain unchanged


def car(height, label='car'):
    return Detection(label, Box(20, 20, 60, 20 + height), .9, 7)


def test_old_track_must_not_create_false_approach_on_reacquisition():
    tracker = ApproachTracker()
    tracker.update(0, [car(40)])
    tracker.update(8, [])
    assert tracker.update(8.1, [car(70)]) == set()


def test_approach_recovers_after_clock_rewind():
    tracker = ApproachTracker()
    tracker.update(100, [car(200)])
    tracker.update(0, [car(40)])
    assert tracker.update(1, [car(60)]) == {7}


def test_approach_works_above_128_fps():
    tracker = ApproachTracker()
    for i in range(241):
        t = i / 200
        result = tracker.update(t, [car(100 * (1 + .7 * t))])
    assert result == {7}


def test_reused_track_id_cannot_compare_different_classes():
    tracker = ApproachTracker()
    tracker.update(0, [car(40)])
    assert tracker.update(1, [car(100, label='person')]) == set()


def test_surface_refreshes_when_camera_resolution_changes(fake_pipeline):
    pipe, clock = fake_pipeline
    pipe.process(np.zeros((120, 200, 3), np.uint8), annotate=False)
    clock[0] = .1
    out = pipe.process(np.zeros((200, 120, 3), np.uint8), annotate=False)
    assert out.surface.labels.shape == (200, 120)
    assert pipe.surface_seg.calls == 2


def test_surface_cache_expires_on_slow_camera(fake_pipeline):
    pipe, clock = fake_pipeline
    frame = np.zeros((120, 200, 3), np.uint8)
    pipe.process(frame, annotate=False)
    clock[0] = .8
    pipe.process(frame, annotate=False)
    assert pipe.surface_seg.calls == 2


def test_surface_cache_is_reused_at_normal_camera_rate(fake_pipeline):
    pipe, clock = fake_pipeline
    frame = np.zeros((120, 200, 3), np.uint8)
    for i in range(6):
        clock[0] = i / 30
        pipe.process(frame, annotate=False)
    assert pipe.surface_seg.calls == 1


def test_surface_refreshes_on_video_rewind(fake_pipeline):
    pipe, clock = fake_pipeline
    frame = np.zeros((120, 200, 3), np.uint8)
    clock[0] = 10
    pipe.process(frame, annotate=False)
    clock[0] = 1
    pipe.process(frame, annotate=False)
    assert pipe.surface_seg.calls == 2


def test_sam_cannot_reuse_result_from_a_future_video_timestamp():
    assert not SignalBoxes(capture_ts=10).fresh(1, 1.5)


def test_sam_preview_scales_request_coordinates_and_preserves_input():
    from streetsense.meta_api import SamSegment, overlay_sam_segments

    frame = np.zeros((100, 200, 3), np.uint8)
    seg = SamSegment('crosswalk', boxes=[{'id': 1, 'x1': 20, 'y1': 20, 'x2': 40, 'y2': 40,
                                          'w': 100, 'h': 50}],
                     masks=[{'id': 1, 'bounds': (20, 20, 40, 40),
                             'raster': np.ones((20, 20), np.uint8)}])
    out = overlay_sam_segments(frame, [seg])
    assert out[60, 60].any()  # scaled mask interior
    assert out[80, 60, 1] == 255  # scaled box edge stays opaque
    assert not out[30, 30].any()  # unscaled location stays clear
    assert not frame.any()


def test_sam_preview_clips_mask_at_frame_edge():
    from streetsense.meta_api import SamSegment, overlay_sam_segments

    seg = SamSegment('curb', masks=[{'id': 1, 'bounds': (-5, -5, 5, 5),
                                   'raster': np.ones((10, 10), np.uint8)}])
    out = overlay_sam_segments(np.zeros((10, 10, 3), np.uint8), [seg])
    assert out[0, 0].any()
    assert not out[8, 8].any()


def test_background_sam_owns_an_immutable_frame_snapshot():
    gate, done = threading.Event(), threading.Event()
    means = []

    def segment(frame, prompt):
        gate.wait(2)
        means.append(float(frame.mean()))
        done.set()
        return SimpleNamespace(boxes=[])

    frame = np.zeros((20, 30, 3), np.uint8)
    fallback = SamSignalFallback(segment, SamFallbackConfig(), clock=lambda: 1)
    assert fallback.maybe_request(frame, 1, have_confident_light=False)
    frame[:] = 255  # caller reuses the capture buffer while SAM is running
    gate.set()
    assert done.wait(2)
    assert means == [0.0]


def test_background_sam_discards_inflight_result_after_scene_reset():
    gate = threading.Event()

    def segment(frame, prompt):
        gate.wait(2)
        return SimpleNamespace(boxes=[{'x1': 1, 'y1': 1, 'x2': 10, 'y2': 10,
                                       'w': 30, 'h': 20, 'c': .9}])

    fallback = SamSignalFallback(segment, SamFallbackConfig(), clock=lambda: 1)
    assert fallback.maybe_request(np.zeros((20, 30, 3), np.uint8), 1, have_confident_light=False)
    fallback.invalidate()
    gate.set()
    deadline = time.monotonic() + 2
    while fallback._busy and time.monotonic() < deadline:
        time.sleep(.001)
    assert not fallback._busy
    assert fallback.latest(1) is None


def test_sam_thread_start_failure_does_not_leave_worker_busy(monkeypatch):
    def failed_start(self):
        raise RuntimeError('thread unavailable')

    monkeypatch.setattr(threading.Thread, 'start', failed_start)
    fallback = SamSignalFallback(lambda *args: None, SamFallbackConfig(), clock=lambda: 1)
    assert not fallback.maybe_request(np.zeros((20, 30, 3), np.uint8), 1, have_confident_light=False)
    assert not fallback._busy


def report(capture_ts, hazards):
    data = {'summary': 'Street scene', 'hazards': hazards, 'crossing_visible': False,
            'pedestrian_signal': 'none_visible', 'safe_to_cross': 'unsure'}
    value = SceneReport(time.time(), .5, data, 'q')
    value.capture_ts = capture_ts
    return value


def test_stale_meta_hazards_are_never_requeued(fake_pipeline):
    pipe, clock = fake_pipeline
    pipe.vlm.latest = report(0, ['wet leaves ahead'])
    clock[0] = 10
    pipe.process(np.zeros((120, 200, 3), np.uint8), annotate=False)
    assert not pipe.alerts.pending


def test_queued_meta_hazard_expires_from_capture_time(fake_pipeline):
    pipe, clock = fake_pipeline
    pipe.vlm.latest = report(0, ['wet leaves ahead'])
    frame = np.zeros((120, 200, 3), np.uint8)
    clock[0] = 2.5
    pipe.process(frame, annotate=False)
    assert any('wet leaves' in a.text for a in pipe.alerts.pending)
    clock[0] = 3.1  # queue is only .6 seconds old, but source image is too old
    pipe.process(frame, annotate=False)
    assert not pipe.alerts.pending


@pytest.mark.parametrize('phrase', ['Go.', 'Safe to cross now.', 'The walk signal is green.'])
def test_background_meta_cannot_bypass_dual_crossing_check(fake_pipeline, phrase):
    pipe, clock = fake_pipeline
    pipe.vlm.latest = report(0, [phrase])
    pipe.process(np.zeros((120, 200, 3), np.uint8), annotate=False)
    assert not pipe.alerts.pending


def test_fresh_meta_hazard_is_still_available(fake_pipeline):
    pipe, clock = fake_pipeline
    pipe.vlm.latest = report(0, ['wet leaves ahead'])
    frame = np.zeros((120, 200, 3), np.uint8)
    first = pipe.process(frame, annotate=False)
    clock[0] = 2.0  # allow the initial local warning's speech gap to finish
    second = pipe.process(frame, annotate=False)
    assert any(o.spoken and 'wet leaves' in o.spoken.text.lower() for o in [first, second])


def test_geometry_change_discards_old_walk_votes(fake_pipeline, monkeypatch):
    from streetsense.traffic_light import LightReading

    pipe, clock = fake_pipeline
    pipe.detector = lambda frame: [Detection('traffic light', Box(25, 10, 45, 70), .9, 1)]
    monkeypatch.setattr('streetsense.pipeline.classify_light', lambda *a: LightReading('walk', .9, .1, .2))
    frame = np.zeros((120, 200, 3), np.uint8)
    for i in range(3):
        clock[0] = i * .1
        out = pipe.process(frame, annotate=False)
    assert out.detections[0].attrs['light_state'] == 'walk'
    clock[0] = .3
    changed = pipe.process(np.zeros((200, 120, 3), np.uint8), annotate=False)
    assert changed.detections[0].attrs['light_state'] == 'unknown'


def test_meta_capture_age_and_pending_result_invalidation(monkeypatch):
    from streetsense.config import VisionLLMConfig
    from streetsense.meta_api import MetaVisionAssistant

    monkeypatch.setenv('MODEL_API_KEY', 'test-key')
    gate, done = threading.Event(), threading.Event()
    assistant = MetaVisionAssistant(VisionLLMConfig())

    def reply(*args):
        gate.wait(2)
        done.set()
        return report(0, []).data

    monkeypatch.setattr(assistant, '_call', reply)
    assert assistant.maybe_request(np.zeros((30, 40, 3), np.uint8), now=10)
    assistant.reset()
    gate.set()
    assert done.wait(2)
    deadline = time.monotonic() + 2
    while assistant._busy.locked() and time.monotonic() < deadline:
        time.sleep(.001)
    assert assistant.latest is None
    assert not report(10, []).fresh(9, 3)
    assert report(10, []).fresh(12, 3)
    assert not report(10, []).fresh(14, 3)


def test_meta_encoding_failure_does_not_permanently_lock_worker(monkeypatch):
    from streetsense.config import VisionLLMConfig
    from streetsense.meta_api import MetaVisionAssistant

    monkeypatch.setenv('MODEL_API_KEY', 'test-key')
    assistant = MetaVisionAssistant(VisionLLMConfig())
    monkeypatch.setattr('streetsense.meta_api.encode_jpeg_data_url', lambda *a: (_ for _ in ()).throw(ValueError('bad frame')))
    assert not assistant.maybe_request(np.zeros((10, 10, 3), np.uint8), now=1)
    assert not assistant._busy.locked()


def test_shared_signal_classifier_serializes_inference(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from streetsense.signal_cnn import SignalCNN

    class Compiled:
        active = 0
        max_active = 0

        def output(self, n):
            return 0

        def __call__(self, args):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            time.sleep(.01)
            self.active -= 1
            return {0: np.zeros((1, 4))}

    compiled = Compiled()
    core = SimpleNamespace(read_model=lambda *a: None, compile_model=lambda *a: compiled)
    monkeypatch.setattr('openvino.Core', lambda: core)
    classifier = SignalCNN()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(classifier, [np.zeros((20, 20, 3), np.uint8)] * 8))
    assert compiled.max_active == 1
    assert all(sum(r.probs) == pytest.approx(1) for r in results)
