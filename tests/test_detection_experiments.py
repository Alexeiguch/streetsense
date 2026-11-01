"""Integrity checks for the experiment, independent of inference weights."""
from types import SimpleNamespace

import numpy as np

from scripts.experiment_light_detection import PairedDetector, roi_windows
from streetsense.types import Box, Detection


def light(box, conf=.9, **attrs):
    return Detection('traffic light', Box(*box), conf, None, attrs)


class FakeDetector:
    backend = 'openvino'
    model_path = 'test'
    model = SimpleNamespace(predictor=None)
    cfg = SimpleNamespace(light_zoom_scales=(.5, .3), light_zoom_conf=.35)

    def __init__(self, detections):
        self.detections = detections
        self.calls = self.zoom_calls = 0

    def __call__(self, img):
        self.calls += 1
        return self.detections

    def zoomed_lights(self, img, scales, conf):
        self.zoom_calls += 1
        return []


def test_common_detections_are_reused_without_reader_attribute_leakage():
    base = FakeDetector([light((10, 10, 90, 90))])
    detector = PairedDetector(base)
    img = np.zeros((100, 100, 3), np.uint8)
    detector(img)[0].attrs['light_state'] = 'walk'
    second = detector(img)
    assert base.calls == 1
    assert 'light_state' not in second[0].attrs
    assert 'light_state' not in base.detections[0].attrs
    detector.reset()
    detector(img)
    assert base.calls == 2


def test_baseline_keeps_original_overlapping_boxes_and_order():
    # A benchmark baseline must not add NMS that production did not run.
    base = FakeDetector([light((10, 10, 90, 90), .7), light((11, 11, 89, 89), .9)])
    out = PairedDetector(base)(np.zeros((100, 100, 3), np.uint8))
    assert [d.conf for d in out] == [.7, .9]


def test_reference_tiles_replace_zoom_as_in_production():
    base = FakeDetector([])
    detector = PairedDetector(base)
    detector.method = 'tiles2x3'
    assert detector.zoomed_lights(np.zeros((100, 100, 3), np.uint8), (.5, .3), .35) == []
    assert base.zoom_calls == 0


def test_targeted_tile_keeps_parent_even_when_child_has_higher_confidence():
    parent = light((5, 5, 95, 95), .6, light_raw='red')
    base = FakeDetector([parent])
    detector = PairedDetector(base)
    detector.method = 'saliency_tile'
    img = np.zeros((100, 100, 3), np.uint8)
    img[65:85, 65:85] = (0, 0, 255)
    seen = []

    def predict(img, source, offset):
        seen.append((img.shape, offset))
        return [light((60, 60, 90, 90), .99), light((6, 6, 94, 94), .99)]

    detector.predict_lights = predict
    out = detector(img)
    assert len(seen) == 1
    assert len(out) == 2  # overlapping replacement rejected; smaller child retained
    assert out[0].attrs['light_raw'] == 'red'
    assert out[0].box == parent.box


def test_targeted_tile_does_not_run_without_a_head_or_lit_pixels():
    for detections in ([], [light((10, 10, 90, 90))]):
        detector = PairedDetector(FakeDetector(detections))
        detector.method = 'saliency_tile'
        def no_predict(*args, **kwargs):
            raise AssertionError('unexpected detector pass')
        detector.predict_lights = no_predict
        assert len(detector(np.zeros((100, 100, 3), np.uint8))) == len(detections)


def test_context_regions_are_clipped_deduplicated_and_bounded():
    lights = [light((-5, -5, 20, 20)), light((-5, -5, 20, 20)),
              light((180, 130, 200, 150)), light((90, 60, 110, 90))]
    windows = roi_windows(lights, 200, 150)
    assert len(windows) == 2
    assert all(0 <= x1 < x2 <= 200 and 0 <= y1 < y2 <= 150 for x1, y1, x2, y2 in windows)
