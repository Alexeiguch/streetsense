"""Refined boxes are candidates with current pixels, a rate cap and parent vetoes."""
from types import SimpleNamespace

import numpy as np
import pytest

from streetsense.detector import SignalHeadRefiner, focused_detections, focused_tile_window
from streetsense.traffic_light import LightReading, LightStateSmoother
from streetsense.types import Box, Detection


def scene():
    frame = np.zeros((360, 640, 3), np.uint8)
    frame[150:200, 280:320] = (100, 240, 0)
    return frame, Detection('traffic light', Box(200, 40, 420, 340), .9, 7)


def child(parent):
    return Detection('traffic light', Box(240, 100, 320, 220), .8, None,
                     {'source': 'signal_refine', 'refine_parent': list(parent.box.__dict__.values())})


def test_refinement_uses_one_tile_and_preserves_parent():
    frame, parent = scene()
    (x, y, _, _), _ = focused_tile_window(frame, [parent])
    calls = []

    def predict(tile):
        calls.append(tile)
        return [([240-x, 100-y, 320-x, 220-y], 'traffic light', .8),
                ([240-x, 100-y, 320-x, 220-y], 'person', .99),
                ([240-x, 100-y, 320-x, 220-y], 'traffic light', .4),
                ([500-x, 100-y, 550-x, 220-y], 'traffic light', .99)]

    result = focused_detections(frame, [parent], .5, predict)
    assert len(calls) == len(result) == 1
    assert result[0].box == Box(240, 100, 320, 220)
    assert result[0].track_id is None
    assert result[0].attrs['refine_parent'] == [200, 40, 420, 340]
    assert parent.box == Box(200, 40, 420, 340) and parent.track_id == 7


def test_refinement_does_not_duplicate_existing_local_head():
    frame, parent = scene()
    existing = child(parent)
    (x, y, _, _), _ = focused_tile_window(frame, [parent])
    predict = lambda tile: [([240-x, 100-y, 320-x, 220-y], 'traffic light', .95)]
    assert focused_detections(frame, [parent, existing], .5, predict) == []


def test_no_head_or_no_lit_pixels_skips_extra_inference():
    frame, parent = scene()
    assert focused_tile_window(frame, []) is None
    assert focused_tile_window(np.zeros_like(frame), [parent]) is None


@pytest.mark.parametrize('box', [Box(280, 100, 320, 190), Box(280, 0, 410, 360)])
def test_ordinary_small_or_vehicle_shaped_heads_skip_extra_inference(box):
    frame, _ = scene()
    calls = []
    detector = SimpleNamespace(focused_lights=lambda *a: calls.append(a) or [])
    head = Detection('traffic light', box, .9, 1)
    assert SignalHeadRefiner(min_width_fraction=.2).update(frame, 0, [head], detector) == []
    assert not calls


def test_refiner_caps_calls_and_reuses_only_coordinates():
    frame, parent = scene()
    calls = []
    detector = SimpleNamespace(focused_lights=lambda *a: calls.append(a) or [child(parent)])
    refiner = SignalHeadRefiner(.25)
    first = refiner.update(frame, 0, [parent], detector)
    cached = refiner.update(frame, .1, [parent], detector)
    refiner.update(frame, .2, [parent], detector)
    refiner.update(frame, .3, [parent], detector)
    assert len(calls) == 2
    assert cached[0].attrs['refine_age_s'] == pytest.approx(.1)
    assert 'light_state' not in cached[0].attrs
    assert first[0].attrs['refine_age_s'] == 0
    assert first[0].track_id < 0
    assert first[0].track_id == cached[0].track_id


def test_refined_identity_persists_on_refresh_but_not_after_parent_disappears():
    frame, parent = scene()
    detector = SimpleNamespace(focused_lights=lambda *a: [child(parent)])
    refiner = SignalHeadRefiner(.25)
    first = refiner.update(frame, 0, [parent], detector)[0].track_id
    assert refiner.update(frame, .3, [parent], detector)[0].track_id == first
    assert refiner.update(frame, .4, [], detector) == []
    assert refiner.update(frame, .6, [parent], detector)[0].track_id != first


def test_parent_motion_discards_cache_without_bypassing_rate_cap():
    frame, parent = scene()
    calls = []
    detector = SimpleNamespace(focused_lights=lambda *a: calls.append(a) or [child(parent)])
    refiner = SignalHeadRefiner(.25)
    assert refiner.update(frame, 0, [parent], detector)
    moved = Detection('traffic light', Box(280, 40, 500, 340), .9, 7)
    assert refiner.update(frame, .1, [moved], detector) == []
    assert len(calls) == 1


def test_refinement_reset_and_rewind_discard_coordinates():
    frame, parent = scene()
    detector = SimpleNamespace(focused_lights=lambda *a: [child(parent)])
    refiner = SignalHeadRefiner(.25)
    assert refiner.update(frame, 10, [parent], detector)
    assert refiner.update(frame, 1, [parent], detector)[0].attrs['refine_age_s'] == 0
    refiner.reset()
    assert refiner.boxes == []


def test_refinement_failure_drops_old_boxes_and_keeps_base_reader_available():
    frame, parent = scene()
    def failed(*args):
        raise RuntimeError('inference unavailable')
    refiner = SignalHeadRefiner(.25)
    assert refiner.update(frame, 0, [parent], SimpleNamespace(focused_lights=lambda *a: [child(parent)]))
    assert refiner.update(frame, .3, [parent], SimpleNamespace(focused_lights=failed)) == []
    assert refiner.error == 'RuntimeError'


@pytest.mark.parametrize('track_id', [None, 5])
def test_parent_veto_cancels_walk_and_requires_fresh_votes(track_id):
    smoother = LightStateSmoother()
    box = Box(10, 10, 30, 70)
    for i in range(3):
        state = smoother.update(track_id, 'walk', box, t=i * .1)
    assert state == 'walk'
    smoother.discard_walk(track_id, box)
    assert smoother.update(track_id, 'unknown', box, t=.3) != 'walk'
    assert smoother.update(track_id, 'walk', box, t=.4) != 'walk'
    assert smoother.update(track_id, 'walk', box, t=.5) != 'walk'
    assert smoother.update(track_id, 'walk', box, t=.6) == 'walk'


@pytest.mark.parametrize('negative', ['red', 'yellow', 'countdown'])
def test_pipeline_parent_negative_evidence_blocks_refined_child(monkeypatch, negative):
    from streetsense import pipeline as mod
    from streetsense.config import Config

    frame, parent = scene()
    now = [0.]
    parent_state = ['green']

    class Detector:
        def __call__(self, frame):
            return [Detection(parent.label, parent.box, parent.conf, parent.track_id)]

        def focused_lights(self, *args):
            return [child(parent)]

    monkeypatch.setattr(mod, 'build_detector', lambda cfg: Detector())
    monkeypatch.setattr(mod.signal_cnn, 'get', lambda: None)
    monkeypatch.setattr(mod, 'classify_light', lambda crop, kind: LightReading(
        parent_state[0] if crop.shape[0] > 200 else 'walk', .9, .1, .75))
    cfg = Config()
    cfg.reader.country = 'GB'
    cfg.vlm.enabled = False
    pipe = mod.Pipeline(cfg, use_surface=False, clock=lambda: now[0])
    for i in range(3):
        now[0] = i * .1
        out = pipe.process(frame, annotate=False)
    assert out.detections[-1].attrs['light_state'] == 'walk'
    parent_state[0] = negative
    now[0] = .3
    out = pipe.process(frame, annotate=False)
    assert out.detections[-1].attrs['light_state'] != 'walk'
    assert out.detections[-1].attrs['light_parent_veto'] == negative
