"""Focused candidates must not remove or overrule evidence in their parent head."""
from types import SimpleNamespace

import numpy as np
import pytest

from streetsense import detector as d
from streetsense.config import Config
from streetsense.signal_cnn import SignalPrediction
from streetsense.traffic_light import LightReading
from streetsense.types import Box, Detection


def test_focus_is_explicitly_opt_in(monkeypatch):
    monkeypatch.delenv('STREETSENSE_LIGHT_FOCUS', raising=False)
    assert not d.light_focus_enabled(Config().detector)
    for value in ('0', 'off', 'invalid'):
        monkeypatch.setenv('STREETSENSE_LIGHT_FOCUS', value)
        assert not d.light_focus_enabled()
    monkeypatch.setenv('STREETSENSE_LIGHT_FOCUS', '1')
    assert d.light_focus_enabled()


def test_focus_uses_one_tile_ignores_hue_and_preserves_parents():
    frame = np.zeros((200, 300, 3), np.uint8)
    parent = Detection('traffic light', Box(160, 50, 295, 195), .6, 1)
    frame[160:180, 250:270] = (0, 255, 0)
    chosen = d.focused_tile_window(frame, [parent])
    frame[160:180, 250:270] = (0, 0, 255)
    assert d.focused_tile_window(frame, [parent]) == chosen
    (x, y, w, h), _ = chosen
    calls = []

    def predict(tile):
        calls.append(tile.shape)
        return [((250-x, 150-y, 280-x, 190-y), 'traffic light', .99),
                ((160-x, 50-y, 295-x, 195-y), 'traffic light', .99),  # parent replacement rejected
                ((0, 0, 4, 4), 'car', .99),
                ((250-x, 150-y, 280-x, 190-y), 'traffic light', .1)]

    extra = d.focused_detections(frame, [parent], .5, predict)
    assert calls == [(h, w, 3)]
    assert len(extra) == 1 and extra[0].box == Box(250, 150, 280, 190)
    assert extra[0].attrs['focus_parent_box'] == [160, 50, 295, 195]
    assert parent.box == Box(160, 50, 295, 195) and parent.track_id == 1


def test_no_head_dark_head_or_invalid_box_uses_no_extra_pass():
    frame = np.zeros((200, 300, 3), np.uint8)
    def fail(_):
        raise AssertionError('unexpected inference')
    for lights in ([], [Detection('traffic light', Box(10, 10, 90, 90), .9)],
                   [Detection('traffic light', Box(300, 200, 300, 200), .9)]):
        assert d.focused_detections(frame, lights, .5, fail) == []


@pytest.fixture
def focus_pipeline(monkeypatch):
    import streetsense.pipeline as sp

    monkeypatch.setenv('STREETSENSE_SAM_FALLBACK', '0')
    monkeypatch.setenv('STREETSENSE_LIGHT_FOCUS', '1')
    monkeypatch.setenv('STREETSENSE_LIGHT_TILES', '0')
    monkeypatch.setenv('STREETSENSE_SIGNAL_RESET_ON_VETO', '0')
    ctl = SimpleNamespace(parent_state='green', parent_probs=(.6, .1, .1, .2),
                          child_probs=(.99, .0, .0, .01), calls=0)
    parent_box = Box(120, 20, 200, 180)
    child_box = Box(140, 90, 170, 160)

    class Fake:
        def __call__(self, img):
            return [Detection('traffic light', parent_box, .9, 1)]
        def focused_lights(self, img, lights, conf):
            ctl.calls += 1
            return [Detection('traffic light', child_box, .9, None, {
                'source': 'focus', 'focus_parent_box': [120, 20, 200, 180]})]
        def tiled_lights(self, *args):
            return []

    class CNN:
        t_walk, t_veto = .95, .8
        def __call__(self, crop):
            return SignalPrediction(ctl.parent_probs if crop.shape[1] > 50 else ctl.child_probs)

    def reading(crop, kind):
        return LightReading(ctl.parent_state if crop.shape[1] > 50 else 'walk', .99, .1, None)

    monkeypatch.setattr(sp, 'build_detector', lambda cfg: Fake())
    monkeypatch.setattr(sp, 'classify_light', reading)
    monkeypatch.setattr(sp, 'figure_walk_guard', lambda r, *args: r)
    monkeypatch.setattr(sp.signal_cnn, 'get', lambda: CNN())
    cfg = Config()
    cfg.vlm.enabled = cfg.sam.enabled = cfg.depth.enabled = cfg.open_vocab.enabled = False
    pipe = sp.Pipeline(cfg, use_surface=False)
    img = np.zeros((240, 320, 3), np.uint8)
    def step(t):
        return pipe.process(img, annotate=False, t=t)
    return ctl, pipe, step


@pytest.mark.parametrize('state,probs', [
    ('red', (.01, .97, .01, .01)),
    ('countdown', (.01, .01, .97, .01)),
    ('unknown', (.01, .01, .01, .97)),
])
def test_parent_veto_cancels_child_walk_and_requires_fresh_votes(focus_pipeline, state, probs):
    ctl, pipe, step = focus_pipeline
    out = [step(t) for t in (0, .25, .5)]
    assert [o.detections[1].attrs['light_state'] for o in out] == ['unknown', 'unknown', 'walk']
    ctl.parent_state, ctl.parent_probs = state, probs
    child = step(.75).detections[1]
    assert child.attrs['focus_parent_veto'] and child.attrs['light_state'] != 'walk'
    ctl.parent_state, ctl.parent_probs = 'green', (.6, .1, .1, .2)
    reads = [step(t).detections[1].attrs['light_state'] for t in (1, 1.25, 1.5)]
    assert reads == ['unknown', 'unknown', 'walk']
    assert not pipe.smoother.reset_walk_on_veto  # original-head policy unchanged


def test_parent_and_child_cannot_double_vote_in_one_frame(focus_pipeline):
    ctl, _, step = focus_pipeline
    ctl.parent_state, ctl.parent_probs = 'walk', (.99, 0, 0, .01)
    assert all(d.attrs['light_state'] != 'walk' for d in step(0).detections)
    assert all(d.attrs['light_state'] != 'walk' for d in step(.25).detections)
    assert all(d.attrs['light_state'] == 'walk' for d in step(.5).detections)


def test_focus_does_not_stack_with_six_tile_mode(focus_pipeline, monkeypatch):
    ctl, _, step = focus_pipeline
    monkeypatch.setenv('STREETSENSE_LIGHT_TILES', '2x3')
    monkeypatch.setenv('STREETSENSE_LIGHT_TILES_MODE', 'always')
    assert 'detect_tiles' in step(0).timings_ms
    assert ctl.calls == 0


def test_default_does_not_add_focus_pass(focus_pipeline, monkeypatch):
    ctl, _, step = focus_pipeline
    monkeypatch.delenv('STREETSENSE_LIGHT_FOCUS')
    assert 'detect_focus' not in step(0).timings_ms
    assert ctl.calls == 0


def test_focus_trace_replays_independent_votes_and_parent_veto(focus_pipeline):
    from scripts.replay_signal_trace import replay
    from streetsense.signal_trace import summarise_signals
    ctl, _, step = focus_pipeline
    records = [summarise_signals(step(t), 320, 240) for t in (0, .25, .5)]
    ctl.parent_state, ctl.parent_probs = 'red', (.01, .97, .01, .01)
    records.append(summarise_signals(step(.75), 320, 240))
    ctl.parent_state, ctl.parent_probs = 'green', (.6, .1, .1, .2)
    records += [summarise_signals(step(t), 320, 240) for t in (1, 1.25, 1.5)]
    results = list(replay(records))
    assert [[h['state'] for h in r['heads']] for r in results] == [
        [h['state'] for h in r['heads']] for r in records]
    assert [r['local'] for r in results] == [r['local'] for r in records]


@pytest.mark.parametrize('gap', [1.1, 2.0, 5.0])
def test_focus_can_confirm_a_steady_walk_on_sparse_frames(focus_pipeline, gap):
    _, _, step = focus_pipeline
    states = [step(t).detections[1].attrs['light_state'] for t in (0, gap, 2*gap)]
    assert states == ['unknown', 'unknown', 'walk']
