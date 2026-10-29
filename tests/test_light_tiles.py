import numpy as np

from streetsense import detector as D
from streetsense.types import Box, Detection


def _tl(x1, y1, x2, y2, c, **a):
    return Detection("traffic light", Box(x1, y1, x2, y2), c, None, a)


def test_light_tiles_env_default_off(monkeypatch):
    monkeypatch.delenv("STREETSENSE_LIGHT_TILES", raising=False)
    assert D.light_tiles()[0] == (0, 0)
    for v in ("0", "off", "1x1", "bad", "5x5"):
        monkeypatch.setenv("STREETSENSE_LIGHT_TILES", v)
        assert D.light_tiles()[0] == (0, 0)


def test_light_tiles_env_parse(monkeypatch):
    monkeypatch.setenv("STREETSENSE_LIGHT_TILES", "2x3")
    monkeypatch.setenv("STREETSENSE_LIGHT_TILES_EVERY", "5")
    monkeypatch.delenv("STREETSENSE_LIGHT_TILES_MODE", raising=False)
    assert D.light_tiles() == ((2, 3), "fallback", 5)
    monkeypatch.setenv("STREETSENSE_LIGHT_TILES_MODE", "always")
    assert D.light_tiles()[1] == "always"


def test_tile_windows_cover_frame():
    w = D.tile_windows(1557, 2011, (2, 3))
    assert len(w) == 6
    assert min(x for x, *_ in w) == 0 and max(x + tw for x, _, tw, _ in w) == 1557
    assert max(y + th for _, y, _, th in w) == 2011


def test_tile_detections_map_to_frame_and_filter():
    frame = np.zeros((200, 300, 3), np.uint8)
    seen = []

    def pred(tile):
        seen.append(tile.shape)
        return [((10, 10, 30, 50), "traffic light", 0.9), ((0, 0, 20, 20), "traffic light", 0.2),
                ((0, 0, 50, 50), "car", 0.9)]

    out = D.tile_detections(frame, (2, 3), 0.5, pred)
    assert len(seen) == 6 and len(out) == 6
    xs = sorted({d.box.x1 for d in out})
    assert xs[0] == 10 and xs[-1] > 10  # offsets applied
    assert all(d.track_id is None and "tile" in d.attrs for d in out)


def test_nms_merges_overlaps_keeps_best():
    a = _tl(0, 0, 10, 20, 0.6); b = _tl(1, 1, 10, 20, 0.9, tile=0); c = _tl(100, 100, 110, 120, 0.5)
    out = D.nms_lights([a, b, c])
    assert out == [b, c]


def test_pipeline_tiles_off_by_default_and_merge_when_on(monkeypatch):
    from streetsense.config import Config
    from streetsense.pipeline import Pipeline

    calls = []

    class Fake:
        c = 0.3

        def __call__(self, frame):
            return [_tl(10, 10, 20, 30, self.c)]

        def zoomed_lights(self, frame, scales, conf):
            calls.append("zoom"); return []

        def tiled_lights(self, frame, grid, conf):
            calls.append(("tiles", grid, conf)); return [_tl(11, 10, 20, 30, 0.8, tile=1), _tl(200, 50, 210, 70, 0.7, tile=2)]

    monkeypatch.delenv("STREETSENSE_LIGHT_TILES", raising=False)
    import streetsense.pipeline as sp
    monkeypatch.setattr(sp, "build_detector", lambda cfg: Fake())
    cfg = Config(); cfg.vlm.enabled = False
    p = Pipeline(cfg, use_surface=False)
    frame = np.zeros((240, 320, 3), np.uint8)
    p.process(frame, annotate=False, t=0.0)
    assert not any(isinstance(c, tuple) for c in calls)

    calls.clear(); monkeypatch.setenv("STREETSENSE_LIGHT_TILES", "2x3")
    out = p.process(frame, annotate=False, t=0.25)
    assert calls == [("tiles", (2, 3), 0.5)]  # tiles replace zoom on this frame
    lights = [d for d in out.detections if d.label == "traffic light"]
    assert len(lights) == 2  # overlap merged
    # fallback mode: a confident full-frame light skips the tiles; "always" runs them anyway
    calls.clear(); p.detector.c = 0.6
    p.process(frame, annotate=False, t=0.5)
    assert calls == []
    monkeypatch.setenv("STREETSENSE_LIGHT_TILES_MODE", "always")
    p.process(frame, annotate=False, t=0.75)
    assert calls == [("tiles", (2, 3), 0.5)]
