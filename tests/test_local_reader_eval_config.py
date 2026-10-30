from types import SimpleNamespace

import pytest

from scripts import eval_local_reader as ev
from streetsense.signal_trace import summarise_signals
from streetsense.types import Box, Detection


def test_no_tiles_overrides_an_inherited_tiled_run(monkeypatch):
    monkeypatch.setenv("STREETSENSE_LIGHT_TILES", "2x3")
    monkeypatch.setenv("STREETSENSE_LIGHT_TILES_MODE", "always")
    monkeypatch.setenv("STREETSENSE_LIGHT_TILES_EVERY", "1")
    result = ev.eval_configuration(no_tiles=True)
    assert result["tiles"]["grid"] == [0, 0]
    assert result["tiles"]["every_n_frames"] == 0


def test_eval_reports_effective_inherited_tile_configuration(monkeypatch):
    monkeypatch.setenv("STREETSENSE_LIGHT_TILES", "2x3")
    monkeypatch.setenv("STREETSENSE_LIGHT_TILES_MODE", "always")
    monkeypatch.setenv("STREETSENSE_LIGHT_TILES_EVERY", "5")
    assert ev.eval_configuration()["tiles"] == {"grid": [2, 3], "mode": "always", "every_n_frames": 5}


def test_tile_zero_provenance_is_preserved():
    detection = Detection("traffic light", Box(10, 10, 30, 60), 0.9, None,
                          {"tile": 0, "light_state": "unknown"})
    out = SimpleNamespace(detections=[detection], index=0, t=0.0, crosswalk=None)
    assert ev.heads(out)[0]["source"] == "tile"
    assert summarise_signals(out, 100, 100)["heads"][0]["source"] == "tile"


def test_missing_dense_clip_cannot_report_zero_errors(monkeypatch):
    monkeypatch.setitem(ev.ev.CLIPS, "missing", "samples/missing.mp4")
    monkeypatch.setattr(ev.cv2, "VideoCapture", lambda path: SimpleNamespace(isOpened=lambda: False))
    with pytest.raises(FileNotFoundError, match="cannot open dense-scan clip"):
        ev.dense_scan([("missing", 0.0, 1.0)])
