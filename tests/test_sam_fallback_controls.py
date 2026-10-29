from types import SimpleNamespace

import numpy as np
import pytest

from streetsense.meta_api import fuse_crossing_decision
from streetsense.sam_fallback import SamControlConfig, SamFallbackController, sam_fallback_enabled_from_env


BOX = {"x1": 100, "y1": 50, "x2": 120, "y2": 100, "w": 640, "h": 480, "c": 0.9}


def mk(cfg=None, boxes=(BOX,), budget_ok=True, latency=0.8):
    calls = []

    def seg(img, prompt):
        calls.append(img.shape)
        return SimpleNamespace(boxes=list(boxes))

    def budget(x):
        if not budget_ok:
            raise RuntimeError("cap")

    logs = []
    c = SamFallbackController(cfg or SamControlConfig(enabled=True), seg, cost_per_call=0.0025, budget_check=budget,
                              sync=True, sim_latency_s=latency, log=logs.append)
    return c, calls, logs


def frame(v=0):
    return np.full((480, 640, 3), (v * 37) % 256, np.uint8)  # distinct thumbnails per v


def run(c, t, f, cw=True, conf=0.0, state="unknown"):
    return c.step(f, t, crosswalk_ahead=cw, local_conf=conf, local_state=state)


def test_env_switch(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False); monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    monkeypatch.delenv("STREETSENSE_SAM_FALLBACK", raising=False)
    assert not sam_fallback_enabled_from_env()
    monkeypatch.setenv("STREETSENSE_SAM_FALLBACK", "1")
    assert not sam_fallback_enabled_from_env()  # no key -> cannot be on
    monkeypatch.setenv("MODEL_API_KEY", "x")
    monkeypatch.delenv("STREETSENSE_SAM_FALLBACK")
    assert sam_fallback_enabled_from_env()
    monkeypatch.setenv("STREETSENSE_SAM_FALLBACK", "0")
    assert not sam_fallback_enabled_from_env()
    monkeypatch.setenv("STREETSENSE_SAM_MAX_PER_MIN", "4"); monkeypatch.setenv("STREETSENSE_SAM_SESSION_USD", "0.1")
    c = SamControlConfig.from_env()
    assert c.max_per_min == 4 and c.session_usd == 0.1 and not c.enabled


def test_no_crossing_context_no_calls():
    c, calls, _ = mk()
    for i in range(20):
        run(c, i * 0.5, frame(i), cw=False)
    assert calls == [] and c.stats["calls"] == 0


def test_context_sources_check_and_maps():
    c, calls, _ = mk()
    c.note_crossing_check(0.0)
    run(c, 0.0, frame(1), cw=False)
    assert len(calls) == 1
    c2, calls2, _ = mk()
    c2.note_crossing_distance(60.0); run(c2, 0.0, frame(1), cw=False); assert calls2 == []
    c2.note_crossing_distance(40.0); run(c2, 0.1, frame(1), cw=False); assert len(calls2) == 1


def test_confident_readable_local_light_skips_sam():
    c, calls, _ = mk()
    run(c, 0.0, frame(1), conf=0.6, state="red")
    assert calls == []
    run(c, 2.0, frame(2), conf=0.6, state="unknown")  # confident box but unreadable -> needed
    assert len(calls) == 1


def test_one_in_flight_interval_and_stale_drop():
    c, calls, _ = mk(latency=0.8)
    assert run(c, 0.0, frame(1)) == []
    run(c, 0.5, frame(2)); assert len(calls) == 1  # min interval 1 s
    out = run(c, 0.9, frame(3))
    assert len(out) == 1 and out[0][4] == 0.9  # delivered after latency, still fresh
    assert len(calls) == 1
    c.cfg.max_per_crossing = 1
    assert run(c, 2.5, frame(4)) == []  # 2.5 - 0.0 > 1.5 s -> stale dropped


def test_near_duplicate_reuses_result():
    c, calls, _ = mk(latency=0.1)
    f = frame(7)
    run(c, 0.0, f); run(c, 0.2, f)
    out = run(c, 1.2, f)
    assert len(calls) == 1 and c.stats["dup_reused"] == 1 and len(out) == 1


def test_per_crossing_cap_and_reset_on_context_end():
    cfg = SamControlConfig(enabled=True, max_per_crossing=2, context_grace_s=2.0)
    c, calls, logs = mk(cfg)
    for i in range(10):
        run(c, i * 1.1, frame(i))
    assert len(calls) == 2 and c.stats["cap_skips"]["per_crossing_cap"] >= 1
    assert sum("per_crossing_cap" in m for m in logs) == 1  # logged once
    run(c, 20.0, frame(50), cw=False)  # context gone > grace -> crossing closed
    assert not c.in_crossing
    run(c, 21.0, frame(51))
    assert len(calls) == 3 and c.crossing_id == 2


def test_per_minute_cap():
    cfg = SamControlConfig(enabled=True, max_per_crossing=100, max_per_min=6)
    c, calls, _ = mk(cfg)
    for i in range(30):
        run(c, i * 1.1, frame(i))
    assert len(calls) == 6
    run(c, 61.0, frame(99))
    assert len(calls) == 7


def test_session_usd_cap_disables_quietly():
    cfg = SamControlConfig(enabled=True, max_per_crossing=100, max_per_min=100, session_usd=0.005)
    c, calls, logs = mk(cfg)
    for i in range(10):
        run(c, i * 1.1, frame(i))
    assert len(calls) == 2 and c.disabled_reason == "session_usd_cap" and c.stats["usd"] == 0.005
    assert sum("disabled" in m for m in logs) == 1


def test_global_budget_cap_disables():
    c, calls, _ = mk(budget_ok=False)
    run(c, 0.0, frame(1))
    assert calls == [] and c.disabled_reason == "global_budget_cap"


def test_stable_local_signal_stops_sam_for_crossing():
    cfg = SamControlConfig(enabled=True, stable_frames=3)
    c, calls, _ = mk(cfg)
    for i in range(3):
        run(c, i * 0.1, frame(i), conf=0.7, state="red")
    assert c.answered
    run(c, 2.0, frame(9), conf=0.0, state="unknown")  # signal lost again: still no SAM this crossing
    assert calls == []


def test_api_error_counts_spend_and_returns_nothing():
    def seg(img, prompt):
        raise TimeoutError
    c = SamFallbackController(SamControlConfig(enabled=True), seg, cost_per_call=0.0025, budget_check=lambda x: None,
                              sync=True, sim_latency_s=0.0, log=lambda m: None)
    assert run(c, 0.0, frame(1)) == []
    assert c.stats["calls"] == 1 and c.stats["errors"] == 1


def _fake_pipeline(monkeypatch, boxes):
    import streetsense.pipeline as sp
    from streetsense.config import Config

    class Fake:
        def __call__(self, f):
            return []

    monkeypatch.setattr(sp, "build_detector", lambda cfg: Fake())
    cfg = Config(); cfg.vlm.enabled = False
    p = sp.Pipeline(cfg, use_surface=False)
    p.sam_fallback, _, _ = mk(boxes=boxes, latency=0.0)
    return p


def test_pipeline_injects_sam_boxes_as_candidates_only(monkeypatch):
    p = _fake_pipeline(monkeypatch, [BOX])
    p.note_crossing_check(0.0)
    f = np.zeros((480, 640, 3), np.uint8)
    out = p.process(f, annotate=False, t=0.0)
    sam = [d for d in out.detections if d.attrs.get("source") == "sam_fallback"]
    assert len(sam) == 1 and sam[0].label == "traffic light"
    assert sam[0].attrs["light_state"] in ("unknown", "red", "yellow", "green", "countdown", "walk")
    assert p.sam_fallback_summary()["calls"] == 1


def test_sam_boxes_alone_cannot_produce_go(monkeypatch):
    """Even if a SAM box sits on a lit walk figure and the local reader reads walk, go needs Muse yes."""
    from streetsense.answers import crossing_answer

    p = _fake_pipeline(monkeypatch, [BOX])
    f = np.zeros((480, 640, 3), np.uint8)
    f[50:100, 100:120] = (255, 255, 255)
    for muse in ({"error": "timeout"}, {"safe_to_cross": "unsure", "pedestrian_signal": "none_visible"}, {}):
        p.vlm = SimpleNamespace(ask_sync=lambda *a, m=muse, **k: m, maybe_request=lambda *a, **k: False, latest=None)
        ans = crossing_answer(p, f, capture_ts=0.0, clock=lambda: 0.0)
        assert ans["fuse"]["decision"] != "go"
    # and the fusion rule itself: local walk without Muse yes is never go
    for m in (None, {"safe_to_cross": "unsure"}, {"safe_to_cross": "no"}):
        assert fuse_crossing_decision("walk", m)["decision"] != "go"


def test_sam_crop_read_as_walk_still_needs_muse_yes(monkeypatch):
    """Worst case: the local reader reads the SAM-proposed crop as walk. Without Muse yes -> no go."""
    import streetsense.answers as ans_mod

    p = _fake_pipeline(monkeypatch, [BOX])
    monkeypatch.setattr(ans_mod, "local_signal_state", lambda out, shape: "walk")
    f = np.zeros((480, 640, 3), np.uint8)
    for muse in ({"error": "x"}, {"safe_to_cross": "unsure", "pedestrian_signal": "walk"},
                 {"safe_to_cross": "no", "pedestrian_signal": "red"}):
        p.vlm = SimpleNamespace(ask_sync=lambda *a, m=muse, **k: m, maybe_request=lambda *a, **k: False, latest=None)
        assert ans_mod.crossing_answer(p, f, capture_ts=0.0, clock=lambda: 0.0)["fuse"]["decision"] != "go"
