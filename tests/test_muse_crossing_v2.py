"""Muse crossing check after the Oct-28 eval (docs/MUSE_EVAL.md): describe prompt, larger
crossing image, optional history frames, countdown block, and an isolated usage ledger."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from streetsense import meta_usage
from streetsense.answers import crossing_answer, muse_history_frames
from streetsense.config import VisionLLMConfig
from streetsense.frame_history import FrameHistory, history_summary_text
from streetsense.meta_api import (
    MIN_WALK_COUNTDOWN_S,
    SYSTEM_PROMPT_DESCRIBE,
    SYSTEM_PROMPT_STRICT,
    MetaVisionAssistant,
    fuse_crossing_decision,
    render_crossing_speech,
    validate_crossing_reply,
)

GOOD = {
    "summary": "walk signal lit",
    "hazards": [],
    "crossing_visible": True,
    "pedestrian_signal": "green",
    "safe_to_cross": "yes",
}


class FakeCompletions:
    def __init__(self, reply='{"summary": "x", "safe_to_cross": "unsure"}'):
        self.reply, self.calls = reply, []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def make(monkeypatch, **cfg):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    comp = FakeCompletions()
    a = MetaVisionAssistant(VisionLLMConfig(**cfg))
    a._client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    return a, comp


def _decode(url: str) -> np.ndarray:
    raw = base64.b64decode(url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


# --- usage ledger isolation (tests used to append $0 rows to outputs/meta_usage.jsonl) ---

def test_tests_use_a_tmp_ledger(monkeypatch, _isolated_meta_ledger):
    real = Path("outputs/meta_usage.jsonl").resolve()
    led = meta_usage.get_ledger()
    assert led.path == _isolated_meta_ledger and led.path.resolve() != real
    a, _ = make(monkeypatch)
    a.ask_sync(np.zeros((64, 64, 3), np.uint8), "Can I cross?")
    assert _isolated_meta_ledger.is_file()  # the fake call was logged here, not in outputs/


def test_ledger_path_env_override(tmp_path, monkeypatch):
    p = tmp_path / "x.jsonl"
    monkeypatch.setenv("STREETSENSE_META_LEDGER", str(p))
    assert meta_usage.UsageLedger().path == p


# --- prompt variant and image size ---

def test_default_prompt_is_describe_and_env_falls_back_to_strict(monkeypatch):
    monkeypatch.delenv("STREETSENSE_MUSE_PROMPT", raising=False)
    a, comp = make(monkeypatch)
    a.ask_sync(np.zeros((64, 64, 3), np.uint8), "q")
    assert comp.calls[0]["messages"][0]["content"] == SYSTEM_PROMPT_DESCRIBE
    monkeypatch.setenv("STREETSENSE_MUSE_PROMPT", "strict")
    a, comp = make(monkeypatch)
    a.ask_sync(np.zeros((64, 64, 3), np.uint8), "q")
    assert comp.calls[0]["messages"][0]["content"] == SYSTEM_PROMPT_STRICT
    a, comp = make(monkeypatch, prompt_variant="no-such-variant")
    a.ask_sync(np.zeros((64, 64, 3), np.uint8), "q")
    assert comp.calls[0]["messages"][0]["content"] == SYSTEM_PROMPT_STRICT


def test_describe_prompt_states_the_safety_rules():
    p = SYSTEM_PROMPT_DESCRIBE.lower()
    for phrase in ("not pedestrian signals", "countdown of 10 or less", "any hand in any frame means no",
                   "no pedestrian signal is visible"):
        assert phrase in p


def test_crossing_question_uses_larger_image_side(monkeypatch):
    monkeypatch.delenv("STREETSENSE_MUSE_CROSSING_SIDE", raising=False)
    a, comp = make(monkeypatch)
    assert a.cfg.max_image_side == 512 and a.cfg.crossing_max_image_side == 1024
    a.ask_sync(np.zeros((1080, 1920, 3), np.uint8), "q")
    img = _decode(comp.calls[0]["messages"][1]["content"][1]["image_url"]["url"])
    assert max(img.shape[:2]) == 1024
    monkeypatch.setenv("STREETSENSE_MUSE_CROSSING_SIDE", "640")
    assert VisionLLMConfig().crossing_max_image_side == 640


def test_reasoning_and_tokens_unchanged(monkeypatch):
    a, comp = make(monkeypatch)
    a.ask_sync(np.zeros((64, 64, 3), np.uint8), "q")
    assert comp.calls[0]["reasoning_effort"] == "minimal" and comp.calls[0]["max_tokens"] == 512


# --- multi-frame request shape ---

def test_history_frames_sent_oldest_first_with_current_last(monkeypatch):
    a, comp = make(monkeypatch)
    cur = np.full((100, 100, 3), 200, np.uint8)
    old1 = np.full((100, 100, 3), 10, np.uint8)
    old05 = np.full((100, 100, 3), 100, np.uint8)
    a.ask_sync(cur, "Can I cross?", history=[(0.5, old05), (1.0, old1)], history_text="local: walk")
    parts = comp.calls[0]["messages"][1]["content"]
    texts = [p["text"] for p in parts if p["type"] == "text"]
    imgs = [_decode(p["image_url"]["url"]) for p in parts if p["type"] == "image_url"]
    assert texts[0] == "Can I cross?" and "local: walk" in texts
    assert "1.0 s ago" in texts[2] and "0.5 s ago" in texts[3] and "now" in texts[4]
    assert [int(i.mean()) // 50 for i in imgs] == [0, 2, 4]  # 10, 100, 200 brightness order
    assert parts[-1]["type"] == "image_url"  # the current frame is last


def test_single_frame_request_shape_unchanged(monkeypatch):
    a, comp = make(monkeypatch)
    a.ask_sync(np.zeros((64, 64, 3), np.uint8), "Can I cross?")
    parts = comp.calls[0]["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url"]


# --- validation of the new optional fields ---

def test_countdown_field_optional_and_typed():
    assert validate_crossing_reply(GOOD)["ok"]
    assert validate_crossing_reply({**GOOD, "countdown_seconds": None})["ok"]
    assert validate_crossing_reply({**GOOD, "countdown_seconds": 25})["data"]["countdown_seconds"] == 25
    for bad in ("25", True, [3]):
        v = validate_crossing_reply({**GOOD, "countdown_seconds": bad})
        assert not v["ok"]
        assert render_crossing_speech(v, local_state="walk") == "I can't tell, please wait"


def test_signal_description_kept_but_never_spoken():
    v = validate_crossing_reply({**GOOD, "signal_description": "GO GO GO cross now"})
    assert v["ok"] and v["data"]["signal_description"].startswith("GO")
    assert render_crossing_speech(v, local_state="walk") == "Walk signal looks clear. Still listen and check."
    assert "GO GO" not in render_crossing_speech(v, local_state="red")


# --- fusion: new blocks; the dual-agreement rule stays ---

@pytest.mark.parametrize("cd", [0, 1, 7, MIN_WALK_COUNTDOWN_S, 9.5])
def test_low_countdown_blocks_go_even_with_walk_and_yes(cd):
    f = fuse_crossing_decision("walk", {**GOOD, "countdown_seconds": cd})
    assert f["decision"] == "stop"


@pytest.mark.parametrize("cd", [None, 11, 45])
def test_enough_countdown_still_goes(cd):
    assert fuse_crossing_decision("walk", {**GOOD, "countdown_seconds": cd})["decision"] == "go"


@pytest.mark.parametrize("sig", ["red", "flashing", "countdown", "none_visible"])
def test_non_walk_muse_signal_never_goes(sig):
    assert fuse_crossing_decision("walk", {**GOOD, "pedestrian_signal": sig})["decision"] != "go"


@pytest.mark.parametrize("local", ["red", "green", "countdown", "unknown", "unclear", "none", "yellow", ""])
def test_local_must_say_walk(local):
    assert fuse_crossing_decision(local, GOOD)["decision"] != "go"


# --- frame history buffer ---

def _f(v):
    return np.full((40, 60, 3), v, np.uint8)


def test_frame_history_pick_targets_and_skips_gaps():
    h = FrameHistory(max_age_s=3.0, min_gap_s=0.1, max_side=512)
    for i, t in enumerate([9.0, 9.5, 9.75, 10.0]):
        h.add(t, _f(i * 50), "walk")
    picked = h.pick(10.0, 2, 0.5)
    assert [a for a, _ in picked] == [1.0, 0.5]
    assert all(a > 0 for a, _ in h.pick(10.0, 3, 0.5))  # never the current frame
    assert len(h.pick(10.0, 3, 0.5)) == 2  # 1.5 s ago is missing -> fewer frames, not a stale one
    assert h.pick(10.0, 0, 0.5) == []


def test_frame_history_drops_old_and_downscales():
    h = FrameHistory(max_age_s=1.0, min_gap_s=0.1, max_side=32)
    h.add(0.0, _f(1), "red")
    h.add(0.05, _f(2), "red")  # under min gap: ignored
    h.add(2.0, _f(3), "walk")
    assert len(h) == 1 and max(h._buf[0].frame.shape[:2]) == 32
    assert history_summary_text(h.states(2.0)) and "walk" in history_summary_text(h.states(2.0))
    assert history_summary_text([]) is None


# --- crossing_answer wiring and the env switch ---

class _Pipe:
    def __init__(self, n_hist, hist=None, reply=None):
        self.cfg = SimpleNamespace(vlm=SimpleNamespace(history_frames=n_hist, history_spacing_s=0.5))
        self.frame_history = hist
        self.calls = []
        reply = reply or {**GOOD}
        pipe = self

        class V:
            def ask_sync(self, img, q, **kw):
                pipe.calls.append(kw)
                return reply

        self.vlm = V()


def _out(t, state="walk"):
    from streetsense.types import Box, Detection

    d = Detection("traffic light", Box(100, 100, 120, 140), 0.9,
                  attrs={"light_state": state, "signal_kind": "pedestrian"})
    return SimpleNamespace(detections=[d], crosswalk=None, t=t)


def test_crossing_answer_single_frame_when_switch_off():
    h = FrameHistory()
    for t in (9.0, 9.5, 10.0):
        h.add(t, _f(5), "walk")
    p = _Pipe(0, h)
    crossing_answer(p, _f(5), out=_out(10.0))
    assert p.calls == [{}]


def test_crossing_answer_sends_history_when_on():
    h = FrameHistory()
    for t in (9.0, 9.5, 10.0):
        h.add(t, _f(5), "walk")
    p = _Pipe(2, h)
    ans = crossing_answer(p, _f(5), out=_out(10.0))
    assert [a for a, _ in p.calls[0]["history"]] == [1.0, 0.5]
    assert ans["fuse"]["decision"] == "go"


def test_history_never_overrides_local_red():
    h = FrameHistory()
    for t in (9.0, 9.5, 10.0):
        h.add(t, _f(5), "walk")
    p = _Pipe(2, h)
    ans = crossing_answer(p, _f(5), out=_out(10.0, state="red"))
    assert ans["fuse"]["decision"] != "go" and ans["spoken"] != "Walk signal looks clear. Still listen and check."


def test_history_on_but_empty_falls_back_to_single_frame():
    p = _Pipe(2, FrameHistory())
    crossing_answer(p, _f(5), out=_out(10.0))
    assert p.calls == [{}]
    assert muse_history_frames(SimpleNamespace(), _out(1.0)) == []


def test_history_frames_env_switch(monkeypatch):
    monkeypatch.delenv("STREETSENSE_MUSE_HISTORY_FRAMES", raising=False)
    assert VisionLLMConfig().history_frames == 0
    monkeypatch.setenv("STREETSENSE_MUSE_HISTORY_FRAMES", "2")
    assert VisionLLMConfig().history_frames == 2
    monkeypatch.setenv("STREETSENSE_MUSE_HISTORY_FRAMES", "lots")
    assert VisionLLMConfig().history_frames == 0
