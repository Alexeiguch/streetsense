"""Muse Wonder guide: offline, mocked. Wonder can never authorise crossing."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from streetsense import meta_usage
from streetsense.config import WonderConfig
from streetsense.meta_api import SAFE_CROSSING_FALLBACK
from streetsense.wonder_agent import (
    GuideTools,
    WonderGuide,
    is_crossing_question,
    sanitize_guidance,
)


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("STREETSENSE_META_BUDGET_USD", "1.0")
    led = meta_usage.UsageLedger(tmp_path / "usage.jsonl")
    monkeypatch.setattr(meta_usage, "_LEDGER", led)
    return led


def call(name, args="{}", cid=None):
    return NS(type="function_call", name=name, arguments=args, call_id=cid or f"c_{name}")


def text_resp(text, inp=1000, out=100):
    return NS(id="r_final", output=[NS(type="message", content=[NS(type="output_text", text=text)])],
              output_text=text, usage=NS(input_tokens=inp, output_tokens=out,
                                         input_tokens_details=NS(cached_tokens=0),
                                         output_tokens_details=NS(reasoning_tokens=40)))


def tool_resp(*calls, inp=1000, out=50):
    return NS(id="r_tools", output=list(calls), output_text="",
              usage=NS(input_tokens=inp, output_tokens=out, input_tokens_details=NS(cached_tokens=0),
                       output_tokens_details=NS(reasoning_tokens=20)))


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.responses = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(json.loads(json.dumps(kw, default=str)))
        if not self.script:
            raise AssertionError("unexpected extra model call")
        return self.script.pop(0)


def tools(crossing_spoken="Do not cross.", local="red", segment=None):
    hist = [{"t": 1.0, "objects": [{"label": "car", "zone": "left", "in_path": False, "dist_m": 8.0}],
             "signal": local, "crosswalk": {"found": True, "zone": "ahead"}, "underfoot": "sidewalk"}]
    return GuideTools(
        history=lambda s: hist,
        current=lambda: hist[-1],
        segment=segment or (lambda c: {"concept": c, "count": 1, "boxes": [{"zone": "ahead"}]}),
        crossing=lambda: {"spoken": crossing_spoken, "local_light": local,
                          "fuse": {"decision": "stop"}, "validated": True, "muse_raw": {"safe_to_cross": "no"}},
    )


FRAME = np.zeros((64, 64, 3), np.uint8)


@pytest.mark.parametrize("bad", [
    "It is safe to cross now.",
    "You can cross.",
    "Go ahead.",
    "The walk signal is on.",
    "The light is green.",
    "There is a red traffic light ahead.",
    "Cross now while the road is empty.",
    "All clear, start crossing.",
    "You may proceed.",
    "It's your turn.",
])
def test_sanitizer_drops_crossing_permission_and_signal_state(bad):
    kept, dropped = sanitize_guidance(f"The crosswalk is 5 metres ahead. {bad} Turn slightly left.")
    assert bad.strip() in " ".join(dropped)
    assert kept == "The crosswalk is 5 metres ahead. Turn slightly left."


def test_sanitizer_keeps_directions_and_removes_dashes_and_markdown():
    kept, dropped = sanitize_guidance("**Head** left \u2014 the curb ramp is 3 metres away.")
    assert dropped == []
    assert kept == "Head left, the curb ramp is 3 metres away."


def test_crossing_question_detection():
    assert is_crossing_question("Can I cross here?")
    assert is_crossing_question("where is the crosswalk")
    assert not is_crossing_question("What has been moving around me?")


def test_wonder_text_cannot_authorise_crossing(ledger):
    client = FakeClient([
        tool_resp(call("get_recent_history", '{"seconds": 5}'), call("check_crossing_signal")),
        text_resp("It is safe to cross now. The crosswalk is ahead, keep straight."),
    ])
    res = WonderGuide(WonderConfig(enabled=True), client=client).run("Can I cross?", FRAME, tools())
    assert res.crossing_line == "Do not cross."
    assert res.speech.startswith("Do not cross.")
    assert "safe" not in res.speech.lower()
    assert any("safe to cross" in d for d in res.dropped)
    assert res.turns == 2


def test_go_line_only_comes_from_fusion_path():
    """Even a fused go is passed through verbatim; Wonder text never adds or edits it."""
    go = "Walk signal looks clear. Still listen and check."
    client = FakeClient([tool_resp(call("check_crossing_signal")), text_resp("Head straight, 6 metres.")])
    res = WonderGuide(WonderConfig(enabled=True), client=client).run("Can I cross?", FRAME,
                                                                      tools(crossing_spoken=go, local="walk"))
    assert res.speech == go + " Head straight, 6 metres."


def test_crossing_verdict_is_hidden_from_wonder():
    client = FakeClient([tool_resp(call("check_crossing_signal")), text_resp("Wait here.")])
    WonderGuide(WonderConfig(enabled=True), client=client).run("Can I cross?", FRAME, tools())
    second = client.calls[1]["input"]
    outputs = [i for i in second if isinstance(i, dict) and i.get("type") == "function_call_output"]
    payload = json.loads(outputs[0]["output"])
    assert payload["verdict_spoken_by_app"] is True
    assert "spoken" not in payload and "fuse" not in payload and "decision" not in outputs[0]["output"]


def test_crossing_question_without_tool_call_gets_safe_fallback():
    client = FakeClient([text_resp("The crosswalk is ahead.")])
    res = WonderGuide(WonderConfig(enabled=True), client=client).run("Can I cross the road?", FRAME, tools())
    assert res.crossing_line == SAFE_CROSSING_FALLBACK
    assert res.speech.startswith(SAFE_CROSSING_FALLBACK)


def test_turn_cap_forces_final_answer_and_stops():
    cfg = WonderConfig(enabled=True, max_turns=3)
    client = FakeClient([tool_resp(call("get_current_detections")) for _ in range(3)])
    res = WonderGuide(cfg, client=client).run("What is around me?", FRAME, tools())
    assert res.turns == 3
    assert client.calls[-1]["tool_choice"] == "none"
    assert all(c["tool_choice"] == "auto" for c in client.calls[:-1])
    assert res.error and "turn cap" in res.error
    assert res.speech == "I'm not sure. Please ask again."


def test_usage_logged_with_wonder_pricing(ledger):
    client = FakeClient([text_resp("Bench on your right, 2 metres.", inp=2000, out=300)])
    res = WonderGuide(WonderConfig(enabled=True), client=client).run("What is near me?", FRAME, tools())
    rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["model"] == "muse-wonder-1.0" and rows[0]["kind"] == "agent"
    expected = 2000 * 4.0 / 1e6 + 300 * 20.0 / 1e6
    assert rows[0]["estimated_cost_usd"] == pytest.approx(expected, abs=1e-6)
    assert res.cost_usd == pytest.approx(expected, abs=1e-6)
    assert res.reasoning_tokens == 40


def test_global_budget_blocks_before_any_call(monkeypatch, ledger):
    monkeypatch.setenv("STREETSENSE_META_BUDGET_USD", "0.01")
    client = FakeClient([])
    res = WonderGuide(WonderConfig(enabled=True), client=client).run("Can I cross?", FRAME, tools())
    assert client.calls == []
    assert "budget" in res.error.lower()
    assert res.speech == SAFE_CROSSING_FALLBACK + "."


def test_per_question_cost_cap():
    cfg = WonderConfig(enabled=True, max_cost_usd=0.05, turn_estimate_usd=0.03)
    client = FakeClient([tool_resp(call("get_current_detections"), inp=5000, out=500), text_resp("x")])
    res = WonderGuide(cfg, client=client).run("What is around me?", FRAME, tools())
    assert len(client.calls) == 1
    assert "cost cap" in res.error


def test_segment_calls_capped_and_args_validated():
    seen = []
    t = tools(segment=lambda c: seen.append(c) or {"concept": c, "count": 0, "boxes": []})
    client = FakeClient([
        tool_resp(call("segment_concept", '{"concept":"curb ramp"}', "a"),
                  call("segment_concept", '{"concept":"crosswalk"}', "b"),
                  call("segment_concept", '{"concept":"bollard"}', "c"),
                  call("segment_concept", '{"concept":"' + "x" * 60 + '"}', "d"),
                  call("segment_concept", "not json", "e"),
                  call("rm_rf", "{}", "f")),
        text_resp("Curb ramp ahead."),
    ])
    res = WonderGuide(WonderConfig(enabled=True), client=client).run("Find the curb ramp", FRAME, t)
    assert seen == ["curb ramp", "crosswalk"]
    errs = [c["error"] for c in res.tool_calls]
    assert errs[2] == "segment limit reached for this question"
    assert "1 to 40" in errs[3] and "invalid JSON" in errs[4] and "unknown tool" in errs[5]


def test_stateless_mode_resends_conversation_and_chained_mode_uses_previous_id():
    script = lambda: [tool_resp(call("get_current_detections")), text_resp("Wait.")]  # noqa: E731
    c1 = FakeClient(script())
    WonderGuide(WonderConfig(enabled=True, stateless=True), client=c1).run("What is near?", FRAME, tools())
    types = [i.get("type") or i.get("role") for i in c1.calls[1]["input"]]
    assert types[:2] == ["system", "user"] and "function_call" in types and "function_call_output" in types
    assert "previous_response_id" not in c1.calls[1]
    c2 = FakeClient(script())
    WonderGuide(WonderConfig(enabled=True, stateless=False), client=c2).run("What is near?", FRAME, tools())
    assert c2.calls[1]["previous_response_id"] == "r_tools"
    assert [i["type"] for i in c2.calls[1]["input"]] == ["function_call_output"]


def test_no_key_means_unavailable_without_raising(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    g = WonderGuide(WonderConfig(enabled=True))
    assert not g.available
    res = g.run("Can I cross?", FRAME, tools())
    assert res.speech == SAFE_CROSSING_FALLBACK + "." and "unavailable" in res.error


def test_api_errors_are_masked(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "sk-secret-123")

    class Boom:
        responses = NS(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("bad key sk-secret-123")))

    res = WonderGuide(WonderConfig(enabled=True), client=Boom()).run("What is near?", FRAME, tools())
    assert "sk-secret-123" not in res.error and "***" in res.error


def test_wonder_is_opt_in(monkeypatch):
    monkeypatch.delenv("STREETSENSE_WONDER", raising=False)
    assert WonderConfig().enabled is False
    assert WonderConfig.from_env().enabled is False
    monkeypatch.setenv("STREETSENSE_WONDER", "1")
    assert WonderConfig.from_env().enabled is True
