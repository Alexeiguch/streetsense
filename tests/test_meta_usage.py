"""Usage ledger + budget cap (no network)."""

import json
from pathlib import Path

import pytest

from streetsense.meta_usage import (
    BudgetExceeded,
    UsageLedger,
    UsageRecord,
    estimate_cost,
    mask_secret,
)


def test_estimate_spark_and_sam():
    c = estimate_cost("muse-spark-1.3", prompt_tokens=1000, completion_tokens=100)
    assert 0.001 < c < 0.01
    assert estimate_cost("sam-3.1", images=1) == pytest.approx(0.0025)
    assert estimate_cost("muse-voice-transcribe-1.0", audio_seconds=60) == pytest.approx(0.18 / 60)


def test_ledger_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("STREETSENSE_META_BUDGET_USD", "0.01")
    path = tmp_path / "usage.jsonl"
    led = UsageLedger(path)
    led.record(UsageRecord(ts=1.0, model="sam-3.1", kind="sam", images=1, estimated_cost_usd=0.0025))
    led.record(UsageRecord(ts=2.0, model="sam-3.1", kind="sam", images=1, estimated_cost_usd=0.0025))
    with pytest.raises(BudgetExceeded):
        led.record(UsageRecord(ts=3.0, model="sam-3.1", kind="sam", images=1, estimated_cost_usd=0.006))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["model"] == "sam-3.1"


def test_mask_secret():
    assert mask_secret("boom LLM_SECRET_XYZ failed", "LLM_SECRET_XYZ") == "boom *** failed"
