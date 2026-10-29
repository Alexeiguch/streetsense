"""Shared test setup.

Every test gets a throwaway Meta usage ledger so fake-client calls never append
zero-cost rows to the real outputs/meta_usage.jsonl.
"""

import pytest

from streetsense import meta_usage


@pytest.fixture(autouse=True)
def _isolated_meta_ledger(tmp_path, monkeypatch):
    path = tmp_path / "meta_usage.jsonl"
    monkeypatch.setenv("STREETSENSE_META_LEDGER", str(path))
    monkeypatch.setattr(meta_usage, "_LEDGER", meta_usage.UsageLedger(path))
    yield path


@pytest.fixture(autouse=True)
def _sam_fallback_off_by_default(monkeypatch):
    """Pipelines in tests must never make real SAM calls; SAM fallback tests opt in explicitly."""
    monkeypatch.setenv("STREETSENSE_SAM_FALLBACK", "0")
