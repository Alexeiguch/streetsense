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
