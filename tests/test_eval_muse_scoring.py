"""The eval harness must count any Muse yes or fused go on a non-walk frame."""
import importlib.util
import pathlib

_p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "eval_muse_crossing.py"
_spec = importlib.util.spec_from_file_location("eval_muse_crossing", _p)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _row(label, safe, fused, truncated=False):
    return {"label": label, "muse_safe": safe, "fused": fused, "latency_s": 1.0,
            "cost_usd": 0.001, "prompt_tokens": 100, "completion_tokens": 50, "truncated": truncated}


def test_non_walk_yes_and_go_are_counted():
    rows = [_row("walk", "yes", "go"), _row("red", "yes", "go"), _row("flashing", "yes", "stop"),
            _row("no_signal", "unsure", "unsure"), _row("countdown_ending", "invalid", "unsure", truncated=True),
            _row("unclear", "yes", "stop")]
    s = ev.summarise(rows)
    assert s["false_yes_non_walk"] == 2
    assert s["fused_go_non_walk"] == 1
    assert s["walk_yes_rate"] == "1/1"
    assert s["truncated"] == 1
    assert s["fused_go_unclear"] == 0


def test_unclear_and_symbol_off_not_in_non_walk():
    assert ev.NON_WALK == {"red", "flashing", "countdown_ending", "no_signal"}
