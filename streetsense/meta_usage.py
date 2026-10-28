"""Usage ledger and hard budget cap for Meta Model API calls.

Writes one JSON line per call to outputs/meta_usage.jsonl (gitignored via outputs/*).
Never logs the API key. Costs are estimates from public Meta pricing (Oct 2026).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Standard-mode Muse Spark / Wonder token prices (USD per 1M tokens).
# Contributor Spark is cheaper; we key by model id.
_TOKEN_PRICES: dict[str, tuple[float, float, float]] = {
    # (input, output, cached_input) per 1M tokens
    "muse-spark-1.3": (1.25, 4.25, 0.15),
    "muse-spark-1.2": (1.25, 4.25, 0.15),
    "muse-spark-1.1": (1.25, 4.25, 0.15),
    "muse-spark-1.3-contributor": (0.10, 0.20, 0.002),
    "muse-spark-1.2-contributor": (0.10, 0.20, 0.002),
    "muse-wonder-1.0": (4.00, 20.00, 0.20),
}

# SAM 3.1: $2.50 / 1k images, $0.20 / 1k video frames (min 50 frames billed).
SAM_IMAGE_USD = 2.50 / 1000.0
SAM_VIDEO_FRAME_USD = 0.20 / 1000.0

# Muse Voice Transcribe: $0.18 / hour of audio.
VOICE_USD_PER_HOUR = 0.18

DEFAULT_LEDGER = Path("outputs/meta_usage.jsonl")
DEFAULT_BUDGET_USD = 20.0


class BudgetExceeded(RuntimeError):
    """Raised when STREETSENSE_META_BUDGET_USD would be exceeded by the next call."""


@dataclass
class UsageRecord:
    ts: float
    model: str
    kind: str  # chat | sam | voice
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    images: int = 0
    video_frames: int = 0
    audio_seconds: float = 0.0
    latency_s: float = 0.0
    estimated_cost_usd: float = 0.0
    note: str = ""


def budget_usd() -> float:
    raw = os.environ.get("STREETSENSE_META_BUDGET_USD")
    if raw is None or raw.strip() == "":
        return DEFAULT_BUDGET_USD
    return float(raw)


def estimate_cost(
    model: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    images: int = 0,
    video_frames: int = 0,
    audio_seconds: float = 0.0,
) -> float:
    mid = (model or "").lower()
    if mid.startswith("sam"):
        if video_frames > 0:
            billed = max(50, int(video_frames))
            return billed * SAM_VIDEO_FRAME_USD
        return max(0, int(images)) * SAM_IMAGE_USD
    if "voice" in mid or "transcribe" in mid:
        return (max(0.0, float(audio_seconds)) / 3600.0) * VOICE_USD_PER_HOUR
    prices = _TOKEN_PRICES.get(mid)
    if prices is None:
        # Unknown text model: use Muse Spark standard as a conservative-ish default.
        prices = _TOKEN_PRICES["muse-spark-1.3"]
    inp, out, cached = prices
    uncached = max(0, int(prompt_tokens) - int(cached_tokens))
    return (
        uncached * inp / 1_000_000.0
        + int(cached_tokens) * cached / 1_000_000.0
        + int(completion_tokens) * out / 1_000_000.0
    )


class UsageLedger:
    """Thread-safe append-only ledger with a hard spend cap."""

    def __init__(self, path: Path | str | None = None):
        # STREETSENSE_META_LEDGER lets tests and evals point the ledger somewhere else.
        self.path = Path(path) if path else Path(os.environ.get("STREETSENSE_META_LEDGER") or DEFAULT_LEDGER)
        self._lock = threading.Lock()
        self.total_usd = 0.0
        self.n_calls = 0
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.is_file():
            return
        try:
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.total_usd += float(rec.get("estimated_cost_usd") or 0.0)
                self.n_calls += 1
        except Exception:
            # Corrupt ledger must not block the assistant; start counting fresh in-memory.
            pass

    def remaining_usd(self) -> float:
        return max(0.0, budget_usd() - self.total_usd)

    def check_budget(self, estimated_next: float = 0.0) -> None:
        if self.total_usd + max(0.0, estimated_next) > budget_usd() + 1e-12:
            raise BudgetExceeded(
                f"Meta budget cap ${budget_usd():.2f} would be exceeded "
                f"(spent ${self.total_usd:.4f}, remaining ${self.remaining_usd():.4f})"
            )

    def record(self, rec: UsageRecord) -> UsageRecord:
        with self._lock:
            self.check_budget(0.0)  # refuse if already over
            if self.total_usd + rec.estimated_cost_usd > budget_usd() + 1e-12:
                raise BudgetExceeded(
                    f"Meta budget cap ${budget_usd():.2f} would be exceeded "
                    f"(spent ${self.total_usd:.4f} + ${rec.estimated_cost_usd:.4f})"
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(asdict(rec), separators=(",", ":")) + "\n")
            self.total_usd += rec.estimated_cost_usd
            self.n_calls += 1
        return rec

    def summary(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "total_usd": round(self.total_usd, 6),
            "budget_usd": budget_usd(),
            "remaining_usd": round(self.remaining_usd(), 6),
            "ledger": str(self.path),
        }


_LEDGER: UsageLedger | None = None
_LEDGER_LOCK = threading.Lock()


def get_ledger() -> UsageLedger:
    global _LEDGER
    with _LEDGER_LOCK:
        if _LEDGER is None:
            _LEDGER = UsageLedger()
        return _LEDGER


def mask_secret(text: str, secret: str | None) -> str:
    """Strip a secret from error strings so it never appears in logs."""
    if not text or not secret:
        return text
    return text.replace(secret, "***")


def usage_from_openai_response(resp, *, model: str, kind: str, latency_s: float, note: str = "") -> UsageRecord:
    u = getattr(resp, "usage", None)
    prompt = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
    completion = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
    cached = 0
    if u is not None:
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
    cost = estimate_cost(
        model, prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached
    )
    return UsageRecord(
        ts=time.time(),
        model=model,
        kind=kind,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        latency_s=round(latency_s, 3),
        estimated_cost_usd=round(cost, 6),
        note=note,
    )
