"""Muse Wonder as an on-demand guide agent over StreetSense tools.

Why Wonder here: the fast loop answers "what is in front of me right now" every
frame for free. Some questions need planning across time and tools, for example
"how do I get to the crosswalk?" or "what has been moving around me?". Wonder
reads the rolling observation log, the current detections and the frame, may
ask SAM 3.1 to find a specific thing, and turns that into one or two short
spoken directions.

Safety contract (enforced in code, not by the prompt):
  * Wonder never decides whether to cross. Its tool for crossing runs the
    existing fusion path (local reader says walk AND Muse Spark says yes) and
    only tells Wonder that the app will speak the verdict.
  * Any Wonder sentence about crossing permission or signal state is dropped.
  * The crossing line in the final speech always comes from that fusion path,
    or from the safe fallback when no fused answer exists.

Opt in only: the CLI `guide` command, or STREETSENSE_WONDER=1 for the web /guide
route. Every model turn is logged to outputs/meta_usage.jsonl and checked
against STREETSENSE_META_BUDGET_USD.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .config import WonderConfig
from .meta_api import SAFE_CROSSING_FALLBACK, encode_jpeg_data_url
from .meta_usage import BudgetExceeded, UsageRecord, estimate_cost, get_ledger, mask_secret

INSTRUCTIONS = (
    "You are the planning step of StreetSense, a walking aid for blind and low-vision people. "
    "Answer the user's spoken question about their surroundings. Use the tools: local facts are free, "
    "segment_concept costs money so use it only to locate one specific thing the local facts miss. "
    "Hard rules: you never decide whether it is safe to cross. Never say go, cross now, walk now, "
    "safe or clear to cross, and never say what a traffic or walk signal shows. If the question is about "
    "crossing, call check_crossing_signal once; the app speaks that verdict itself, so do not repeat it. "
    "Reply with at most 2 short sentences, under 35 words, plain text, no lists. Use left, right, ahead "
    "and rounded metres. Use verbs like head, turn, keep, stop, wait. Only state what tools or the image "
    "show, and say when you are unsure."
)

TOOLS: list[dict] = [
    {
        "type": "function",
        "name": "get_recent_history",
        "description": (
            "Compact facts from the on-device detector for the last N seconds, thinned to at most 15 frames. "
            "Each entry: t (s), objects [label, zone left|ahead|right, in_path, dist_m, track], "
            "crosswalk {found, zone}, underfoot surface, spoken alert. Free."
        ),
        "parameters": {
            "type": "object",
            "properties": {"seconds": {"type": "number", "description": "Look-back window, 1 to 30."}},
            "required": ["seconds"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_current_detections",
        "description": "Facts for the current frame from the on-device detector, same shape as one history entry. Free.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "segment_concept",
        "description": (
            "Ask SAM 3.1 to find every instance of one short noun phrase in the current frame, e.g. "
            "'crosswalk', 'curb ramp', 'open manhole'. Returns boxes with zone and size. Costs money; max 2 per question."
        ),
        "parameters": {
            "type": "object",
            "properties": {"concept": {"type": "string", "description": "Short noun phrase, max 40 characters."}},
            "required": ["concept"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "check_crossing_signal",
        "description": (
            "Run the app's crossing check (local signal reader fused with a separate vision check). "
            "The app speaks the verdict itself. You only learn whether a signal and crosswalk are visible."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]

# Sentences matching any of these never reach the user from Wonder.
_FORBIDDEN = [
    re.compile(p, re.I)
    for p in (
        r"\b(safe|clear|ok|okay|fine|good)\s+to\s+(cross|go|walk|proceed)\b",
        r"\b(can|may|could|should|must)\s+(now\s+|safely\s+)?(cross|go|walk|proceed|start)\b",
        r"\bgo\s+ahead\b",
        r"\b(cross|walk|go)\s+(now|across)\b",
        r"\bstart\s+(crossing|walking)\b",
        r"\bproceed\b",
        r"\byour\s+turn\b",
        r"\ball\s+clear\b",
        r"\b(walk|green|red|amber|yellow|white|orange|flashing)\s+(traffic\s+|pedestrian\s+|crossing\s+|walk\s+)?(signal|light|man|figure|person|sign|hand)s?\b",
        r"\b(signal|light|lights)\s+(is|are|shows?|turned|says|changed)\b",
        r"\bsafe\b",
    )
]

_CROSSING_Q = re.compile(r"\b(cross|crossing|crosswalk|street|road|signal|light|walk sign|zebra)\b", re.I)


def is_crossing_question(question: str) -> bool:
    return bool(_CROSSING_Q.search(question or ""))


def sanitize_guidance(text: str, max_words: int = 40) -> tuple[str, list[str]]:
    """Drop any sentence that could read as crossing permission or a signal state.

    Returns (kept_text, dropped_sentences).
    """
    text = re.sub(r"[*_`#>]+", "", text or "")
    text = re.sub(r"\s*[\u2014\u2013]\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "", []
    parts = re.split(r"(?<=[.!?;])\s+|\s+-\s+|\s*\n\s*", text)
    kept, dropped = [], []
    for s in parts:
        s = s.strip()
        if not s:
            continue
        if any(p.search(s) for p in _FORBIDDEN):
            dropped.append(s)
        else:
            kept.append(s)
    words = " ".join(kept).split()
    out = " ".join(words[:max_words])
    if out and out[-1] not in ".!?":
        out += "."
    return out, dropped


@dataclass
class GuideTools:
    """Callables Wonder may use. Built from a live Pipeline or injected by tests."""

    history: Callable[[float], list[dict]]
    current: Callable[[], dict | None]
    segment: Callable[[str], dict] | None = None
    crossing: Callable[[], dict] | None = None

    @classmethod
    def from_pipeline(cls, pipe: Any, frame: np.ndarray, out: Any = None, clock=time.monotonic) -> "GuideTools":
        from .answers import crossing_answer

        H, W = frame.shape[:2]

        def segment(concept: str) -> dict:
            from .meta_api import MetaSamSegmenter
            from .geometry import horizontal_zone
            from .types import Box

            sam = pipe.sam or MetaSamSegmenter()
            if not sam.enabled:
                return {"error": "SAM unavailable"}
            seg = sam.segment(frame, concept, decode_masks=False)
            boxes = []
            for b in seg.boxes[:6]:
                try:
                    sx = W / float(b.get("w") or W)
                    sy = H / float(b.get("h") or H)
                    bx = Box(b["x1"] * sx, b["y1"] * sy, b["x2"] * sx, b["y2"] * sy).clip(W, H)
                except Exception:
                    continue
                boxes.append({
                    "zone": horizontal_zone(bx, W),
                    "area_frac": round(bx.area / float(W * H), 3),
                    "bottom_frac": round(bx.y2 / float(H), 2),
                    "score": b.get("c"),
                })
            return {"concept": concept, "count": len(boxes), "boxes": boxes}

        def crossing() -> dict:
            return crossing_answer(pipe, frame, out=out, clock=clock)

        return cls(
            history=lambda s: pipe.observations.window(s),
            current=pipe.observations.latest,
            segment=segment,
            crossing=crossing,
        )


@dataclass
class GuideResult:
    speech: str
    guidance: str = ""
    crossing_line: str | None = None
    crossing: dict | None = None
    tool_calls: list[dict] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    turns: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _usage(resp: Any) -> tuple[int, int, int, int]:
    u = _get(resp, "usage")
    if u is None:
        return 0, 0, 0, 0
    inp = int(_get(u, "input_tokens", 0) or _get(u, "prompt_tokens", 0) or 0)
    out = int(_get(u, "output_tokens", 0) or _get(u, "completion_tokens", 0) or 0)
    det_in = _get(u, "input_tokens_details") or _get(u, "prompt_tokens_details")
    cached = int(_get(det_in, "cached_tokens", 0) or 0) if det_in is not None else 0
    det_out = _get(u, "output_tokens_details")
    reasoning = int(_get(det_out, "reasoning_tokens", 0) or 0) if det_out is not None else 0
    return inp, out, cached, reasoning


def _output_text(resp: Any) -> str:
    t = _get(resp, "output_text")
    if isinstance(t, str) and t.strip():
        return t
    chunks = []
    for item in _get(resp, "output") or []:
        if _get(item, "type") != "message":
            continue
        for c in _get(item, "content") or []:
            txt = _get(c, "text")
            if txt:
                chunks.append(txt)
    return "".join(chunks)


def _function_calls(resp: Any) -> list[dict]:
    calls = []
    for item in _get(resp, "output") or []:
        if _get(item, "type") == "function_call":
            calls.append({
                "name": _get(item, "name") or "",
                "arguments": _get(item, "arguments") or "{}",
                "call_id": _get(item, "call_id") or _get(item, "id") or "",
            })
    return calls


class WonderGuide:
    """Capped Responses API tool loop. One instance can serve many questions."""

    def __init__(self, cfg: WonderConfig | None = None, client: Any = None, clock=time.perf_counter):
        self.cfg = cfg or WonderConfig(enabled=True)
        self.clock = clock
        self._client = client
        if self._client is None and self.cfg.api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url, timeout=self.cfg.timeout_s)

    @property
    def available(self) -> bool:
        return self._client is not None

    # Tool dispatch -----------------------------------------------------------------
    def _dispatch(self, name: str, raw_args: str, tools: GuideTools, state: dict) -> dict:
        try:
            args = json.loads(raw_args or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except Exception:
            return {"error": "invalid JSON arguments"}
        if name == "get_recent_history":
            try:
                secs = float(args.get("seconds", 10))
            except (TypeError, ValueError):
                return {"error": "seconds must be a number"}
            secs = min(30.0, max(1.0, secs))
            return {"seconds": secs, "entries": tools.history(secs)}
        if name == "get_current_detections":
            return {"current": tools.current() or {"note": "no frames yet"}}
        if name == "segment_concept":
            concept = str(args.get("concept", "")).strip()
            if not concept or len(concept) > 40:
                return {"error": "concept must be 1 to 40 characters"}
            if tools.segment is None:
                return {"error": "SAM unavailable"}
            if state["segments"] >= self.cfg.max_segment_calls:
                return {"error": "segment limit reached for this question"}
            state["segments"] += 1
            try:
                return tools.segment(concept)
            except BudgetExceeded as e:
                return {"error": str(e)[:160]}
            except Exception as e:
                return {"error": mask_secret(f"{type(e).__name__}: {e}", self.cfg.api_key)[:160]}
        if name == "check_crossing_signal":
            if tools.crossing is None:
                return {"error": "crossing check unavailable"}
            if state["crossing"] is None:
                try:
                    state["crossing"] = tools.crossing()
                except Exception as e:
                    state["crossing"] = {"spoken": SAFE_CROSSING_FALLBACK, "error": type(e).__name__}
            c = state["crossing"]
            # Wonder learns visibility only, never the verdict.
            local = str(c.get("local_light", "unknown"))
            cur = tools.current() or {}
            return {
                "verdict_spoken_by_app": True,
                "pedestrian_signal_visible": local not in {"unknown", "none"},
                "crosswalk_found": bool((cur.get("crosswalk") or {}).get("found")),
            }
        return {"error": f"unknown tool {name}"}

    # Main loop -----------------------------------------------------------------------
    def run(self, question: str, frame: np.ndarray | None, tools: GuideTools) -> GuideResult:
        question = (question or "").strip()[:300] or "What is around me?"
        t_start = self.clock()
        res = GuideResult(speech="")
        state: dict = {"segments": 0, "crossing": None}
        if not self.available:
            res.error = "Muse Wonder unavailable: set MODEL_API_KEY"
        else:
            self._loop(question, frame, tools, state, res)

        guidance, dropped = sanitize_guidance(res.guidance)
        res.guidance = guidance
        res.dropped.extend(dropped)
        if state["crossing"] is not None:
            res.crossing = {k: v for k, v in state["crossing"].items() if k != "muse_raw"}
            res.crossing_line = state["crossing"].get("spoken") or SAFE_CROSSING_FALLBACK
        elif is_crossing_question(question):
            res.crossing_line = SAFE_CROSSING_FALLBACK
        parts = [p if p.rstrip()[-1:] in ".!?" else p.rstrip() + "." for p in (res.crossing_line, guidance) if p]
        res.speech = " ".join(parts) if parts else "I'm not sure. Please ask again."
        res.latency_s = round(self.clock() - t_start, 3)
        res.cost_usd = round(res.cost_usd, 6)
        return res

    def _loop(self, question: str, frame: np.ndarray | None, tools: GuideTools, state: dict, res: GuideResult) -> None:
        cfg = self.cfg
        ledger = get_ledger()
        content: list[dict] = [{"type": "input_text", "text": question}]
        if frame is not None:
            content.append({"type": "input_image", "image_url": encode_jpeg_data_url(frame, cfg.max_image_side)})
        next_input: list[dict] = [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": content},
        ]
        history = next_input  # grows in stateless mode
        prev_id = None
        for turn in range(cfg.max_turns):
            if res.cost_usd + cfg.turn_estimate_usd > cfg.max_cost_usd:
                res.error = "per-question Wonder cost cap reached"
                break
            try:
                ledger.check_budget(cfg.turn_estimate_usd)
            except BudgetExceeded as e:
                res.error = str(e)[:200]
                break
            last_turn = turn == cfg.max_turns - 1
            kwargs: dict = {
                "model": cfg.model,
                "input": next_input,
                "tools": TOOLS,
                "tool_choice": "auto",
                "reasoning": {"effort": cfg.reasoning_effort},
                "max_output_tokens": cfg.max_output_tokens,
            }
            if last_turn:
                kwargs["tool_choice"] = "none"
            if prev_id and not cfg.stateless:
                kwargs["previous_response_id"] = prev_id
            t0 = time.perf_counter()
            try:
                resp = self._client.responses.create(**kwargs)
            except Exception as e:
                res.error = mask_secret(f"{type(e).__name__}: {e}", cfg.api_key)[:200]
                break
            latency = time.perf_counter() - t0
            res.turns += 1
            inp, out, cached, reasoning = _usage(resp)
            cost = estimate_cost(cfg.model, prompt_tokens=inp, completion_tokens=out, cached_tokens=cached)
            res.cost_usd += cost
            res.input_tokens += inp
            res.output_tokens += out
            res.reasoning_tokens += reasoning
            try:
                ledger.record(UsageRecord(
                    ts=time.time(), model=cfg.model, kind="agent", prompt_tokens=inp,
                    completion_tokens=out, cached_tokens=cached, images=1 if (frame is not None and (turn == 0 or cfg.stateless)) else 0,
                    latency_s=round(latency, 3), estimated_cost_usd=round(cost, 6),
                    note=f"wonder guide turn {turn + 1}: {question[:60]}",
                ))
            except BudgetExceeded as e:
                res.error = str(e)[:200]
                break
            calls = _function_calls(resp)
            if not calls:
                res.guidance = _output_text(resp)
                return
            prev_id = _get(resp, "id")
            if cfg.stateless:
                history.extend({"type": "function_call", "name": c["name"], "arguments": c["arguments"],
                                "call_id": c["call_id"]} for c in calls)
                next_input = history
            else:
                next_input = []
            for c in calls:
                t1 = time.perf_counter()
                result = self._dispatch(c["name"], c["arguments"], tools, state)
                res.tool_calls.append({
                    "name": c["name"], "arguments": c["arguments"][:200],
                    "ms": round((time.perf_counter() - t1) * 1000, 1),
                    "error": result.get("error") if isinstance(result, dict) else None,
                })
                next_input.append({
                    "type": "function_call_output",
                    "call_id": c["call_id"],
                    "output": json.dumps(result, separators=(",", ":"), default=str)[:6000],
                })
        if not res.guidance and res.error is None:
            res.error = "Wonder did not finish within the turn cap"
