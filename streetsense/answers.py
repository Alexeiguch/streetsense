"""One shared crossing answer path, used by /ask, the glasses gateway and the Wonder guide.

The rule never changes: say go only when the local reader says walk AND a
validated Muse Spark reply says yes. Every caller gets its spoken crossing line
from here, so no model free text can authorise a crossing.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .meta_api import (
    SAFE_CROSSING_FALLBACK,
    fuse_crossing_decision,
    render_crossing_speech,
    validate_crossing_reply,
)
from .rules import select_crossing_signal

DEFAULT_CROSSING_QUESTION = "Is it safe to cross the street right now?"


def local_signal_state(out, frame_shape: tuple[int, ...]) -> str:
    """Pedestrian signal state from the local reader for one processed frame."""
    H, W = frame_shape[:2]
    sel = select_crossing_signal(out.detections, W, H, out.crosswalk)
    return sel.decision if sel is not None else "unknown"


def muse_history_frames(pipe: Any, out: Any) -> list:
    """Older frames to send with the crossing question, per cfg.vlm.history_frames.

    Empty when the switch is off, the pipeline keeps no history, or nothing recent is
    buffered. Extra frames only help Muse see a flickering walk symbol; fusion is unchanged.
    """
    cfg = getattr(getattr(pipe, "cfg", None), "vlm", None)
    hist = getattr(pipe, "frame_history", None)
    n = int(getattr(cfg, "history_frames", 0) or 0) if cfg is not None else 0
    now = getattr(out, "t", None)
    if n <= 0 or hist is None or not isinstance(now, (int, float)):
        return []
    try:
        return hist.pick(float(now), n, float(cfg.history_spacing_s))
    except Exception:
        return []


def crossing_answer(
    pipe: Any,
    img: np.ndarray,
    question: str = DEFAULT_CROSSING_QUESTION,
    *,
    out: Any = None,
    capture_ts: float | None = None,
    clock=time.monotonic,
) -> dict:
    """Run local reader + Muse Spark check and return the fused, validated answer.

    `out` may be a FrameOutput already computed for `img` (saves a detector pass).
    """
    capture_ts = clock() if capture_ts is None else capture_ts
    note = getattr(pipe, "note_crossing_check", None)
    if callable(note):  # opens a crossing context for the SAM signal fallback (boxes only)
        try:
            note()
        except Exception:
            pass
    if out is None:
        out = pipe.process(img, annotate=False)
    local = local_signal_state(out, img.shape)
    history = muse_history_frames(pipe, out)
    if history:
        muse_raw = pipe.vlm.ask_sync(img, question, history=history)
    else:  # single-frame fallback (history off, or no recent frames buffered)
        muse_raw = pipe.vlm.ask_sync(img, question)
    validated = validate_crossing_reply(
        muse_raw if isinstance(muse_raw, dict) and "error" not in muse_raw else None,
        capture_ts=capture_ts,
    )
    spoken = render_crossing_speech(validated, local_state=local, now=clock())
    fused = fuse_crossing_decision(local, validated.get("data") if validated.get("ok") else None)
    if not validated.get("ok"):
        spoken = SAFE_CROSSING_FALLBACK
        fused = {**fused, "decision": "unsure" if fused["decision"] == "go" else fused["decision"]}
    return {
        "muse_raw": muse_raw if isinstance(muse_raw, dict) else {},
        "local_light": local,
        "fuse": fused,
        "spoken": spoken,
        "validated": bool(validated.get("ok")),
        "capture_ts": capture_ts,
    }
