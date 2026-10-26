"""Slow reasoning loop on Meta Model API (OpenAI-compatible, https://api.meta.ai/v1).

- MetaVisionAssistant: sends a frame to Muse Spark (vision) for a short scene summary or a
  "can I cross?" judgement, as strict JSON. Runs in a background thread so it never blocks
  the fast loop. Disabled (no-op) when MODEL_API_KEY is not set.
- parse_sam_boxes: parses SAM 3.1 segmentation wire output (boxes; masks kept as raw tokens)
  so SAM can later segment "sidewalk", "crosswalk", "curb" on demand.

The API key is read from the environment and never logged.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import VisionLLMConfig

SYSTEM_PROMPT = (
    "You are the eyes of a blind pedestrian. Look at the camera frame (taken from their chest or "
    "glasses, facing forward). Reply ONLY with JSON: "
    '{"summary": "<= 15 words, most safety-relevant first", '
    '"hazards": ["short phrases"], '
    '"crossing_visible": true|false, '
    '"pedestrian_signal": "red"|"green"|"flashing"|"none_visible", '
    '"safe_to_cross": "yes"|"no"|"unsure"}. '
    "Never say safe_to_cross=yes unless a pedestrian green/walk signal is clearly visible and no "
    "vehicle is moving toward the crossing. When in doubt, say unsure."
)


def encode_jpeg_data_url(frame_bgr: np.ndarray, max_side: int = 768, quality: int = 80) -> str:
    h, w = frame_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def parse_json_reply(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates code fences / prose)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {"summary": (text or "").strip()[:200]}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"summary": (text or "").strip()[:200]}


@dataclass
class SceneReport:
    ts: float
    latency_s: float
    data: dict
    question: str


class MetaVisionAssistant:
    def __init__(self, cfg: VisionLLMConfig):
        self.cfg = cfg
        self.enabled = bool(cfg.enabled and cfg.api_key)
        self._client = None
        self._busy = threading.Lock()
        self._last_request = -1e9
        self.latest: SceneReport | None = None
        self.error: str | None = None
        if self.enabled:
            from openai import OpenAI

            self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout_s)

    def maybe_request(self, frame_bgr: np.ndarray, now: float, question: str | None = None, force: bool = False) -> bool:
        """Fire a background request if enabled, idle, and the interval has passed (or forced)."""
        if not self.enabled:
            return False
        if not force and now - self._last_request < self.cfg.interval_s:
            return False
        if not self._busy.acquire(blocking=False):
            return False
        self._last_request = now
        img = encode_jpeg_data_url(frame_bgr, self.cfg.max_image_side)
        q = question or "Describe what matters for walking safely right now."
        threading.Thread(target=self._run, args=(img, q), daemon=True).start()
        return True

    def ask_sync(self, frame_bgr: np.ndarray, question: str) -> dict:
        if not self.enabled:
            return {"error": "Meta Model API disabled: set MODEL_API_KEY"}
        return self._call(encode_jpeg_data_url(frame_bgr, self.cfg.max_image_side), question)

    def _call(self, img_url: str, question: str) -> dict:
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                # Images are only accepted in user-role messages on Meta Model API.
                {"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ]},
            ],
        )
        return parse_json_reply(resp.choices[0].message.content)

    def _run(self, img_url: str, question: str) -> None:
        t0 = time.monotonic()
        try:
            data = self._call(img_url, question)
            self.latest = SceneReport(time.time(), time.monotonic() - t0, data, question)
            self.error = None
        except Exception as e:  # network / quota errors must never crash the fast loop
            self.error = f"{type(e).__name__}: {str(e)[:120]}"
        finally:
            self._busy.release()


# A record id is preceded either by the frame marker "f>" or by the previous mask's "|>,".
# Box fields only use [a-z0-9=;.], so a strict class keeps base85 mask payloads from matching.
_REC_RE = re.compile(r"(?:f>|\|>,)(\d+)<\|box;([a-z0-9=;.]+)\|>")
_FRAME_RE = re.compile(r"<(\d+)f>")


def _kv(s: str) -> dict:
    out = {}
    for part in s.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = float(v) if "." in v else int(v)
    return out


def parse_sam_boxes(text: str) -> list[dict]:
    """Parse SAM 3.1 wire lines into [{frame, id, x1, y1, x2, y2, w, h[, c]}].

    Mask payloads (base85 rasters) are skipped here; decode them with Meta's `meta_sam_parser`
    when pixel masks are needed.
    """
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        fm = _FRAME_RE.match(line)
        if not fm:
            continue
        frame = int(fm.group(1))
        for m in _REC_RE.finditer(line):
            rec = {"frame": frame, "id": int(m.group(1))}
            rec.update(_kv(m.group(2)))
            out.append(rec)
    return out
