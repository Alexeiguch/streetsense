"""Slow reasoning + SAM + voice helpers on Meta Model API (https://api.meta.ai/v1).

- MetaVisionAssistant: Muse Spark vision for scene JSON / "can I cross?" (background + sync).
- MetaSamSegmenter: text-prompted SAM 3.1 boxes (and optional masks) for crosswalk/curb/etc.
- MuseVoiceTranscribe: optional file STT for spoken "can I cross?" in the web demo.
- fuse_crossing_decision: never say go if local light OR Muse says stop/unsure.

The API key is read from MODEL_API_KEY (or LLAMA_API_KEY) and never logged.
Token / SAM / voice usage is appended to outputs/meta_usage.jsonl with a hard budget cap.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .config import VisionLLMConfig
from .meta_usage import (
    BudgetExceeded,
    UsageRecord,
    estimate_cost,
    get_ledger,
    mask_secret,
    usage_from_openai_response,
)

SYSTEM_PROMPT = (
    "You are the eyes of a blind pedestrian. Look at the camera frame (taken from their chest or "
    "glasses, facing forward). Reply ONLY with JSON: "
    '{"summary": "<= 15 words, most safety-relevant first", '
    '"hazards": ["short phrases"], '
    '"crossing_visible": true|false, '
    '"pedestrian_signal": "red"|"green"|"flashing"|"countdown"|"none_visible", '
    '"safe_to_cross": "yes"|"no"|"unsure"}. '
    "Never say safe_to_cross=yes unless a pedestrian green/walk signal is clearly visible and no "
    "vehicle is moving toward the crossing. When in doubt, say unsure."
)

# Local HSV reader states that mean "do not walk".
_LOCAL_STOP = frozenset({"red", "yellow", "unknown", "countdown", "green", "none"})
_LOCAL_GO = frozenset({"walk"})  # vehicle "green" is NOT a ped walk signal
_MUSE_STOP = frozenset({"no", "unsure", None, ""})


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


def fuse_crossing_decision(local_state: str | None, muse: dict | None) -> dict:
    """Safety fusion: never say go if either source says stop or unsure.

    Returns {decision: "go"|"stop"|"unsure", reason, local, muse_safe, muse_signal}.
    """
    muse = muse or {}
    local = (local_state or "unknown").lower().strip()
    muse_safe = str(muse.get("safe_to_cross") or "unsure").lower().strip()
    muse_sig = str(muse.get("pedestrian_signal") or "none_visible").lower().strip()

    local_go = local in _LOCAL_GO
    local_stop = local in _LOCAL_STOP or local not in _LOCAL_GO
    muse_go = muse_safe == "yes"
    muse_block = muse_safe in _MUSE_STOP or muse_sig in {"red", "flashing", "none_visible"}

    if local_go and muse_go and not muse_block:
        decision = "go"
        reason = "local walk/green and Muse yes"
    elif local_stop or muse_block or not muse_go:
        # Prefer stop over unsure when either side is negative.
        if (local == "red") or muse_safe == "no" or muse_sig == "red":
            decision = "stop"
            reason = "local or Muse indicates stop"
        elif local in {"unknown", "countdown"} or muse_safe == "unsure" or muse_sig == "none_visible":
            decision = "unsure"
            reason = "local or Muse unsure / no clear walk signal"
        else:
            decision = "stop"
            reason = "disagree or incomplete evidence"
    else:
        decision = "unsure"
        reason = "incomplete evidence"

    # Hard rule restated: never go unless BOTH agree go.
    if decision == "go" and not (local_go and muse_go):
        decision = "unsure"
        reason = "fusion refused go without dual agreement"

    return {
        "decision": decision,
        "reason": reason,
        "local": local,
        "muse_safe": muse_safe,
        "muse_signal": muse_sig,
    }


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
        self.last_usage: UsageRecord | None = None
        self.max_tokens = int(getattr(cfg, "max_tokens", 512) or 512)
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
        try:
            get_ledger().check_budget(0.01)  # coarse pre-check (~1c)
            return self._call(encode_jpeg_data_url(frame_bgr, self.cfg.max_image_side), question)
        except BudgetExceeded as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": mask_secret(f"{type(e).__name__}: {e}", self.cfg.api_key)[:200]}

    def _call(self, img_url: str, question: str, retries: int = 2) -> dict:
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                # Muse Spark always reasons; reasoning tokens share max_tokens with the
                # visible answer. Use minimal effort + enough headroom so JSON is not empty.
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
                    max_tokens=self.max_tokens,
                    reasoning_effort=getattr(self.cfg, "reasoning_effort", "minimal"),
                )
                latency = time.perf_counter() - t0
                content = resp.choices[0].message.content
                data = parse_json_reply(content)
                data["_latency_s"] = round(latency, 3)
                data["_model"] = self.cfg.model
                try:
                    rec = usage_from_openai_response(
                        resp, model=self.cfg.model, kind="chat", latency_s=latency, note=question[:80]
                    )
                    get_ledger().record(rec)
                    self.last_usage = rec
                    data["_usage"] = {
                        "prompt_tokens": rec.prompt_tokens,
                        "completion_tokens": rec.completion_tokens,
                        "estimated_cost_usd": rec.estimated_cost_usd,
                        "ledger_total_usd": round(get_ledger().total_usd, 6),
                    }
                except BudgetExceeded:
                    raise
                except Exception:
                    pass
                return data
            except BudgetExceeded:
                raise
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # Retry transient errors; don't retry 4xx (except 429).
                retryable = ("429" in msg) or ("timeout" in msg) or ("503" in msg) or ("502" in msg)
                if attempt < retries and retryable:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise
        assert last_err is not None
        raise last_err

    def _run(self, img_url: str, question: str) -> None:
        t0 = time.monotonic()
        try:
            data = self._call(img_url, question)
            self.latest = SceneReport(time.time(), time.monotonic() - t0, data, question)
            self.error = None
        except Exception as e:  # network / quota errors must never crash the fast loop
            self.error = mask_secret(f"{type(e).__name__}: {str(e)[:120]}", self.cfg.api_key)
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


def parse_sam_result(text: str) -> dict:
    """Parse SAM wire text via meta_sam_parser when available; fall back to boxes only."""
    boxes = parse_sam_boxes(text)
    masks: list[dict] = []
    try:
        from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format

        fmt = image_segmentation_format()
        parser = fmt.create_parser()
        parser.push(text or "")
        fin = parser.finish(CompletedOutcome())
        for rec in fin.result.records:
            kind = getattr(rec, "kind", None)
            if kind == "box":
                continue
            if kind == "mask":
                raster = decode_mask_to_raster(rec.mask)
                h, w = rec.mask.height, rec.mask.width
                arr = np.frombuffer(raster, dtype=np.uint8).reshape(h, w)
                b = rec.bounds
                masks.append({
                    "id": int(rec.object_id) if str(rec.object_id).isdigit() else rec.object_id,
                    "raster": arr,
                    "bounds": (int(b.left), int(b.top), int(b.right), int(b.bottom)),
                    "confidence": rec.confidence,
                })
    except Exception:
        pass
    return {"boxes": boxes, "masks": masks, "raw": text or ""}


@dataclass
class SamSegment:
    prompt: str
    boxes: list[dict] = field(default_factory=list)
    masks: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    cost_usd: float = 0.0
    raw: str = ""


class MetaSamSegmenter:
    """Text-prompted SAM 3.1 over POST /v1/responses. One concept per request."""

    DEFAULT_PROMPTS = ("crosswalk", "curb", "tactile paving", "sidewalk")

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.meta.ai/v1",
        model: str = "sam-3.1",
        timeout_s: float = 60.0,
        max_image_side: int = 768,
    ):
        import os

        self.api_key = api_key or os.environ.get("MODEL_API_KEY") or os.environ.get("LLAMA_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_image_side = max_image_side
        self.enabled = bool(self.api_key)
        self._client = None
        if self.enabled:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout_s)

    def segment(self, frame_bgr: np.ndarray, prompt: str, *, decode_masks: bool = True) -> SamSegment:
        if not self.enabled:
            raise RuntimeError("Meta SAM disabled: set MODEL_API_KEY")
        get_ledger().check_budget(estimate_cost(self.model, images=1))
        img_url = encode_jpeg_data_url(frame_bgr, self.max_image_side, quality=85)
        t0 = time.perf_counter()
        # Non-streaming is fine for stills; streaming recommended for video.
        resp = self._client.responses.create(
            model=self.model,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": img_url},
                    ],
                }
            ],
            metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
        )
        latency = time.perf_counter() - t0
        text = _responses_output_text(resp)
        parsed = parse_sam_result(text) if decode_masks else {"boxes": parse_sam_boxes(text), "masks": [], "raw": text}
        cost = estimate_cost(self.model, images=1)
        # Prefer usage.video_frames / images from response when present.
        usage = getattr(resp, "usage", None)
        if usage is not None:
            vf = int(getattr(usage, "video_frames_processed", 0) or 0)
            if vf:
                cost = estimate_cost(self.model, video_frames=vf)
        rec = UsageRecord(
            ts=time.time(),
            model=self.model,
            kind="sam",
            images=1,
            latency_s=round(latency, 3),
            estimated_cost_usd=round(cost, 6),
            note=f"prompt={prompt}",
        )
        get_ledger().record(rec)
        return SamSegment(
            prompt=prompt,
            boxes=parsed["boxes"],
            masks=parsed["masks"] if decode_masks else [],
            latency_s=round(latency, 3),
            cost_usd=round(cost, 6),
            raw=parsed.get("raw") or text,
        )

    def segment_concepts(
        self, frame_bgr: np.ndarray, prompts: tuple[str, ...] | list[str] | None = None
    ) -> list[SamSegment]:
        prompts = tuple(prompts or self.DEFAULT_PROMPTS)
        return [self.segment(frame_bgr, p) for p in prompts]


def _responses_output_text(resp: Any) -> str:
    """Pull concatenated output_text from a Responses API result."""
    # SDK shapes vary; try common fields.
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    chunks: list[str] = []
    for item in getattr(resp, "output", None) or []:
        for c in getattr(item, "content", None) or []:
            t = getattr(c, "text", None)
            if t:
                chunks.append(t)
            elif isinstance(c, dict) and c.get("text"):
                chunks.append(c["text"])
    if chunks:
        return "".join(chunks)
    # dict-like fallback
    if isinstance(resp, dict):
        if resp.get("output_text"):
            return resp["output_text"]
    return ""


def overlay_sam_segments(
    frame_bgr: np.ndarray,
    segments: list[SamSegment],
    colors: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw SAM boxes (and mask rasters when present) onto a copy of the frame."""
    colors = colors or {
        "crosswalk": (0, 255, 255),
        "curb": (255, 128, 0),
        "tactile paving": (255, 0, 255),
        "sidewalk": (0, 200, 0),
    }
    out = frame_bgr.copy()
    overlay = out.copy()
    for seg in segments:
        color = colors.get(seg.prompt, (200, 200, 200))
        for m in seg.masks:
            raster = m["raster"]
            x1, y1, x2, y2 = m["bounds"]
            # Place raster into bounds region (may be smaller than box).
            rh, rw = raster.shape[:2]
            x2 = min(out.shape[1], x1 + rw)
            y2 = min(out.shape[0], y1 + rh)
            region = overlay[y1:y2, x1:x2]
            mask = raster[: y2 - y1, : x2 - x1] > 0
            if mask.any():
                region[mask] = (0.45 * np.array(color, dtype=np.float32) + 0.55 * region[mask]).astype(np.uint8)
        for b in seg.boxes:
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, seg.prompt, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if any(s.masks for s in segments):
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
    return out


class MuseVoiceTranscribe:
    """File transcription via POST /v1/asr/transcribe (WAV mono 16/24 kHz PCM)."""

    def __init__(self, api_key: str | None = None, model: str = "muse-voice-transcribe-1.0"):
        import os

        self.api_key = api_key or os.environ.get("MODEL_API_KEY") or os.environ.get("LLAMA_API_KEY")
        self.model = model
        self.enabled = bool(self.api_key)
        self.base_url = "https://api.meta.ai/v1"

    def transcribe_wav(self, wav_path: str, *, mode: str = "PUSH_TO_TALK") -> dict:
        if not self.enabled:
            return {"error": "Meta Voice disabled: set MODEL_API_KEY"}
        import requests

        get_ledger().check_budget(0.01)
        t0 = time.perf_counter()
        with open(wav_path, "rb") as f:
            r = requests.post(
                f"{self.base_url}/asr/transcribe",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={
                    "request": (
                        None,
                        json.dumps({
                            "mode": mode,
                            "model": self.model,
                            "audioEncoding": "WAV",
                        }),
                        "application/json",
                    ),
                    "audio": (wav_path, f, "audio/wav"),
                },
                timeout=60,
            )
        latency = time.perf_counter() - t0
        if r.status_code >= 400:
            return {"error": mask_secret(f"HTTP {r.status_code}: {r.text[:200]}", self.api_key), "latency_s": latency}
        data = r.json()
        audio_ms = float(data.get("audioDurationMs") or 0)
        audio_s = audio_ms / 1000.0
        cost = estimate_cost(self.model, audio_seconds=audio_s)
        get_ledger().record(
            UsageRecord(
                ts=time.time(),
                model=self.model,
                kind="voice",
                audio_seconds=audio_s,
                latency_s=round(latency, 3),
                estimated_cost_usd=round(cost, 6),
                note="transcribe_wav",
            )
        )
        return {
            "transcript": data.get("transcript", ""),
            "turns": data.get("turns") or [],
            "audio_duration_ms": audio_ms,
            "latency_s": round(latency, 3),
            "estimated_cost_usd": round(cost, 6),
        }


def list_models(api_key: str | None = None, base_url: str = "https://api.meta.ai/v1") -> list[str]:
    import os

    from openai import OpenAI

    key = api_key or os.environ.get("MODEL_API_KEY") or os.environ.get("LLAMA_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY not set")
    client = OpenAI(base_url=base_url, api_key=key, timeout=30)
    return sorted(m.id for m in client.models.list().data)



def sam_boxes_to_crosswalk(boxes: list[dict], frame_shape: tuple[int, ...]):
    """Turn SAM crosswalk boxes into a CrosswalkResult for the classical path to merge with.

    Returns None when there are no boxes. Score uses SAM confidence when present.
    """
    from .crosswalk import CrosswalkResult
    from .types import Box

    if not boxes:
        return None
    H, W = frame_shape[:2]
    # Prefer highest confidence, else largest area.
    def key(b):
        area = max(0, (b.get("x2", 0) - b.get("x1", 0))) * max(0, (b.get("y2", 0) - b.get("y1", 0)))
        return (float(b.get("c") or 0.0), area)

    b = max(boxes, key=key)
    # SAM w/h are source dims of the (possibly downscaled) request image; scale to frame.
    sw = float(b.get("w") or W) or W
    sh = float(b.get("h") or H) or H
    sx, sy = W / sw, H / sh
    box = Box(
        int(b["x1"] * sx), int(b["y1"] * sy),
        int(b["x2"] * sx), int(b["y2"] * sy),
    ).clip(W, H)
    return CrosswalkResult(
        found=True,
        box=box,
        n_stripes=0,
        score=float(b.get("c") or 0.5),
        stripes=[],
    )
