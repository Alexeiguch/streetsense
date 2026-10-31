"""Tiny on-device classifier for traffic-light crops (part of the local light reader).

Classes: 0 walk (lit walk figure), 1 stop (lit red man / hand / red lamp),
2 countdown (digits, incl. walk + digits), 3 other (off, vehicle green, unclear, not a light).

The classifier is only one input to the local reader; it can never authorise crossing
on its own (see ``combine`` and docs/READER.md). Any doubt resolves to "not walk".
Disable with STREETSENSE_SIGNAL_CNN=0.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("walk", "stop", "countdown", "other")
SIZE = 64
MODELS = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR = MODELS / "signal_cnn"  # v1 (PTL + Commons stills + dev crops)
DEFAULT_VERSION = "v1"


def selected_model_dir() -> Path:
    """STREETSENSE_SIGNAL_CNN_VERSION=v1|v2|v3|v4_ptld picks models/signal_cnn or models/signal_cnn_vN (rollback = v1).

    v4_ptld is trained on PTLD (licence unconfirmed): opt-in experiments only, never the default, not for shipping.
    """
    v = (os.environ.get("STREETSENSE_SIGNAL_CNN_VERSION") or DEFAULT_VERSION).strip().lower()
    return MODEL_DIR if v in ("", "v1", "1") else MODELS / f"signal_cnn_{v if v.startswith('v') else 'v' + v}"


def enabled() -> bool:
    return os.environ.get("STREETSENSE_SIGNAL_CNN", "1").strip().lower() not in ("0", "false", "off", "no")


def preprocess(crop: np.ndarray, size: int = SIZE) -> np.ndarray:
    """BGR crop (any size) -> float32 (3,size,size): aspect-preserving letterbox on black, x/255-0.5."""
    h, w = crop.shape[:2]
    out = np.zeros((size, size, 3), np.uint8)
    if h < 1 or w < 1:
        return (out.astype(np.float32) / 255.0 - 0.5).transpose(2, 0, 1)
    s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = im
    return (out.astype(np.float32) / 255.0 - 0.5).transpose(2, 0, 1)


@dataclass
class SignalPrediction:
    probs: tuple[float, float, float, float]
    ms: float = 0.0

    @property
    def p_walk(self) -> float:
        return self.probs[0]

    @property
    def label(self) -> str:
        return CLASSES[int(np.argmax(self.probs))]


class SignalCNN:
    def __init__(self, path: Path | str | None = None):
        import openvino as ov

        model_dir = Path(path) if path else selected_model_dir()
        self.model_dir = model_dir
        self.meta = json.loads((model_dir / "signal_cnn.json").read_text())
        self.t_walk = float(self.meta["t_walk"])
        self.t_veto = float(self.meta.get("t_veto", 0.8))
        core = ov.Core()
        model = core.read_model(str(model_dir / "signal_cnn.onnx"))
        self._compiled = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 1})
        self._out = self._compiled.output(0)

    def __call__(self, crop: np.ndarray) -> SignalPrediction:
        t0 = time.perf_counter()
        x = preprocess(crop)[None]
        logits = np.asarray(self._compiled([x])[self._out], np.float64)[0]
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        return SignalPrediction(tuple(float(v) for v in p), (time.perf_counter() - t0) * 1000)


_INSTANCE: SignalCNN | None = None
_FAILED = False


def get() -> SignalCNN | None:
    """Shared instance, or None when disabled / model missing (reader then uses hand rules only)."""
    global _INSTANCE, _FAILED
    if not enabled() or _FAILED:
        return None
    if _INSTANCE is None:
        try:
            _INSTANCE = SignalCNN()
        except Exception as e:  # missing model file or runtime: never fatal, never walk
            print(f"[signal-cnn] disabled: {type(e).__name__}: {e}")
            _FAILED = True
            return None
    return _INSTANCE
