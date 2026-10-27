"""RF-DETR Nano detector (Apache-2.0, Roboflow) as a selectable StreetSense backend.

Class ids follow COCO 91-category indexing (1=person, 3=car, 10=traffic light, …).
We prefer the `class_name` field supervision returns; fall back to a COCO-91 map.
ByteTrack is not wired through RF-DETR here (no track ids); ApproachTracker simply
sees untracked boxes.
"""

from __future__ import annotations

import sys
from typing import Any

import cv2
import numpy as np

from .config import DetectorConfig
from .detector import KEEP
from .types import Box, Detection

# COCO 91-id → name (Detectron/DETR style; missing ids were unused in COCO)
COCO91 = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
    7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
    13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat",
    18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
    24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag",
    32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}


def coco91_name(class_id: int) -> str | None:
    """Map a COCO 91-category id (1..90) to a name."""
    return COCO91.get(int(class_id))


def resolve_label(class_id: int, class_name: str | None = None) -> str | None:
    if class_name:
        return str(class_name).strip().lower()
    return coco91_name(class_id)


class RfDetrDetector:
    """Drop-in callable matching YoloDetector's (frame) -> list[Detection] surface."""

    def __init__(self, cfg: DetectorConfig, variant: str = "nano"):
        self.cfg = cfg
        self.backend = "torch"  # RF-DETR package path; OV/ONNX tried separately in benches
        self.variant = variant
        self.model_path = f"rfdetr-{variant}"
        self.names = {i: n for i, n in COCO91.items()}
        self._model: Any = None
        self._load()

    def _load(self) -> None:
        from rfdetr import RFDETRNano, RFDETRSmall

        print(f"[detector] backend=rfdetr variant={self.variant} (Apache-2.0)", file=sys.stderr)
        if self.variant in ("small", "s"):
            self._model = RFDETRSmall()
        else:
            self._model = RFDETRNano()

    def _accept(self, label: str, conf: float) -> bool:
        by = getattr(self.cfg, "conf_by_class", None) or {}
        thr = float(by.get(label, self.cfg.conf))
        return conf >= thr

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # RF-DETR predict takes threshold; use the global floor so per-class can filter after
        by = getattr(self.cfg, "conf_by_class", None) or {}
        floor = min([float(self.cfg.conf), *[float(v) for v in by.values()]]) if by else float(self.cfg.conf)
        det = self._model.predict(rgb, threshold=floor)
        out: list[Detection] = []
        if det is None or det.class_id is None or len(det.class_id) == 0:
            return out
        names = None
        if getattr(det, "data", None) and "class_name" in det.data:
            names = det.data["class_name"]
        xyxy = det.xyxy
        for i, (cid, conf) in enumerate(zip(det.class_id, det.confidence)):
            label = resolve_label(int(cid), None if names is None else str(names[i]))
            if label is None or label not in KEEP:
                continue
            if not self._accept(label, float(conf)):
                continue
            x1, y1, x2, y2 = map(float, xyxy[i])
            out.append(Detection(label, Box(x1, y1, x2, y2), float(conf), None))
        return out
