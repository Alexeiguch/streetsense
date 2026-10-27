"""Low-frequency open-vocabulary hazard detector (YOLO-World).

COCO misses bollards, kerbs, scooters, construction barriers, open manholes, etc.
This module runs a promptable model every N frames and returns only non-COCO
hazard labels so the fast YOLO loop stays unchanged. Optional: off by default
until measured useful on real hazard footage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .types import Box, Detection

# Prompts tuned for sidewalk hazards. Keep short; long lists dilute YOLO-World.
DEFAULT_PROMPTS = [
    "bollard",
    "traffic cone",
    "scooter",
    "electric scooter",
    "construction barrier",
    "barrier",
    "stairs",
    "step",
    "pothole",
    "open manhole",
    "manhole",
    "kerb",
    "curb",
]

# Labels already covered by the fast COCO detector — drop them from open-vocab output.
COCO_OVERLAP = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "traffic light", "stop sign", "fire hydrant", "parking meter", "bench",
    "dog", "chair", "potted plant", "suitcase", "pole",
}


class OpenVocabHazardDetector:
    """YOLO-World wrapper. Lazy-loads weights on first call."""

    def __init__(
        self,
        weights: str = "models/yolov8s-worldv2.pt",
        prompts: list[str] | None = None,
        conf: float = 0.10,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        self.weights = weights
        self.prompts = list(prompts or DEFAULT_PROMPTS)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        from ultralytics import YOLO

        path = Path(self.weights)
        name = path.name if path.exists() else (path.name if path.suffix else "yolov8s-worldv2.pt")
        print(f"[open-vocab] loading {name} prompts={self.prompts}", file=sys.stderr)
        model = YOLO(str(path) if path.exists() else name)
        model.set_classes(self.prompts)
        self._model = model
        self.names = model.names

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        self._ensure()
        res = self._model.predict(
            frame_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )[0]
        out: list[Detection] = []
        if res.boxes is None:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        for b, c, p in zip(xyxy, cls, conf):
            label = self.names[int(c)]
            # Normalise a few aliases
            canon = label.lower().strip()
            if canon in ("curb",):
                canon = "kerb"
            if canon in ("electric scooter",):
                canon = "scooter"
            if canon in ("open manhole",):
                canon = "manhole"
            if canon in COCO_OVERLAP:
                continue
            out.append(
                Detection(
                    canon,
                    Box(*map(float, b)),
                    float(p),
                    None,
                    attrs={"source": "open_vocab"},
                )
            )
        return out
