"""Fast local detector: Ultralytics YOLO (COCO) with optional ByteTrack ids."""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import DetectorConfig
from .types import Box, Detection

# COCO classes that matter on a sidewalk. Everything else is ignored.
KEEP = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train", "traffic light",
    "stop sign", "fire hydrant", "parking meter", "bench", "dog", "chair", "potted plant", "suitcase",
}


class YoloDetector:
    def __init__(self, cfg: DetectorConfig):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        self.names = self.model.names

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        kw = dict(imgsz=self.cfg.imgsz, conf=self.cfg.conf, iou=self.cfg.iou, device=self.cfg.device, verbose=False)
        if self.cfg.track:
            res = self.model.track(frame_bgr, persist=True, tracker="bytetrack.yaml", **kw)[0]
        else:
            res = self.model.predict(frame_bgr, **kw)[0]
        out: list[Detection] = []
        if res.boxes is None:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(cls)
        for b, c, p, tid in zip(xyxy, cls, conf, ids):
            label = self.names[int(c)]
            if label not in KEEP:
                continue
            out.append(Detection(label, Box(*map(float, b)), float(p), None if tid is None else int(tid)))
        return out


class ApproachTracker:
    """Flags tracks whose box height grows quickly (object getting closer).

    growth = h_now / h_then over roughly `window_s`; > `ratio` means approaching.
    """

    def __init__(self, window_s: float = 1.0, ratio: float = 1.25, max_tracks: int = 256):
        self.window_s = window_s
        self.ratio = ratio
        self.max_tracks = max_tracks
        self._hist: dict[int, deque[tuple[float, float]]] = {}

    def update(self, t: float, detections: list[Detection]) -> set[int]:
        approaching: set[int] = set()
        seen = set()
        for d in detections:
            if d.track_id is None:
                continue
            seen.add(d.track_id)
            q = self._hist.setdefault(d.track_id, deque(maxlen=64))
            q.append((t, d.box.h))
            while len(q) > 2 and t - q[1][0] >= self.window_s:
                q.popleft()
            t0, h0 = q[0]
            if t - t0 >= self.window_s * 0.5 and h0 > 0 and d.box.h / h0 >= self.ratio:
                approaching.add(d.track_id)
        if len(self._hist) > self.max_tracks:
            for k in [k for k in self._hist if k not in seen][: len(self._hist) - self.max_tracks]:
                del self._hist[k]
        return approaching
