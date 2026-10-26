"""Walkable-surface understanding: road vs sidewalk under/ahead of the walker, plus poles.

Uses SegFormer-B0 fine-tuned on Cityscapes (19 classes incl. road, sidewalk, pole), via
Hugging Face transformers. It is the slowest local stage, so the pipeline runs it every N
frames and reuses the last result. Optional: if transformers is missing, the pipeline skips it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROAD, SIDEWALK, POLE, TERRAIN = 0, 1, 5, 9

# BGR colours for the overlay.
PALETTE = {ROAD: (128, 64, 128), SIDEWALK: (232, 35, 244), POLE: (0, 220, 255), TERRAIN: (35, 142, 107)}


@dataclass
class SurfaceResult:
    labels: np.ndarray  # HxW int class map at frame resolution
    underfoot: str  # "road" | "sidewalk" | "terrain" | "unknown"
    road_frac: float
    sidewalk_frac: float
    pole_ahead: bool


def summarise_labels(labels: np.ndarray) -> SurfaceResult:
    """Pure function over a class map (unit-testable without the network)."""
    H, W = labels.shape
    foot = labels[int(H * 0.8):, int(W * 0.35):int(W * 0.65)]  # the patch just ahead of the feet
    n = foot.size or 1
    road = float((foot == ROAD).sum()) / n
    side = float((foot == SIDEWALK).sum()) / n
    terr = float((foot == TERRAIN).sum()) / n
    if max(road, side, terr) < 0.35:
        underfoot = "unknown"
    else:
        underfoot = ["road", "sidewalk", "terrain"][int(np.argmax([road, side, terr]))]
    corridor = labels[int(H * 0.35):, int(W * 0.4):int(W * 0.6)]
    # A pole "ahead" = a pole region in the central corridor that reaches the lower half.
    pole_cols = (corridor == POLE).sum(axis=0)
    pole_ahead = bool(pole_cols.max(initial=0) > corridor.shape[0] * 0.35)
    return SurfaceResult(labels, underfoot, road, side, pole_ahead)


class SurfaceSegmenter:
    def __init__(self, model_id: str, input_width: int = 512, device: str = "cpu"):
        import torch
        from transformers import SegformerForSemanticSegmentation

        self.torch = torch
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
        self.input_width = input_width
        self.device = device
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, frame_bgr: np.ndarray) -> SurfaceResult:
        import cv2

        h, w = frame_bgr.shape[:2]
        iw = self.input_width
        ih = int(round(h * iw / w / 32) * 32)
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255 - self.mean) / self.std
        t = self.torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(pixel_values=t).logits  # 1 x 19 x ih/4 x iw/4
        lab = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
        return summarise_labels(lab)
