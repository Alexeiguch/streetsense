"""Open-vocab hazard detector: prompt list + COCO-overlap filtering."""

from __future__ import annotations

from unittest import mock

import numpy as np

from streetsense.open_vocab import COCO_OVERLAP, DEFAULT_PROMPTS, OpenVocabHazardDetector
from streetsense.types import Box, Detection


def test_default_prompts_cover_street_hazards():
    for need in ("bollard", "scooter", "pothole", "kerb", "stairs", "traffic cone"):
        assert need in DEFAULT_PROMPTS


def test_coco_overlap_drops_person_car():
    assert "person" in COCO_OVERLAP and "car" in COCO_OVERLAP
    assert "bollard" not in COCO_OVERLAP


def test_open_vocab_filters_coco_and_keeps_hazards(monkeypatch):
    det = OpenVocabHazardDetector(weights="models/yolov8s-worldv2.pt", conf=0.1)

    class FakeBoxes:
        def __init__(self):
            import torch
            # person, bollard, car, scooter
            self.xyxy = torch.tensor([[0, 0, 10, 20], [20, 0, 30, 40], [40, 0, 60, 30], [70, 0, 90, 50]], dtype=torch.float32)
            self.cls = torch.tensor([0, 1, 2, 3], dtype=torch.float32)
            self.conf = torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32)

    class FakeRes:
        boxes = FakeBoxes()
        names = {0: "person", 1: "bollard", 2: "car", 3: "scooter"}

    class FakeModel:
        names = FakeRes.names
        def set_classes(self, prompts): pass
        def predict(self, *a, **k): return [FakeRes()]

    det._model = FakeModel()
    det.names = FakeRes.names
    out = det(np.zeros((64, 64, 3), dtype=np.uint8))
    labels = [d.label for d in out]
    assert "bollard" in labels and "scooter" in labels
    assert "person" not in labels and "car" not in labels
