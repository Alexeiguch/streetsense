"""Runtime configuration. Everything tunable lives here, so the pipeline has no magic numbers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# Per-class confidence overrides (applied after the global floor). Person/soft road users
# get a lower bar because recall was the measured weak point on the 14-frame hand sample.
DEFAULT_CONF_BY_CLASS: dict[str, float] = {
    "person": 0.28,
    "bicycle": 0.28,
    "motorcycle": 0.28,
    "dog": 0.28,
}


@dataclass
class DetectorConfig:
    # Default YOLO11s + OpenVINO (see docs/BENCHMARK.md §7). Alternatives:
    #   --weights models/yolo11n.pt | models/yolo11m.pt | models/rtdetr-l.pt | models/yolo26n.pt
    #   --backend torch
    weights: str = "models/yolo11s.pt"
    backend: str = "openvino"  # "openvino" (default) or "torch"; falls back to torch if OV unavailable
    imgsz: int = 640
    conf: float = 0.35  # global floor; class overrides in conf_by_class may go lower
    conf_by_class: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CONF_BY_CLASS))
    iou: float = 0.5
    device: str = "cpu"
    track: bool = True  # ByteTrack ids -> lets us tell "approaching" from "standing still"


@dataclass
class SurfaceConfig:
    enabled: bool = True
    model_id: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
    backend: str = "openvino"  # "openvino" (default) or "torch"; falls back to torch if OV unavailable
    every_n_frames: int = 6  # segmentation is the slowest local stage; refresh it periodically
    input_width: int = 512


@dataclass
class OpenVocabConfig:
    """Slow open-vocab hazard loop (YOLO-World). Off by default; enable with --open-vocab."""

    enabled: bool = False
    weights: str = "models/yolov8s-worldv2.pt"
    every_n_frames: int = 15  # ~0.5 Hz at 30 FPS input
    conf: float = 0.15
    imgsz: int = 640


@dataclass
class CameraConfig:
    # Rough pinhole model used for distance estimates. Phone main cameras are ~65-75 deg HFOV.
    hfov_deg: float = 70.0


@dataclass
class AlertConfig:
    global_min_gap_s: float = 1.5  # never speak two alerts closer than this (unless critical)
    default_cooldown_s: float = 6.0  # same alert key is not repeated within this window
    critical_cooldown_s: float = 2.5
    max_queue: int = 8


@dataclass
class VisionLLMConfig:
    """Slow reasoning loop on Meta Model API (OpenAI-compatible)."""

    enabled: bool = True
    base_url: str = "https://api.meta.ai/v1"
    model: str = "muse-spark-1.3"
    interval_s: float = 10.0
    timeout_s: float = 20.0
    max_image_side: int = 768

    @property
    def api_key(self) -> str | None:
        # MODEL_API_KEY is the name Meta's docs use; LLAMA_API_KEY kept as an alias.
        return os.environ.get("MODEL_API_KEY") or os.environ.get("LLAMA_API_KEY")


@dataclass
class Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    open_vocab: OpenVocabConfig = field(default_factory=OpenVocabConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    vlm: VisionLLMConfig = field(default_factory=VisionLLMConfig)
    crosswalk_every_n_frames: int = 2
