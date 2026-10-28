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
    #   --weights models/yolo11n.pt | models/yolo11m.pt | models/rtdetr-l.pt | rfdetr-nano | models/yolo26n.pt
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
    conf: float = 0.10  # measured: scooters often score 0.05–0.25 on sidewalk clips
    imgsz: int = 640


@dataclass
class DepthConfig:
    """Monocular depth (Depth Anything V2 Small, Apache-2.0). Off by default; --depth enables."""

    enabled: bool = False
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    backend: str = "openvino"  # openvino|torch
    every_n_frames: int = 6
    input_width: int = 518
    prefer_over_box: bool = True  # use calibrated depth metres when available



@dataclass
class SamConfig:
    """On-demand / low-frequency SAM 3.1 text prompts (crosswalk, curb, ...). Off by default."""

    enabled: bool = False
    model: str = "sam-3.1"
    interval_s: float = 8.0  # slow loop; classical CV stays on the fast path
    max_image_side: int = 640
    prompts: tuple[str, ...] = ("crosswalk",)  # curb/sidewalk/tactile on demand via CLI
    timeout_s: float = 60.0


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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return max(0, int(raw)) if raw else default
    except ValueError:
        return default


@dataclass
class VisionLLMConfig:
    """Slow reasoning loop on Meta Model API (OpenAI-compatible)."""

    enabled: bool = True
    base_url: str = "https://api.meta.ai/v1"
    model: str = "muse-spark-1.3"
    interval_s: float = 10.0
    timeout_s: float = 30.0
    max_image_side: int = 512  # keep images small for cost/latency
    # Must cover reasoning + JSON (docs/META.md). At 512, 1-3 of 40 crossing replies were cut
    # off (invalid JSON -> "can't tell"); at 768, 0 of 40 (longest 659). docs/MUSE_EVAL.md
    max_tokens: int = field(default_factory=lambda: _env_int("STREETSENSE_MUSE_MAX_TOKENS", 768) or 768)
    reasoning_effort: str = "minimal"  # shortest Muse Spark pass; "none" is 400
    # Crossing prompt: "describe" (look at the ped head first) or "strict" (v0). docs/MUSE_EVAL.md
    prompt_variant: str = field(
        default_factory=lambda: os.environ.get("STREETSENSE_MUSE_PROMPT", "describe").strip() or "describe"
    )
    # On-demand questions (ask_sync) send a larger image: at 512 px the DC walk figure is a few
    # pixels and Muse read it as a red hand on 10/10 walk frames; at 1024 px it read 9/10 right.
    crossing_max_image_side: int = field(
        default_factory=lambda: _env_int("STREETSENSE_MUSE_CROSSING_SIDE", 1024) or 1024
    )
    # Older frames sent with a crossing question (0 = single frame). Spaced history_spacing_s apart.
    history_frames: int = field(default_factory=lambda: _env_int("STREETSENSE_MUSE_HISTORY_FRAMES", 0))
    history_spacing_s: float = 0.5

    @property
    def api_key(self) -> str | None:
        # MODEL_API_KEY is the name Meta's docs use; LLAMA_API_KEY kept as an alias.
        return os.environ.get("MODEL_API_KEY") or os.environ.get("LLAMA_API_KEY")


@dataclass
class WonderConfig:
    """Opt-in Muse Wonder guide agent (slow, on demand, never on the safety path)."""

    enabled: bool = False  # opt in: CLI `guide`, or STREETSENSE_WONDER=1 for the web /guide route
    base_url: str = "https://api.meta.ai/v1"
    model: str = "muse-wonder-1.0"
    reasoning_effort: str = "low"  # minimal | low | medium | high | max
    max_turns: int = 4  # model turns per question, our own loop cap
    max_output_tokens: int = 1500  # reasoning + visible text share this
    max_cost_usd: float = 0.25  # per-question hard stop on top of the global ledger cap
    turn_estimate_usd: float = 0.03  # pre-check per turn against the global budget
    max_segment_calls: int = 2  # SAM calls per question (each costs $0.0025)
    max_image_side: int = 512
    timeout_s: float = 90.0
    # True: resend the whole conversation each turn. False: chain with previous_response_id.
    stateless: bool = True

    @property
    def api_key(self) -> str | None:
        return os.environ.get("MODEL_API_KEY") or os.environ.get("LLAMA_API_KEY")

    @classmethod
    def from_env(cls) -> "WonderConfig":
        flag = os.environ.get("STREETSENSE_WONDER", "").strip().lower()
        return cls(enabled=flag in {"1", "true", "yes", "on"})


@dataclass
class MapsConfig:
    """Opt-in walking navigation. Map data is context only and never authorises a crossing."""

    enabled: bool = False  # STREETSENSE_MAPS=1, or the CLI walk-sim command
    # osm: Valhalla + Overpass + Nominatim (free, speakable). fixture: recorded OSM responses, offline.
    provider: str = "osm"
    fixtures_dir: str | None = None  # default tests/fixtures/maps
    # Google Maps via Maps Experience agent tools, only for Wonder place questions. Needs GOOGLE_MAPS_API_KEY.
    google: bool = False

    @property
    def google_key_present(self) -> bool:
        return bool(os.environ.get("GOOGLE_MAPS_API_KEY"))

    @classmethod
    def from_env(cls) -> "MapsConfig":
        def on(name: str) -> bool:
            return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

        return cls(enabled=on("STREETSENSE_MAPS"), provider=os.environ.get("STREETSENSE_MAPS_PROVIDER", "osm"),
                   fixtures_dir=os.environ.get("STREETSENSE_MAPS_FIXTURES"), google=on("STREETSENSE_MAPS_GOOGLE"))


@dataclass
class Config:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    open_vocab: OpenVocabConfig = field(default_factory=OpenVocabConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    sam: SamConfig = field(default_factory=SamConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    vlm: VisionLLMConfig = field(default_factory=VisionLLMConfig)
    wonder: WonderConfig = field(default_factory=WonderConfig)
    maps: MapsConfig = field(default_factory=MapsConfig)
    crosswalk_every_n_frames: int = 2
