"""Walkable-surface understanding: road vs sidewalk under/ahead of the walker, plus poles.

Uses SegFormer-B0 fine-tuned on Cityscapes (19 classes incl. road, sidewalk, pole).
Default inference backend is OpenVINO (auto-converted from the HF PyTorch weights on
first run into models/segformer_b0_cityscapes_openvino/). PyTorch remains selectable;
failures fall back with a clear log line. The pipeline runs it every N frames.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROAD, SIDEWALK, POLE, TERRAIN = 0, 1, 5, 9

# BGR colours for the overlay.
PALETTE = {ROAD: (128, 64, 128), SIDEWALK: (232, 35, 244), POLE: (0, 220, 255), TERRAIN: (35, 142, 107)}

DEFAULT_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
DEFAULT_OV_DIR = Path("models/segformer_b0_cityscapes_openvino")
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def normalize_backend(name: str) -> str:
    n = (name or "openvino").strip().lower()
    if n in ("pytorch", "pt", "torch"):
        return "torch"
    if n in ("openvino", "ov"):
        return "openvino"
    raise ValueError(f"unknown surface backend {name!r}; expected 'openvino' or 'torch'")


def openvino_available() -> bool:
    try:
        import openvino  # noqa: F401
        return True
    except ImportError:
        return False


def ov_cache_dir(model_id: str, cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    # Stable short name for the default HF id; otherwise slug the id.
    if model_id == DEFAULT_MODEL_ID:
        return DEFAULT_OV_DIR
    slug = model_id.replace("/", "_").replace(" ", "_")
    return Path("models") / f"{slug}_openvino"


def _load_torch_segformer(model_id: str, device: str = "cpu"):
    import torch
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    return torch, model


def ensure_openvino_segformer(
    model_id: str = DEFAULT_MODEL_ID,
    input_width: int = 512,
    cache_dir: str | Path | None = None,
) -> Path:
    """Convert HF SegFormer to OpenVINO IR on first use. Returns the cache directory."""
    import openvino as ov
    import torch

    out_dir = ov_cache_dir(model_id, cache_dir)
    xml = out_dir / "segformer.xml"
    if xml.is_file() and xml.stat().st_size > 0:
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[surface] exporting OpenVINO SegFormer -> {out_dir} (first run)", file=sys.stderr)
    t0 = time.perf_counter()
    _, model = _load_torch_segformer(model_id, device="cpu")

    class _Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).logits

    wrapped = _Wrapper(model)
    # Representative size: width=input_width, height multiple of 32 (16:9-ish).
    ih = int(round(input_width * 9 / 16 / 32) * 32) or 32
    example = torch.zeros(1, 3, ih, input_width, dtype=torch.float32)
    traced = torch.jit.trace(wrapped, example, strict=False)
    ov_model = ov.convert_model(
        traced,
        example_input=example,
        input=[ov.PartialShape([1, 3, -1, -1])],
    )
    ov.save_model(ov_model, str(xml))
    elapsed = time.perf_counter() - t0
    print(f"[surface] OpenVINO SegFormer export done in {elapsed:.1f}s", file=sys.stderr)
    return out_dir


def _preprocess(frame_bgr: np.ndarray, input_width: int) -> tuple[np.ndarray, int, int]:
    import cv2

    h, w = frame_bgr.shape[:2]
    iw = input_width
    ih = int(round(h * iw / w / 32) * 32) or 32
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    return x.transpose(2, 0, 1)[None].astype(np.float32), h, w


def _labels_from_logits(logits: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    import cv2

    # logits: 1 x C x h' x w'  or  C x h' x w'
    if logits.ndim == 4:
        lab = logits.argmax(1)[0].astype(np.uint8)
    else:
        lab = logits.argmax(0).astype(np.uint8)
    return cv2.resize(lab, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


class SurfaceSegmenter:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        input_width: int = 512,
        device: str = "cpu",
        backend: str = "openvino",
        cache_dir: str | Path | None = None,
    ):
        self.model_id = model_id
        self.input_width = input_width
        self.device = device
        self.backend = normalize_backend(backend)
        self._ov_cache = cache_dir
        self._torch = None
        self._torch_model = None
        self._ov_infer = None
        self._ov_input = None

        requested = self.backend
        if self.backend == "openvino":
            if not openvino_available():
                print("[surface] openvino not installed; falling back to PyTorch", file=sys.stderr)
                self.backend = "torch"
            else:
                try:
                    self._init_openvino()
                    print(f"[surface] backend=openvino cache={ov_cache_dir(model_id, self._ov_cache)}", file=sys.stderr)
                    return
                except Exception as e:
                    print(
                        f"[surface] OpenVINO load/export failed ({type(e).__name__}: {e}); "
                        f"falling back to PyTorch",
                        file=sys.stderr,
                    )
                    self.backend = "torch"

        self._init_torch()
        if requested != self.backend:
            print(f"[surface] backend=torch (fallback) model={model_id}", file=sys.stderr)
        else:
            print(f"[surface] backend=torch model={model_id}", file=sys.stderr)

    def _init_torch(self) -> None:
        self._torch, self._torch_model = _load_torch_segformer(self.model_id, self.device)

    def _init_openvino(self) -> None:
        import openvino as ov

        cache = ensure_openvino_segformer(self.model_id, self.input_width, self._ov_cache)
        xml = cache / "segformer.xml"
        core = ov.Core()
        compiled = core.compile_model(str(xml), "CPU")
        self._ov_infer = compiled.create_infer_request()
        self._ov_input = compiled.input(0)

    def _forward_logits(self, x: np.ndarray) -> np.ndarray:
        if self.backend == "openvino":
            self._ov_infer.infer({self._ov_input: x})
            return np.asarray(self._ov_infer.get_output_tensor(0).data)
        t = self._torch.from_numpy(x).to(self.device)
        with self._torch.inference_mode():
            return self._torch_model(pixel_values=t).logits.detach().cpu().numpy()

    def __call__(self, frame_bgr: np.ndarray) -> SurfaceResult:
        x, h, w = _preprocess(frame_bgr, self.input_width)
        logits = self._forward_logits(x)
        lab = _labels_from_logits(logits, w, h)
        return summarise_labels(lab)
