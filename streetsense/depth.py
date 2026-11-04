"""Monocular depth via Depth Anything V2 Small (Apache-2.0).

Relative depth is turned into metres by calibrating against box-height distance
estimates for person/car detections in the same frame (known real-world height).
Depth Anything V2 Small outputs larger values for nearer surfaces on our samples,
so metres ≈ scale / relative_depth.

OpenVINO is the default backend (auto-converts on first use); torch is selectable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .geometry import REAL_HEIGHT_M, estimate_distance_m
from .types import Box, Detection

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
OV_CACHE = Path("models/depth_anything_v2_small_openvino")
OV_INPUT_HW = (294, 518)  # H, W used for the traced OpenVINO graph
CALIB_LABELS = {"person", "car", "bicycle", "motorcycle", "bus", "truck"}


def openvino_available() -> bool:
    try:
        import openvino  # noqa: F401
        return True
    except ImportError:
        return False


def normalize_backend(name: str) -> str:
    n = (name or "openvino").strip().lower()
    if n in ("pytorch", "pt", "torch"):
        return "torch"
    if n in ("openvino", "ov"):
        return "openvino"
    raise ValueError(f"unknown depth backend {name!r}")


def ensure_openvino_depth(model_id: str = MODEL_ID, input_hw: tuple[int, int] = OV_INPUT_HW) -> Path:
    """Trace HF Depth Anything V2 Small → OpenVINO IR. Returns the XML path."""
    import openvino as ov
    import torch
    from transformers import AutoModelForDepthEstimation

    xml = OV_CACHE / "depth_anything_v2_small.xml"
    if xml.is_file():
        return xml
    OV_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"[depth] exporting OpenVINO model -> {OV_CACHE} (first run)", file=sys.stderr)
    t0 = time.perf_counter()
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()
    h, w = input_hw
    dummy = torch.randn(1, 3, h, w)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).predicted_depth

    wrap = Wrap(model)
    wrap.eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrap, dummy, strict=False)
    ov_model = ov.convert_model(traced, example_input=dummy)
    ov.save_model(ov_model, str(xml))
    print(f"[depth] OpenVINO export done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return xml


class DepthEstimator:
    """Relative depth map + known-height calibration → metres per detection."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        backend: str = "openvino",
        input_width: int = 518,
        device: str = "cpu",
        ema: float = 0.4,
    ):
        self.model_id = model_id
        self.backend = normalize_backend(backend)
        self.input_width = input_width
        self.device = device
        self.ema = ema
        self._scale: float | None = None  # metres * relative_depth (nearer → larger rel)
        self._proc = None
        self._torch_model = None
        self._ov = None
        self._ov_input = None
        self.last_map: np.ndarray | None = None  # float32 HxW relative depth
        self.last_ms: float = 0.0
        self._init_backend()

    def _init_backend(self) -> None:
        requested = self.backend
        if self.backend == "openvino":
            if not openvino_available():
                print("[depth] openvino not installed; falling back to PyTorch", file=sys.stderr)
                self.backend = "torch"
            else:
                try:
                    import openvino as ov
                    from transformers import AutoImageProcessor

                    xml = ensure_openvino_depth(self.model_id)
                    core = ov.Core()
                    compiled = core.compile_model(str(xml), "CPU")
                    self._ov = compiled
                    self._ov_input = compiled.inputs[0]
                    self._proc = AutoImageProcessor.from_pretrained(self.model_id)
                    print(f"[depth] backend=openvino path={xml.parent}", file=sys.stderr)
                    return
                except Exception as e:
                    print(
                        f"[depth] OpenVINO load/export failed ({type(e).__name__}: {e}); "
                        f"falling back to PyTorch",
                        file=sys.stderr,
                    )
                    self.backend = "torch"

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch

        self._proc = AutoImageProcessor.from_pretrained(self.model_id)
        self._torch_model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self._torch_model.eval()
        if requested != self.backend:
            print(f"[depth] backend=torch (fallback) model={self.model_id}", file=sys.stderr)
        else:
            print(f"[depth] backend=torch model={self.model_id}", file=sys.stderr)

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Return NCHW float32 pixel_values and (orig_h, orig_w)."""
        from PIL import Image

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "openvino":
            th, tw = OV_INPUT_HW
            pil = Image.fromarray(rgb).resize((tw, th))
        else:
            side = self.input_width
            pil = Image.fromarray(rgb).resize((side, max(1, int(h * side / w))))
        inputs = self._proc(images=pil, return_tensors="np" if self.backend == "openvino" else "pt")
        return inputs["pixel_values"], (h, w)

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 relative-depth map at full frame resolution (larger ≈ nearer)."""
        import torch

        t0 = time.perf_counter()
        pv, (oh, ow) = self._preprocess(frame_bgr)
        if self.backend == "openvino":
            # OpenVINO compiled model: fixed HxW
            out = self._ov(np.asarray(pv, dtype=np.float32))
            raw = next(iter(out.values()))
            depth_small = np.squeeze(np.asarray(raw)).astype(np.float32)
            depth = cv2.resize(depth_small, (ow, oh), interpolation=cv2.INTER_LINEAR)
        else:
            with torch.no_grad():
                outputs = self._torch_model(pixel_values=pv if not isinstance(pv, np.ndarray) else torch.from_numpy(pv))
            post = self._proc.post_process_depth_estimation(
                outputs, target_sizes=[(oh, ow)]
            )
            depth = post[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)
        self.last_map = depth
        self.last_ms = (time.perf_counter() - t0) * 1000
        return depth

    @staticmethod
    def sample_rel(depth: np.ndarray, box: Box) -> float | None:
        """Median relative depth near the bottom-centre of the box (ground contact)."""
        h, w = depth.shape[:2]
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
        # lower 20% of the box
        y_lo = y1 + int(0.8 * (y2 - y1))
        patch = depth[y_lo:y2, x1:x2]
        if patch.size == 0:
            return None
        return float(np.median(patch))

    def reset(self) -> None:
        """A changed camera cannot inherit the previous frame's scale calibration."""
        self._scale = None
        self.last_map = None

    def calibrate(self, depth: np.ndarray, detections: list[Detection], frame_w: int, frame_h: int, hfov_deg: float) -> float | None:
        """Update EMA scale from person/vehicle boxes that also have a box-height distance."""
        ratios = []
        for d in detections:
            if d.label not in CALIB_LABELS:
                continue
            box_d = estimate_distance_m(d.box, d.label, frame_w, frame_h, hfov_deg)
            rel = self.sample_rel(depth, d.box)
            if box_d is None or rel is None or rel < 1e-3 or not (0.5 < box_d < 40):
                continue
            # larger rel → nearer → metres ≈ scale / rel
            ratios.append(box_d * rel)
        if not ratios:
            return self._scale
        med = float(np.median(ratios))
        if self._scale is None:
            self._scale = med
        else:
            self._scale = (1 - self.ema) * self._scale + self.ema * med
        return self._scale

    def distance_m(self, box: Box, depth: np.ndarray | None = None) -> float | None:
        depth = depth if depth is not None else self.last_map
        if depth is None or self._scale is None:
            return None
        rel = self.sample_rel(depth, box)
        if rel is None or rel < 1e-3:
            return None
        return float(self._scale / rel)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """BGR heatmap for overlay (near = warm)."""
    d = depth.astype(np.float32)
    d = (d - d.min()) / max(1e-6, float(d.max() - d.min()))
    u8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
