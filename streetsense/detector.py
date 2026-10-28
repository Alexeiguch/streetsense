"""Fast local detector: Ultralytics YOLO (COCO) with optional ByteTrack ids.

Default weights are YOLO11s (better recall than 11n on our hand sample; see
docs/BENCHMARK.md). Default inference backend is OpenVINO (CPU). PyTorch remains
selectable via DetectorConfig.backend / CLI --backend. Missing OpenVINO exports
are built from the .pt on first use; failures fall back to PyTorch with a clear
log line. Per-class confidence overrides (e.g. lower for person) are applied
after a global floor so recall can be raised without flooding vehicle FPs.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from .config import DetectorConfig
from .types import Box, Detection

# COCO classes that matter on a sidewalk. Everything else is ignored.
KEEP = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train", "traffic light",
    "stop sign", "fire hydrant", "parking meter", "bench", "dog", "chair", "potted plant", "suitcase",
}

# Sidewalk hazards from the optional open-vocab loop (not COCO).
HAZARD_LABELS = {
    "bollard", "traffic cone", "scooter", "construction barrier", "barrier",
    "stairs", "step", "pothole", "manhole", "kerb",
}

BACKENDS = ("openvino", "torch")


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
    raise ValueError(f"unknown detector backend {name!r}; expected one of {BACKENDS}")


def openvino_dir_for(weights: str | Path) -> Path:
    """models/yolo11n.pt -> models/yolo11n_openvino_model (Ultralytics export layout)."""
    p = Path(weights)
    return p.with_name(p.stem + "_openvino_model")


def ensure_pt_weights(weights: str | Path) -> Path:
    """Return a local .pt path, downloading the Ultralytics asset if needed."""
    from ultralytics import YOLO

    pt = Path(weights)
    if pt.exists():
        return pt
    pt.parent.mkdir(parents=True, exist_ok=True)
    name = pt.name  # e.g. yolo11n.pt
    print(f"[detector] downloading {name} -> {pt}", file=sys.stderr)
    # YOLO(name) downloads into CWD; load then save/copy into the requested path.
    model = YOLO(name)
    src = Path(name)
    if src.exists() and src.resolve() != pt.resolve():
        src.replace(pt)
    elif not pt.exists():
        # Some ultralytics versions keep weights under the package cache; re-save.
        try:
            model.save(str(pt))
        except Exception:
            # Last resort: point at whatever YOLO resolved.
            resolved = Path(getattr(model, "ckpt_path", None) or name)
            if resolved.exists():
                import shutil

                shutil.copy2(resolved, pt)
    if not pt.exists():
        raise FileNotFoundError(f"could not obtain weights at {pt}")
    return pt


def ensure_openvino_model(weights: str | Path, imgsz: int = 640) -> Path:
    """Export OpenVINO IR next to the .pt if the export dir is missing. Returns the dir."""
    from ultralytics import YOLO

    pt = ensure_pt_weights(weights)
    ov_dir = openvino_dir_for(pt)
    if ov_dir.is_dir() and any(ov_dir.iterdir()):
        return ov_dir
    print(f"[detector] exporting OpenVINO model -> {ov_dir} (first run)", file=sys.stderr)
    t0 = time.perf_counter()
    model = YOLO(str(pt))
    exported = model.export(format="openvino", imgsz=imgsz, device="cpu")
    elapsed = time.perf_counter() - t0
    out = Path(exported)
    if out != ov_dir and out.exists():
        # Ultralytics should write beside the .pt; if CWD differed, move/rename.
        if ov_dir.exists():
            import shutil

            shutil.rmtree(ov_dir)
        out.rename(ov_dir)
    if not ov_dir.is_dir():
        raise RuntimeError(f"OpenVINO export did not produce {ov_dir} (got {exported!r})")
    print(f"[detector] OpenVINO export done in {elapsed:.1f}s", file=sys.stderr)
    return ov_dir


class YoloDetector:
    def __init__(self, cfg: DetectorConfig):
        from ultralytics import YOLO

        self.cfg = cfg
        self.backend = normalize_backend(getattr(cfg, "backend", "openvino"))
        self.model_path: str
        requested = self.backend

        if self.backend == "openvino":
            if not openvino_available():
                print(
                    "[detector] openvino not installed; falling back to PyTorch",
                    file=sys.stderr,
                )
                self.backend = "torch"
            else:
                try:
                    ov_dir = ensure_openvino_model(cfg.weights, imgsz=cfg.imgsz)
                    self.model = YOLO(str(ov_dir))
                    self.model_path = str(ov_dir)
                    self.names = self.model.names
                    print(f"[detector] backend=openvino path={self.model_path}", file=sys.stderr)
                    return
                except Exception as e:
                    print(
                        f"[detector] OpenVINO load/export failed ({type(e).__name__}: {e}); "
                        f"falling back to PyTorch",
                        file=sys.stderr,
                    )
                    self.backend = "torch"

        pt = ensure_pt_weights(cfg.weights)
        self.model = YOLO(str(pt))
        self.model_path = str(pt)
        self.names = self.model.names
        if requested != self.backend:
            print(f"[detector] backend=torch (fallback) path={self.model_path}", file=sys.stderr)
        else:
            print(f"[detector] backend=torch path={self.model_path}", file=sys.stderr)

    def _conf_floor(self) -> float:
        """Lowest confidence we ask the backend for (global or any per-class override)."""
        by = getattr(self.cfg, "conf_by_class", None) or {}
        vals = [float(self.cfg.conf), *[float(v) for v in by.values()]]
        return min(vals) if vals else float(self.cfg.conf)

    def _accept(self, label: str, conf: float) -> bool:
        by = getattr(self.cfg, "conf_by_class", None) or {}
        thr = float(by.get(label, self.cfg.conf))
        return conf >= thr

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        floor = self._conf_floor()
        kw = dict(imgsz=self.cfg.imgsz, conf=floor, iou=self.cfg.iou, device=self.cfg.device, verbose=False)
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
            if not self._accept(label, float(p)):
                continue
            out.append(Detection(label, Box(*map(float, b)), float(p), None if tid is None else int(tid)))
        return out

    def zoomed_lights(self, frame_bgr: np.ndarray, scales: tuple[float, ...], conf: float) -> list[Detection]:
        """Traffic lights found on zoomed-out copies of the frame, mapped back to frame pixels.

        Runs on a second model handle with plain predict, so the ByteTrack state of the main
        model never sees the rescaled frames. Stops at the first scale that finds a light.
        Detections are untracked (track_id None); the light smoother keys them by IoU.
        """
        if not scales:
            return []
        if getattr(self, "_zoom_model", None) is None:
            from ultralytics import YOLO

            self._zoom_model = YOLO(self.model_path)
        H, W = frame_bgr.shape[:2]
        for s in scales:
            canvas, ox, oy = zoom_out_canvas(frame_bgr, s)
            res = self._zoom_model.predict(canvas, imgsz=self.cfg.imgsz, conf=conf, iou=self.cfg.iou,
                                           device=self.cfg.device, verbose=False)[0]
            if res.boxes is None:
                continue
            out = []
            for b, c, p in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int),
                               res.boxes.conf.cpu().numpy()):
                if self.names[int(c)] != "traffic light" or float(p) < conf:
                    continue
                x1, y1, x2, y2 = ((float(b[0]) - ox) / s, (float(b[1]) - oy) / s,
                                  (float(b[2]) - ox) / s, (float(b[3]) - oy) / s)
                box = Box(max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2))
                if box.w >= 4 and box.h >= 4:
                    out.append(Detection("traffic light", box, float(p), None, {"zoom": s}))
            if out:
                return out
        return []


def light_zoom_scales(cfg: DetectorConfig) -> tuple[float, ...]:
    """Scales for the zoom-out light pass; env STREETSENSE_LIGHT_ZOOM overrides ("0" = off)."""
    import os

    env = os.environ.get("STREETSENSE_LIGHT_ZOOM")
    if env is None:
        return tuple(getattr(cfg, "light_zoom_scales", ()) or ())
    env = env.strip().lower()
    if env in ("", "0", "off", "false", "no"):
        return ()
    try:
        return tuple(float(v) for v in env.split(",") if 0.05 < float(v) < 1.0)
    except ValueError:
        return ()


def zoom_out_canvas(frame: np.ndarray, scale: float) -> tuple[np.ndarray, int, int]:
    """Frame shrunk by `scale` and centred on a grey (114) canvas of the original size."""
    import cv2

    H, W = frame.shape[:2]
    h, w = max(1, int(H * scale)), max(1, int(W * scale))
    oy, ox = (H - h) // 2, (W - w) // 2
    canvas = np.full_like(frame, 114)
    canvas[oy:oy + h, ox:ox + w] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return canvas, ox, oy


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
