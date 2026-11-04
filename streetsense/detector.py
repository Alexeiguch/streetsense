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
from dataclasses import replace
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
        if self.cfg.track and self.cfg.track_conf is not None:
            # ByteTrack needs low-score boxes to recover existing tracks. They are
            # association evidence only: _accept still applies the public class thresholds.
            floor = min(floor, self.cfg.track_conf)
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

    def reset_tracking(self) -> None:
        """Forget pixel-coordinate tracks after a camera geometry change or video rewind."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()

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


    def tiled_lights(self, frame_bgr: np.ndarray, grid: tuple[int, int], conf: float) -> list[Detection]:
        """Traffic lights found on overlapping tiles (rows x cols, 30% overlap), mapped to frame pixels.

        Uses the plain-predict handle (no tracker state). Untracked detections tagged {"tile": (r, c)}.
        """
        rows, cols = grid
        if rows < 1 or cols < 1 or rows * cols < 2:
            return []
        if getattr(self, "_zoom_model", None) is None:
            from ultralytics import YOLO

            self._zoom_model = YOLO(self.model_path)
        H, W = frame_bgr.shape[:2]
        return tile_detections(frame_bgr, grid, conf, lambda im: [
            (b, self.names[int(c)], float(p)) for b, c, p in _boxes(self._zoom_model.predict(
                im, imgsz=self.cfg.imgsz, conf=conf, iou=self.cfg.iou, device=self.cfg.device, verbose=False)[0])])

    def focused_lights(self, frame_bgr: np.ndarray, lights: list[Detection], conf: float = .5) -> list[Detection]:
        """One candidate-refinement tile. Parent boxes remain in the reader input."""
        if getattr(self, "_zoom_model", None) is None:
            from ultralytics import YOLO

            self._zoom_model = YOLO(self.model_path)
        return focused_detections(frame_bgr, lights, conf, lambda image: [
            (b, self.names[int(c)], float(p)) for b, c, p in _boxes(self._zoom_model.predict(
                image, imgsz=self.cfg.imgsz, conf=conf, iou=self.cfg.iou,
                device=self.cfg.device, verbose=False)[0])])


def _boxes(res):
    if res.boxes is None:
        return []
    return list(zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy()))


def tile_windows(W: int, H: int, grid: tuple[int, int], overlap: float = 1.3) -> list[tuple[int, int, int, int]]:
    """(x0, y0, w, h) for a rows x cols grid of tiles each `overlap` times the plain cell size."""
    rows, cols = grid
    tw, th = min(W, int(W / cols * overlap)), min(H, int(H / rows * overlap))
    xs = np.linspace(0, W - tw, cols).astype(int) if cols > 1 else [0]
    ys = np.linspace(0, H - th, rows).astype(int) if rows > 1 else [0]
    return [(int(x0), int(y0), tw, th) for y0 in ys for x0 in xs]


def tile_detections(frame: np.ndarray, grid: tuple[int, int], conf: float, predict) -> list[Detection]:
    """Run `predict(tile) -> [(xyxy, label, conf)]` on each tile; keep traffic lights >= conf in frame pixels."""
    H, W = frame.shape[:2]
    out: list[Detection] = []
    for i, (x0, y0, tw, th) in enumerate(tile_windows(W, H, grid)):
        for b, label, p in predict(frame[y0:y0 + th, x0:x0 + tw]):
            if label != "traffic light" or p < conf:
                continue
            box = Box(max(0.0, float(b[0]) + x0), max(0.0, float(b[1]) + y0),
                      min(float(W), float(b[2]) + x0), min(float(H), float(b[3]) + y0))
            if box.w >= 4 and box.h >= 4:
                out.append(Detection("traffic light", box, float(p), None, {"tile": i}))
    return out


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)); iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    u = a.w * a.h + b.w * b.h - inter
    return inter / u if u > 0 else 0.0


def focused_tile_window(frame: np.ndarray, lights: list[Detection]):
    """Choose one ordinary tile by lit pixels inside a detected head, without a colour vote."""
    import cv2

    lights = [d for d in lights if d.label == "traffic light"]
    if not lights:
        return None
    height, width = frame.shape[:2]
    parent = max(lights, key=lambda d: d.conf).box.clip(width, height)
    x1, y1, x2, y2 = map(int, (parent.x1, parent.y1, parent.x2, parent.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    lit = (hsv[..., 1] >= 70) & (hsv[..., 2] >= 120)
    windows = tile_windows(width, height, (2, 3))
    scores = [int(lit[max(0, y-y1):max(0, min(y+th, y2)-y1),
                      max(0, x-x1):max(0, min(x+tw, x2)-x1)].sum()) for x, y, tw, th in windows]
    return (windows[int(np.argmax(scores))], parent) if max(scores) > 0 else None


def focused_detections(frame: np.ndarray, lights: list[Detection], conf: float, predict) -> list[Detection]:
    """Refine one found head; extra boxes cannot replace its stop/countdown evidence."""
    chosen = focused_tile_window(frame, lights)
    if chosen is None:
        return []
    (x, y, tw, th), parent = chosen
    height, width = frame.shape[:2]
    extra = []
    for coords, label, score in predict(frame[y:y+th, x:x+tw]):
        if label != "traffic light" or score < conf:
            continue
        box = Box(float(coords[0])+x, float(coords[1])+y,
                  float(coords[2])+x, float(coords[3])+y).clip(width, height)
        if box.w < 4 or box.h < 4 or not (parent.x1 <= box.cx <= parent.x2 and parent.y1 <= box.cy <= parent.y2):
            continue
        if any(_iou(box, d.box) >= .5 for d in lights):
            continue
        extra.append(Detection("traffic light", box, float(score), None, {
            "source": "signal_refine", "refine_parent": [parent.x1, parent.y1, parent.x2, parent.y2]}))
    return nms_lights(extra)


class SignalHeadRefiner:
    """Bound refinement frequency and reuse coordinates only while the same parent is present."""

    def __init__(self, interval_s: float = .25, min_width_fraction: float = .33):
        self.interval_s = max(.01, interval_s)
        self.min_width_fraction = min(1., max(0., min_width_fraction))
        self.reset()

    def reset(self) -> None:
        self.last_attempt = -float("inf")
        self.capture_ts = -float("inf")
        self.parent = None
        self.boxes = []
        self.error = None
        self.next_id = -1  # separate from positive detector ids and untracked position keys

    def update(self, frame, t: float, lights, detector) -> list[Detection]:
        from .traffic_light import signal_kind

        eligible = [d for d in lights if d.box.w >= self.min_width_fraction * frame.shape[1]
                    and signal_kind(d.box.w, d.box.h) != "vehicle"]
        chosen = focused_tile_window(frame, eligible)
        if chosen is None:
            self.parent, self.boxes = None, []
            return []
        _, parent = chosen
        stable = self.parent is not None and _iou(parent, self.parent) >= .8
        if t < self.last_attempt:
            self.reset()
            stable = False
        # Movement discards coordinates immediately, but cannot bypass the call-rate cap.
        if not stable:
            self.boxes = []
        if t - self.last_attempt >= self.interval_s:
            self.last_attempt = t
            try:
                previous = list(self.boxes)
                refined = detector.focused_lights(frame, eligible, .5)
                boxes = []
                for detection in refined:
                    matches = [i for i, old in enumerate(previous) if _iou(old.box, detection.box) >= .4]
                    if matches:
                        i = max(matches, key=lambda i: _iou(previous[i].box, detection.box))
                        track_id = previous.pop(i).track_id
                    else:
                        track_id = self.next_id
                        self.next_id -= 1
                    boxes.append(replace(detection, track_id=track_id))
                self.boxes = boxes
                self.error = None
            except Exception as exc:
                self.boxes = []
                self.error = type(exc).__name__
            self.parent, self.capture_ts = parent, t
        if not 0 <= t - self.capture_ts <= self.interval_s:
            return []
        return [replace(d, attrs={**d.attrs, "refine_age_s": t - self.capture_ts}) for d in self.boxes
                if parent.x1 <= d.box.cx <= parent.x2 and parent.y1 <= d.box.cy <= parent.y2
                and all(_iou(d.box, local.box) < .5 for local in lights)]


def nms_lights(dets: list[Detection], iou: float = 0.5) -> list[Detection]:
    """Keep the best box/score and retain an overlapping local track's identity."""
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        overlaps = [i for i, k in enumerate(keep) if _iou(d.box, k.box) >= iou]
        if not overlaps:
            keep.append(d)
        elif d.track_id is not None:
            # Transfer only to the closest retained duplicate, never to another signal.
            i = max(overlaps, key=lambda i: _iou(d.box, keep[i].box))
            if keep[i].track_id is None:
                keep[i] = replace(keep[i], track_id=d.track_id, attrs=dict(keep[i].attrs))
    return keep


def light_tiles(cfg: DetectorConfig | None = None) -> tuple[tuple[int, int], str, int]:
    """Opt-in tiled light pass from env (default off).

    STREETSENSE_LIGHT_TILES="2x3" (rows x cols; unset/"0" = off).
    STREETSENSE_LIGHT_TILES_MODE="always" | "fallback" (default fallback: only when no light
    >= light_zoom_conf at full size, plus every Nth frame if STREETSENSE_LIGHT_TILES_EVERY=N > 0).
    """
    import os

    env = (os.environ.get("STREETSENSE_LIGHT_TILES") or "").strip().lower()
    mode = (os.environ.get("STREETSENSE_LIGHT_TILES_MODE") or "fallback").strip().lower()
    if mode not in ("always", "fallback"):
        mode = "fallback"
    try:
        every = max(0, int(os.environ.get("STREETSENSE_LIGHT_TILES_EVERY") or 0))
    except ValueError:
        every = 0
    try:
        r, c = (int(v) for v in env.split("x"))
    except ValueError:
        return (0, 0), mode, every
    if r < 1 or c < 1 or r * c < 2 or r * c > 16:
        return (0, 0), mode, every
    return (r, c), mode, every


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
        self._labels: dict[int, str] = {}

    def reset(self) -> None:
        self._hist.clear()
        self._labels.clear()

    def update(self, t: float, detections: list[Detection]) -> set[int]:
        approaching: set[int] = set()
        for tid, q in list(self._hist.items()):
            if t < q[-1][0] or t - q[-1][0] > self.window_s:
                del self._hist[tid]
                self._labels.pop(tid, None)
        seen = set()
        for d in detections:
            if d.track_id is None:
                continue
            seen.add(d.track_id)
            q = self._hist.setdefault(d.track_id, deque(maxlen=64))
            if self._labels.get(d.track_id, d.label) != d.label:
                q.clear()
            self._labels[d.track_id] = d.label
            # A fixed 64-frame buffer loses the entire time window above 128 FPS.
            # Sample in time instead, while comparing the current box on every call.
            if not q or t - q[-1][0] >= self.window_s / 32:
                q.append((t, d.box.h))
            while len(q) > 2 and t - q[1][0] >= self.window_s:
                q.popleft()
            t0, h0 = q[0]
            if t - t0 >= self.window_s * 0.5 and h0 > 0 and d.box.h / h0 >= self.ratio:
                approaching.add(d.track_id)
        if len(self._hist) > self.max_tracks:
            for k in [k for k in self._hist if k not in seen][: len(self._hist) - self.max_tracks]:
                del self._hist[k]
                self._labels.pop(k, None)
        return approaching
