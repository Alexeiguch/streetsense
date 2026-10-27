"""The fast loop: frame -> perception -> candidate alerts -> alert manager -> overlay."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .alerts import AlertManager
from .config import Config
from .crosswalk import CrosswalkHysteresis, CrosswalkResult, detect_crosswalk
from .detector import ApproachTracker, YoloDetector
from .open_vocab import OpenVocabHazardDetector
from .meta_api import MetaVisionAssistant
from .overlay import draw
from .rules import FrameFacts, build_alerts
from .surface import SIDEWALK, ROAD, SurfaceResult
from .traffic_light import LightStateSmoother, classify_light, signal_kind
from .types import Alert, Detection


@dataclass
class FrameOutput:
    index: int
    t: float
    detections: list[Detection]
    crosswalk: CrosswalkResult | None
    surface: SurfaceResult | None
    candidates: list[Alert]
    spoken: Alert | None
    timings_ms: dict[str, float] = field(default_factory=dict)
    annotated: np.ndarray | None = None


class Pipeline:
    def __init__(self, cfg: Config | None = None, use_surface: bool | None = None, clock=time.monotonic):
        self.cfg = cfg or Config()
        self.clock = clock
        self.detector = YoloDetector(self.cfg.detector)
        self.surface_seg = None
        if self.cfg.surface.enabled if use_surface is None else use_surface:
            try:
                from .surface import SurfaceSegmenter

                self.surface_seg = SurfaceSegmenter(
                    self.cfg.surface.model_id,
                    self.cfg.surface.input_width,
                    backend=getattr(self.cfg.surface, "backend", "openvino"),
                )
            except Exception as e:  # missing transformers / offline: run without it
                print(f"[surface] disabled: {type(e).__name__}: {e}")
        self.alerts = AlertManager(self.cfg.alerts, clock=clock)
        self.approach = ApproachTracker()
        self.smoother = LightStateSmoother()
        self.cw_hyst = CrosswalkHysteresis()
        self._kinds: dict[int, str] = {}
        self.vlm = MetaVisionAssistant(self.cfg.vlm)
        self._i = 0
        self._last_surface: SurfaceResult | None = None
        self._last_cw: CrosswalkResult | None = None
        self._last_hazards: list[Detection] = []
        self.open_vocab = None
        if getattr(self.cfg, "open_vocab", None) is not None and self.cfg.open_vocab.enabled:
            try:
                self.open_vocab = OpenVocabHazardDetector(
                    weights=self.cfg.open_vocab.weights,
                    conf=self.cfg.open_vocab.conf,
                    imgsz=self.cfg.open_vocab.imgsz,
                    device=self.cfg.detector.device,
                )
            except Exception as e:
                print(f"[open-vocab] disabled: {type(e).__name__}: {e}")
                self.open_vocab = None
        self.recent: list[Alert] = []
        self.timing_totals: dict[str, float] = defaultdict(float)

    def _track_kind(self, track_id: int | None, kind: str) -> str:
        """Sticky per-track kind: once a track shows evidence (walk figure / shape), keep it."""
        if track_id is None:
            return kind
        if kind != "unknown":
            self._kinds[track_id] = kind
        return self._kinds.get(track_id, "unknown")

    def process(self, frame: np.ndarray, annotate: bool = True, t: float | None = None) -> FrameOutput:
        t = self.clock() if t is None else t
        tm: dict[str, float] = {}
        H, W = frame.shape[:2]

        t0 = time.perf_counter()
        dets = self.detector(frame)
        tm["detect"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for d in dets:
            if d.label == "traffic light":
                b = d.box.clip(W, H)
                crop = frame[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
                kind = signal_kind(b.w, b.h)
                r = classify_light(crop, kind)
                if r.state == "walk" or getattr(r, "figure", False):
                    kind = "pedestrian"
                kind = self._track_kind(d.track_id, kind)
                d.attrs["signal_kind"] = kind
                d.attrs["countdown"] = r.countdown
                d.attrs["light_raw"] = r.state
                d.attrs["figure"] = bool(getattr(r, "figure", False))
                d.attrs["light_state"] = self.smoother.update(d.track_id, r.state)
        tm["lights"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if self.open_vocab is not None:
            every = max(1, self.cfg.open_vocab.every_n_frames)
            if self._i % every == 0 or not self._last_hazards:
                try:
                    self._last_hazards = self.open_vocab(frame)
                except Exception as e:
                    print(f"[open-vocab] frame failed: {type(e).__name__}: {e}")
                    self._last_hazards = []
            if self._last_hazards:
                dets = list(dets) + list(self._last_hazards)
        tm["open_vocab"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if self.surface_seg is not None and (self._i % self.cfg.surface.every_n_frames == 0 or self._last_surface is None):
            self._last_surface = self.surface_seg(frame)
        tm["surface"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if self._i % self.cfg.crosswalk_every_n_frames == 0 or self._last_cw is None:
            ground = np.isin(self._last_surface.labels, (ROAD, SIDEWALK)) if self._last_surface is not None else None
            self._last_cw = self.cw_hyst.update(detect_crosswalk(frame, ground_mask=ground))
        tm["crosswalk"] = (time.perf_counter() - t0) * 1000

        approaching = self.approach.update(t, dets)
        facts = FrameFacts(W, H, dets, self._last_cw, self._last_surface, approaching)
        cands = build_alerts(facts, self.cfg.camera.hfov_deg)
        for a in cands:
            self.alerts.submit(a)
        spoken = self.alerts.pop_ready()
        if spoken:
            self.recent.insert(0, spoken)
            del self.recent[5:]

        # Slow loop: periodic Muse Spark scene check (background thread; no-op without a key).
        self.vlm.maybe_request(frame, t)

        ann = None
        if annotate:
            t0 = time.perf_counter()
            vlm_text = self.vlm.latest.data.get("summary") if self.vlm.latest else None
            fps_txt = f"detect {tm['detect']:.0f}ms surface {tm['surface']:.0f}ms"
            fresh = [a for a in self.recent if t - a.ts < 4.0]
            ann = draw(frame, dets, self._last_cw, self._last_surface, fresh, fps_txt, vlm_text)
            tm["overlay"] = (time.perf_counter() - t0) * 1000
        for k, v in tm.items():
            self.timing_totals[k] += v
        out = FrameOutput(self._i, t, dets, self._last_cw, self._last_surface, cands, spoken, tm, ann)
        self._i += 1
        return out
