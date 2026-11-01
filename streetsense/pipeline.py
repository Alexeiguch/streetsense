"""The fast loop: frame -> perception -> candidate alerts -> alert manager -> overlay."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field, replace

import numpy as np

from .alerts import AlertManager
from .config import Config
from .crosswalk import CrosswalkHysteresis, CrosswalkResult, detect_crosswalk
from .detector import ApproachTracker, YoloDetector, light_focus_enabled, light_tiles, light_zoom_scales, nms_lights
from .geometry import estimate_distance_m
from .meta_api import MetaSamSegmenter, MetaVisionAssistant, sam_boxes_to_crosswalk
from .open_vocab import OpenVocabHazardDetector
from .observations import ObservationLog, summarise_frame
from .open_vocab_sched import OpenVocabScheduler
from .overlay import draw
from .frame_history import FrameHistory
from . import signal_cnn, signal_distance
from .rules import FrameFacts, build_alerts, select_crossing_signal
from .surface import SIDEWALK, ROAD, SurfaceResult
from .traffic_light import LightStateSmoother, apply_signal_cnn, cnn_hold_veto, cnn_rejects_walk, classify_light, figure_walk_guard, signal_kind
from .types import Alert, Detection, Priority


def build_detector(cfg: Config):
    """YOLO (Ultralytics) by default; RF-DETR when --weights rfdetr-nano|rfdetr-small."""
    w = (cfg.detector.weights or "").strip().lower()
    if "rfdetr" in w:
        from .rfdetr_detector import RfDetrDetector

        variant = "small" if "small" in w else "nano"
        return RfDetrDetector(cfg.detector, variant=variant)
    return YoloDetector(cfg.detector)


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
        self.detector = build_detector(self.cfg)
        self.surface_seg = None
        if self.cfg.surface.enabled if use_surface is None else use_surface:
            try:
                from .surface import SurfaceSegmenter

                self.surface_seg = SurfaceSegmenter(
                    self.cfg.surface.model_id,
                    self.cfg.surface.input_width,
                    backend=getattr(self.cfg.surface, "backend", "openvino"),
                )
            except Exception as e:
                print(f"[surface] disabled: {type(e).__name__}: {e}")
        self.depth = None
        self._last_depth = None
        if getattr(self.cfg, "depth", None) is not None and self.cfg.depth.enabled:
            try:
                from .depth import DepthEstimator

                self.depth = DepthEstimator(
                    model_id=self.cfg.depth.model_id,
                    backend=getattr(self.cfg.depth, "backend", "openvino"),
                    input_width=self.cfg.depth.input_width,
                    device=self.cfg.detector.device,
                )
            except Exception as e:
                print(f"[depth] disabled: {type(e).__name__}: {e}")
                self.depth = None
        self.alerts = AlertManager(self.cfg.alerts, clock=clock)
        self.approach = ApproachTracker()
        self.smoother = LightStateSmoother(clock=self.clock)
        # Refinements never share votes with parent boxes; parent/CNN vetoes discard their WALK votes.
        # Match NMS: two surviving refinements cannot share a key in the same frame.
        self.focus_smoother = LightStateSmoother(clock=self.clock, reset_walk_on_veto=True, iou_match=.5)
        self.signal_dist_cfg = signal_distance.SignalDistanceConfig.from_env()
        self.cw_hyst = CrosswalkHysteresis()
        self._kinds: dict[int, str] = {}
        self.vlm = MetaVisionAssistant(self.cfg.vlm)
        # Recent frames for multi-frame Muse crossing checks (only kept when enabled).
        self.frame_history: FrameHistory | None = None
        if int(getattr(self.cfg.vlm, "history_frames", 0) or 0) > 0:
            span = self.cfg.vlm.history_frames * self.cfg.vlm.history_spacing_s
            self.frame_history = FrameHistory(
                max_age_s=span + 1.0,
                min_gap_s=min(0.2, self.cfg.vlm.history_spacing_s / 2),
                max_side=self.cfg.vlm.crossing_max_image_side,
            )
        self.sam = None
        self._sam_last = -1e9
        self._sam_segments = []
        if getattr(self.cfg, "sam", None) is not None and self.cfg.sam.enabled:
            try:
                self.sam = MetaSamSegmenter(
                    model=self.cfg.sam.model,
                    max_image_side=self.cfg.sam.max_image_side,
                    timeout_s=self.cfg.sam.timeout_s,
                )
                if not self.sam.enabled:
                    self.sam = None
            except Exception as e:
                print(f"[sam] disabled: {type(e).__name__}: {e}")
                self.sam = None
        # Cost-capped SAM 3.1 signal fallback (docs/SAM_FALLBACK.md). Candidate boxes only; never decides walk.
        self.sam_fallback = None
        self._sam_prev_state = "unknown"
        try:
            from .sam_fallback import SamControlConfig, SamFallbackController

            scfg = SamControlConfig.from_env()
            if scfg.enabled:
                self.sam_fallback = SamFallbackController(scfg)
        except Exception as e:
            print(f"[sam-fallback] disabled: {type(e).__name__}: {e}")
            self.sam_fallback = None
        self._i = 0
        self._last_surface: SurfaceResult | None = None
        self._last_cw: CrosswalkResult | None = None
        self._last_hazards: list[Detection] = []
        self._ov_sched = OpenVocabScheduler(
            every_n_frames=getattr(getattr(self.cfg, 'open_vocab', None), 'every_n_frames', 15) or 15,
        )
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
        # Rolling compact facts for the opt-in Muse Wonder guide (no model calls).
        self.observations = ObservationLog()
        self.timing_totals: dict[str, float] = defaultdict(float)

    # -- SAM fallback hooks (context inputs + accounting) ----------------------------------------------------
    def note_crossing_check(self, t: float | None = None) -> None:
        """User asked 'can I cross?': opens a crossing context for the SAM fallback."""
        if self.sam_fallback is not None:
            self.sam_fallback.note_crossing_check(self.clock() if t is None else t)

    def note_crossing_distance(self, meters: float | None) -> None:
        """Maps distance to the next crossing (None = unknown)."""
        if self.sam_fallback is not None:
            self.sam_fallback.note_crossing_distance(meters)

    def sam_fallback_summary(self) -> dict:
        return self.sam_fallback.summary() if self.sam_fallback is not None else {"enabled": False, "calls": 0, "usd": 0.0}

    def _inject_sam_lights(self, frame, t, dets, W, H):
        """Add SAM signal-head boxes as extra traffic-light candidates (only where no local box overlaps).
        They go through the same reader/CNN/smoother/selection; walk still needs Muse yes in answers.py."""
        from .detector import _iou
        from .types import Box

        lights = [d for d in dets if d.label == "traffic light"]
        local_conf = max((d.conf for d in lights), default=0.0)
        crosswalk_ahead = bool(self._last_cw is not None and getattr(self._last_cw, "found", False))
        try:
            boxes = self.sam_fallback.step(frame, t, crosswalk_ahead=crosswalk_ahead, local_conf=local_conf,
                                           local_state=self._sam_prev_state)
        except Exception as e:  # the fallback must never break the fast loop
            print(f"[sam-fallback] step failed: {type(e).__name__}: {e}")
            return dets
        extra = []
        for x1, y1, x2, y2, c in boxes:
            b = Box(x1, y1, x2, y2).clip(W, H)
            if all(_iou(b, d.box) < 0.3 for d in lights + extra):
                extra.append(Detection("traffic light", b, float(c), None, {"source": "sam_fallback"}))
        return list(dets) + extra

    def _track_kind(self, track_id: int | None, kind: str) -> str:
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

        # Close-up heads that fill the frame are often missed (or scored < 0.35) at full size.
        # Only when no light is confident, look again on zoomed-out copies (traffic lights only).
        dcfg = self.cfg.detector
        zoom = light_zoom_scales(dcfg) if hasattr(self.detector, "zoomed_lights") else ()
        zconf = float(getattr(dcfg, "light_zoom_conf", 0.35))
        confident = any(d.label == "traffic light" and d.conf >= zconf for d in dets)
        # Opt-in tiles (STREETSENSE_LIGHT_TILES=2x3): extra light boxes only, merged by NMS and read by
        # the same reader/CNN/smoother below. They add candidates; they never decide walk themselves.
        grid, tmode, tevery = light_tiles(dcfg) if hasattr(self.detector, "tiled_lights") else ((0, 0), "", 0)
        self._tile_n = getattr(self, "_tile_n", 0) + 1
        if grid[0] and (tmode == "always" or not confident or (tevery and self._tile_n % tevery == 0)):
            t0 = time.perf_counter()
            tconf = float(getattr(dcfg, "light_tile_conf", 0.5))
            extra = self.detector.tiled_lights(frame, grid, tconf)
            lights = [d for d in dets if d.label == "traffic light"]
            dets = [d for d in dets if d.label != "traffic light"] + nms_lights(lights + extra)
            tm["detect_tiles"] = (time.perf_counter() - t0) * 1000
            zoom = ()  # tiles replace the zoom-out pass on this frame (as benchmarked)
        if zoom and not confident:
            t0 = time.perf_counter()
            extra = self.detector.zoomed_lights(frame, zoom, zconf)
            if extra:
                dets = [d for d in dets if d.label != "traffic light"] + extra
            tm["detect_zoom"] = (time.perf_counter() - t0) * 1000

        if (light_focus_enabled(dcfg) and "detect_tiles" not in tm
                and hasattr(self.detector, "focused_lights")):
            t0 = time.perf_counter()
            lights = [d for d in dets if d.label == "traffic light"]
            # Original boxes stay first: their current red/countdown/CNN veto is read before children.
            dets = list(dets) + self.detector.focused_lights(frame, lights, .5)
            tm["detect_focus"] = (time.perf_counter() - t0) * 1000

        if self.sam_fallback is not None:
            t0 = time.perf_counter()
            dets = self._inject_sam_lights(frame, t, dets, W, H)
            tm["sam_fallback"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        parent_heads = {}
        for head in dets:
            if head.label == "traffic light" and "focus_parent_box" not in head.attrs:
                box = head.box.clip(W, H)
                parent_heads[(box.x1, box.y1, box.x2, box.y2)] = head
        for d in dets:
            if d.label == "traffic light":
                b = d.box.clip(W, H)
                crop = frame[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
                kind = signal_kind(b.w, b.h)
                r = classify_light(crop, kind)
                d.attrs["light_heuristic"] = r.state
                cnn = signal_cnn.get()
                pred = None
                if cnn is not None and crop.size:
                    pred = cnn(crop)
                    d.attrs["signal_cnn"] = pred.label
                    d.attrs["signal_cnn_p_walk"] = round(pred.p_walk, 3)
                    d.attrs["signal_cnn_probs"] = dict(zip(signal_cnn.CLASSES, pred.probs))
                    d.attrs["signal_cnn_t_walk"] = cnn.t_walk
                    d.attrs["signal_cnn_t_veto"] = cnn.t_veto
                    r = apply_signal_cnn(r, pred.probs, kind, cnn.t_walk, cnn.t_veto)
                r = figure_walk_guard(r, frame, b)
                parent_box = d.attrs.get("focus_parent_box")
                if parent_box is not None:
                    parent = parent_heads.get(tuple(parent_box))
                    blocked_parent = (
                        parent is None
                        or parent.attrs.get("light_raw") in ("red", "yellow", "countdown")
                        or parent.attrs.get("countdown", False)
                        or parent.attrs.get("light_walk_blocked", False)
                    )
                    d.attrs["focus_parent_veto"] = blocked_parent
                    if blocked_parent:
                        r = replace(r, state="unknown", figure=False)
                if r.state == "walk" or getattr(r, "figure", False):
                    kind = "pedestrian"
                kind = self._track_kind(d.track_id, kind)
                d.attrs["signal_kind"] = kind
                d.attrs["countdown"] = r.countdown
                d.attrs["light_raw"] = r.state
                d.attrs["figure"] = bool(getattr(r, "figure", False))
                blocked = pred is not None and cnn_rejects_walk(pred.probs, cnn.t_veto)
                if parent_box is not None:
                    blocked = blocked or d.attrs.get("focus_parent_veto", False)
                smoother = self.focus_smoother if parent_box is not None else self.smoother
                d.attrs["light_walk_blocked"] = blocked
                d.attrs["light_reset_on_veto"] = smoother.reset_walk_on_veto
                st = smoother.update(d.track_id, r.state, box=b, t=t, walk_blocked=blocked)
                d.attrs["light_smoothed"] = st
                if pred is not None:
                    st = cnn_hold_veto(st, r.state, pred.probs, cnn.t_veto)
                d.attrs["light_cnn_veto"] = st != d.attrs["light_smoothed"]
                if d.attrs.get("focus_parent_veto") and st == "walk":
                    st = "unknown"
                d.attrs["light_state"] = st
            if d.label == "traffic light" or "sign" in d.label:
                est = signal_distance.estimate(d.box, d.label, d.attrs.get("signal_kind", "unknown"),
                                               W, H, self.cfg.camera.hfov_deg, self.signal_dist_cfg)
                d.attrs.update(est.as_attrs())
        tm["lights"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if self.open_vocab is not None:
            if self._ov_sched.should_run(self._i):
                self._ov_sched.mark_attempt(self._i)
                try:
                    haz = self.open_vocab(frame)
                    self._ov_sched.record_result(haz, failed=False, frame_i=self._i)
                except Exception as e:
                    print(f"[open-vocab] frame failed: {type(e).__name__}: {e}")
                    self._ov_sched.record_result([], failed=True, frame_i=self._i)
            self._last_hazards = self._ov_sched.cached(self._i)
            if self._last_hazards:
                dets = list(dets) + list(self._last_hazards)
        tm["open_vocab"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if self.surface_seg is not None and (self._i % self.cfg.surface.every_n_frames == 0 or self._last_surface is None):
            self._last_surface = self.surface_seg(frame)
        tm["surface"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if self.depth is not None:
            every = max(1, self.cfg.depth.every_n_frames)
            if self._i % every == 0 or self._last_depth is None:
                self._last_depth = self.depth.infer(frame)
                self.depth.calibrate(self._last_depth, dets, W, H, self.cfg.camera.hfov_deg)
            prefer = getattr(self.cfg.depth, "prefer_over_box", True)
            for d in dets:
                box_d = estimate_distance_m(d.box, d.label, W, H, self.cfg.camera.hfov_deg)
                depth_d = self.depth.distance_m(d.box, self._last_depth)
                d.attrs["distance_box_m"] = box_d
                d.attrs["distance_depth_m"] = depth_d
                if "signal_distance_source" in d.attrs:
                    est = signal_distance.get_estimate(d, W, H, self.cfg.camera.hfov_deg, self.signal_dist_cfg)
                    d.attrs.update(signal_distance.fuse_depth(est, depth_d, self.signal_dist_cfg).as_attrs())
                    rel = signal_distance.sample_depth_rel(self._last_depth, d.box)
                    if rel is not None:
                        d.attrs["signal_depth_rel"] = rel
                if prefer and depth_d is not None:
                    d.attrs["distance_m"] = depth_d
                    d.attrs["distance_source"] = "depth"
                else:
                    d.attrs["distance_m"] = box_d
                    d.attrs["distance_source"] = "box"
        tm["depth"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if self._i % self.cfg.crosswalk_every_n_frames == 0 or self._last_cw is None:
            ground = np.isin(self._last_surface.labels, (ROAD, SIDEWALK)) if self._last_surface is not None else None
            self._last_cw = self.cw_hyst.update(detect_crosswalk(frame, ground_mask=ground))
        # Optional SAM 3.1 boost (slow): if classical missed a crosswalk, accept a text-prompted box.
        if self.sam is not None and (t - self._sam_last) >= self.cfg.sam.interval_s:
            self._sam_last = t
            try:
                segs = []
                for prompt in self.cfg.sam.prompts:
                    segs.append(self.sam.segment(frame, prompt, decode_masks=False))
                self._sam_segments = segs
                for seg in segs:
                    if seg.prompt == "crosswalk" and seg.boxes:
                        sam_cw = sam_boxes_to_crosswalk(seg.boxes, frame.shape)
                        if sam_cw is not None and (self._last_cw is None or not self._last_cw.found):
                            self._last_cw = sam_cw
                            break
            except Exception as e:
                print(f"[sam] frame failed: {type(e).__name__}: {e}")
        tm["crosswalk"] = (time.perf_counter() - t0) * 1000

        approaching = self.approach.update(t, dets)
        facts = FrameFacts(W, H, dets, self._last_cw, self._last_surface, approaching)
        cands = build_alerts(facts, self.cfg.camera.hfov_deg)
        for a in cands:
            self.alerts.submit(a)
        self.alerts.revalidate(facts)
        spoken = self.alerts.pop_ready()
        if spoken:
            self.recent.insert(0, spoken)
            del self.recent[5:]

        self.vlm.maybe_request(frame, t)
        # Route validated Muse hazard phrases into AlertManager (low priority).
        latest = self.vlm.latest
        if latest and isinstance(latest.data, dict):
            for i, phrase in enumerate(latest.data.get("hazards") or []):
                if isinstance(phrase, str) and phrase.strip():
                    self.alerts.submit(Alert(
                        f"muse_hazard:{phrase.strip().lower()[:40]}",
                        phrase.strip()[:80],
                        Priority.INFO,
                        data={"source": "muse"},
                    ))

        ann = None
        if annotate:
            t0 = time.perf_counter()
            vlm_text = self.vlm.latest.data.get("summary") if self.vlm.latest else None
            depth_ms = tm.get("depth", 0.0)
            fps_txt = f"detect {tm['detect']:.0f}ms surface {tm['surface']:.0f}ms"
            if self.depth is not None:
                fps_txt += f" depth {depth_ms:.0f}ms"
            fresh = [a for a in self.recent if t - a.ts < 4.0]
            depth_map = self._last_depth if self.depth is not None else None
            ann = draw(frame, dets, self._last_cw, self._last_surface, fresh, fps_txt, vlm_text, depth_map=depth_map)
            tm["overlay"] = (time.perf_counter() - t0) * 1000
        for k, v in tm.items():
            self.timing_totals[k] += v
        out = FrameOutput(self._i, t, dets, self._last_cw, self._last_surface, cands, spoken, tm, ann)
        if self.frame_history is not None:
            try:
                sel = select_crossing_signal(dets, W, H, self._last_cw)
                self.frame_history.add(t, frame, sel.decision if sel is not None else "unknown")
            except Exception as e:  # history must never break the fast loop
                print(f"[frame_history] skipped: {type(e).__name__}: {e}")
        if self.sam_fallback is not None and self.sam_fallback.in_crossing:
            try:
                sel = select_crossing_signal(dets, W, H, self._last_cw)
                self._sam_prev_state = sel.decision if sel is not None else "unknown"
            except Exception:
                self._sam_prev_state = "unknown"
        try:
            self.observations.add(summarise_frame(out, W, H))
        except Exception as e:  # the log must never break the fast loop
            print(f"[observations] skipped: {type(e).__name__}: {e}")
        self._i += 1
        return out
