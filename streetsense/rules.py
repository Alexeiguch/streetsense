"""Turn per-frame perception into candidate alerts (the manager decides what is spoken)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .crosswalk import CrosswalkResult
from .geometry import estimate_distance_m, horizontal_zone, in_walking_path, round_distance, truncation_flags
from .surface import SurfaceResult
from .types import Alert, Box, Detection, Priority

VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle", "train"}
OBSTACLES = {"bench", "fire hydrant", "parking meter", "chair", "potted plant", "stop sign", "dog", "suitcase",
            "bollard", "traffic cone", "scooter", "construction barrier", "barrier",
            "stairs", "step", "pothole", "manhole", "kerb"}

# States that may reach the alert layer. Internal classifier states stay out.
_ALERTABLE_LIGHT = frozenset({"red", "yellow", "green", "walk", "unclear"})


@dataclass
class FrameFacts:
    """Everything the rules need about one frame."""

    width: int
    height: int
    detections: list[Detection]
    crosswalk: CrosswalkResult | None = None
    surface: SurfaceResult | None = None
    approaching: set[int] = field(default_factory=set)  # track ids whose boxes are growing fast


@dataclass
class SignalSelection:
    """Result of choosing which pedestrian signal controls the crossing ahead."""

    decision: str  # walk | red | yellow | green | unclear | unknown
    detection: Detection | None = None
    reason: str = ""

    @property
    def attrs(self) -> dict:
        if self.detection is not None:
            a = dict(self.detection.attrs)
            a["light_state"] = self.decision if self.decision != "unclear" else a.get("light_state", "unclear")
            a["signal_kind"] = a.get("signal_kind", "pedestrian")
            return a
        return {"light_state": self.decision, "signal_kind": "pedestrian"}


def _where(zone: str) -> str:
    return {"left": "on your left", "right": "on your right", "ahead": "ahead"}[zone]


def _centrality(box: Box, frame_w: int) -> float:
    """1.0 = dead centre, 0.0 = at the edge."""
    if frame_w <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(box.cx / frame_w - 0.5) * 2.0)


def _relevance_score(d: Detection, frame_w: int, frame_h: int, crosswalk: CrosswalkResult | None) -> float:
    """Higher = more likely to control the crossing the walker is facing."""
    kind_rank = {"pedestrian": 2.0, "unknown": 1.0, "vehicle": 0.0}
    kind = d.attrs.get("signal_kind", "unknown")
    area_norm = min(1.0, d.box.area / max(1.0, frame_w * frame_h * 0.02))
    centr = _centrality(d.box, frame_w)
    # Prefer lower-in-frame heads slightly (closer / on our side of the junction).
    height_bias = min(1.0, d.box.cy / max(1.0, frame_h))
    score = kind_rank.get(kind, 0.5) * 3.0 + centr * 4.0 + area_norm * 3.0 + height_bias * 0.5
    if crosswalk and crosswalk.found and crosswalk.box is not None:
        # Bonus when horizontally near the detected crosswalk.
        dx = abs(d.box.cx - crosswalk.box.cx) / max(1.0, frame_w)
        score += max(0.0, 1.5 * (1.0 - dx * 2.0))
    return score


def select_crossing_signal(
    dets: list[Detection],
    frame_w: int,
    frame_h: int,
    crosswalk: CrosswalkResult | None = None,
) -> SignalSelection | None:
    """Pick the signal that controls the crossing ahead; unclear when relevant heads conflict."""
    lights = [
        d for d in dets
        if d.label == "traffic light"
        and d.attrs.get("light_state", "unknown") not in ("unknown", None, "")
    ]
    if not lights:
        return None

    # Score all; keep pedestrian/unknown (not pure vehicle) as candidates for ped guidance.
    scored = sorted(
        ((_relevance_score(d, frame_w, frame_h, crosswalk), d) for d in lights),
        key=lambda x: x[0],
        reverse=True,
    )
    # Candidates within 70% of the best relevance score.
    best_score = scored[0][0]
    relevant = [d for s, d in scored if s >= best_score * 0.7 and d.attrs.get("signal_kind") != "vehicle"]
    if not relevant:
        # Fall back to best overall (may be vehicle) for informational notice only.
        d = scored[0][1]
        return SignalSelection(d.attrs.get("light_state", "unknown"), d, "best overall")

    states = {d.attrs.get("light_state") for d in relevant}
    ped_action = states & {"walk", "red"}
    if len(ped_action) > 1:
        # Conflicting walk vs red among relevant heads → do not pick walk.
        return SignalSelection("unclear", relevant[0], "conflicting pedestrian signals")

    # Prefer an explicit walk when present and not contradicted by red (sticky vehicle-green
    # heads must not silence a real walk). Otherwise keep relevance order.
    if "walk" in states:
        walk_d = next(d for d in relevant if d.attrs.get("light_state") == "walk")
        return SignalSelection("walk", walk_d, "walk among relevant")

    top = relevant[0]
    state = top.attrs.get("light_state", "unknown")
    if state == "countdown":
        state = "unclear"
    return SignalSelection(state if state in _ALERTABLE_LIGHT else "unknown", top, "relevance")


def most_relevant_light(dets: list[Detection]) -> Detection | None:
    """Back-compat wrapper: returns the detection, or a synthetic unclear detection."""
    # Assume 1280x720 if unknown — callers that care pass through select_crossing_signal.
    sel = select_crossing_signal(dets, 1280, 720, None)
    if sel is None:
        return None
    if sel.decision == "unclear":
        d = sel.detection or Detection(
            "traffic light", Box(0, 0, 1, 1), 0.0,
            attrs={"light_state": "unclear", "signal_kind": "pedestrian"},
        )
        d.attrs = dict(d.attrs)
        d.attrs["light_state"] = "unclear"
        return d
    if sel.detection is not None:
        # Expose selected decision on attrs for callers reading light_state.
        sel.detection.attrs = dict(sel.detection.attrs)
        sel.detection.attrs["light_state"] = sel.decision
        return sel.detection
    return None


def build_alerts(f: FrameFacts, hfov_deg: float = 70.0) -> list[Alert]:
    out: list[Alert] = []
    for d in f.detections:
        zone = horizontal_zone(d.box, f.width)
        trunc = truncation_flags(d.box, f.width, f.height)
        dist = d.attrs.get("distance_m")
        if dist is None:
            dist = estimate_distance_m(d.box, d.label, f.width, f.height, hfov_deg)
            d.attrs["distance_m"] = dist
        d.attrs["zone"] = zone
        d.attrs["truncated"] = trunc
        path = in_walking_path(d.box, f.width, f.height)

        # Conservative proximity when geometry is unreliable (single-edge clip).
        close_uncertain = bool(trunc.get("uncertain")) and path and d.label in (VEHICLES | {"person"} | OBSTACLES)

        if d.label in VEHICLES:
            approaching = d.track_id is not None and d.track_id in f.approaching
            if (dist is not None and dist < 8 and (path or approaching)) or (close_uncertain and approaching):
                word = "bicycle" if d.label == "bicycle" else ("motorbike" if d.label == "motorcycle" else d.label)
                verb = "approaching" if approaching else "close"
                out.append(Alert(f"vehicle:{zone}", f"{word} {verb} {_where(zone)}", Priority.CRITICAL,
                                 data={"label": d.label, "distance_m": dist}))
            elif approaching and dist is not None and dist < 20:
                out.append(Alert(f"vehicle_far:{zone}", f"{d.label} approaching {_where(zone)}", Priority.WARNING,
                                 data={"label": d.label, "distance_m": dist}))
        elif d.label == "person":
            near = (dist is not None and dist < 4 and path) or (close_uncertain and path)
            if near:
                if trunc.get("uncertain") and dist is None:
                    text = f"person very close {_where(zone)}"
                else:
                    text = f"person {_where(zone)}, {round_distance(dist or 1.0)}"
                out.append(Alert("person:ahead" if zone == "ahead" else f"person:{zone}",
                                 text, Priority.WARNING,
                                 data={"distance_m": dist, "truncated": trunc}))
        elif d.label in OBSTACLES:
            if (path and dist is not None and dist < 5) or (close_uncertain and path):
                spoken = {
                    "bollard": "bollard", "traffic cone": "traffic cone", "scooter": "scooter",
                    "construction barrier": "construction barrier", "barrier": "barrier",
                    "stairs": "stairs", "step": "step", "pothole": "pothole",
                    "manhole": "open manhole", "kerb": "kerb",
                }.get(d.label, d.label)
                phrase = f"{spoken} {_where(zone)}"
                if dist is not None and not trunc.get("uncertain"):
                    phrase += f", {round_distance(dist)}"
                elif trunc.get("uncertain"):
                    phrase += ", very close"
                out.append(Alert(f"obstacle:{d.label}:{zone}", phrase,
                                 Priority.WARNING, data={"label": d.label, "distance_m": dist}))

    sel = select_crossing_signal(f.detections, f.width, f.height, f.crosswalk)
    if sel is not None and sel.decision in _ALERTABLE_LIGHT:
        state, kind = sel.decision, sel.attrs.get("signal_kind", "unknown")
        name = "pedestrian light" if kind == "pedestrian" else "traffic light"
        light = sel.detection
        if state == "unclear":
            text = "crossing signals unclear, wait"
            prio = Priority.WARNING
        elif state == "walk":
            text = "walk signal on" + (", countdown running" if light and light.attrs.get("countdown") else "")
            prio = Priority.NOTICE
        else:
            text = f"{name} is " + {"red": "red", "yellow": "amber", "green": "green"}.get(state, state)
            prio = Priority.NOTICE
        if f.crosswalk and f.crosswalk.found and kind != "vehicle" and state != "unclear":
            prio = Priority.WARNING
        family = f"light:{kind}"
        out.append(Alert(
            family,
            text,
            prio,
            data={
                "state": state,
                "kind": kind,
                "family": family,
                "cooldown_key": f"{family}:{state}",
            },
        ))

    if f.crosswalk and f.crosswalk.found and f.crosswalk.box is not None:
        zone = horizontal_zone(f.crosswalk.box, f.width)
        near = f.crosswalk.box.y2 > f.height * 0.85
        out.append(Alert("crosswalk", f"pedestrian crossing {'here' if near else _where(zone)}", Priority.NOTICE,
                         data={"stripes": f.crosswalk.n_stripes}))

    if f.surface is not None:
        if f.surface.underfoot == "road":
            on_crossing = bool(f.crosswalk and f.crosswalk.found)
            if not on_crossing:
                out.append(Alert("surface:road", "careful, road ahead of you", Priority.WARNING))
        if f.surface.pole_ahead:
            out.append(Alert("obstacle:pole", "pole ahead", Priority.WARNING))
    return out
