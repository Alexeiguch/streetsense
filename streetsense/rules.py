"""Turn per-frame perception into candidate alerts (the manager decides what is spoken)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .crosswalk import CrosswalkResult
from .geometry import estimate_distance_m, horizontal_zone, in_walking_path, round_distance, truncation_flags
from .signal_distance import (
    CAUTION_SPEECH,
    DistanceEstimate,
    SignalDistanceConfig,
    camera_hfov_deg,
    get_estimate,
    spoken_distance,
    walk_caution,
)
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
    distance: DistanceEstimate | None = None  # estimate for `detection`
    caution: str | None = None  # signal_distance caution code when distance blocked a walk

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
    hfov_deg: float | None = None,
    dist_cfg: SignalDistanceConfig | None = None,
) -> SignalSelection | None:
    """Pick the signal that controls the crossing ahead; unclear when relevant heads conflict.

    Distance rules (signal_distance.walk_caution) can only turn a walk into unclear.
    """
    hfov = camera_hfov_deg() if hfov_deg is None else hfov_deg
    dcfg = dist_cfg or SignalDistanceConfig.from_env()

    def sel(decision: str, d: Detection | None, reason: str, caution: str | None = None) -> SignalSelection:
        est = get_estimate(d, frame_w, frame_h, hfov, dcfg) if d is not None else None
        return SignalSelection(decision, d, reason, est, caution)

    lights = [
        d for d in dets
        if d.label == "traffic light"
        and d.attrs.get("light_state", "unknown") not in ("unknown", None, "")
    ]
    if not lights:
        return None
    # Heads we could not read (bloom, washout): never usable themselves, but a nearer one
    # ahead means the readable walk/green head may belong to a different crossing.
    unreadable = [
        d for d in dets
        if d.label == "traffic light"
        and d.attrs.get("light_state", "unknown") in ("unknown", None, "")
        and d.attrs.get("signal_kind") != "vehicle"
    ]

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
        state = d.attrs.get("light_state", "unknown")
        if state == "walk":  # never reachable via the vehicle filter, but keep it impossible
            state = "unclear"
        return sel(state, d, "best overall")

    states = {d.attrs.get("light_state") for d in relevant}
    ped_action = states & {"walk", "red"}
    if len(ped_action) > 1:
        # Conflicting walk vs red among relevant heads → do not pick walk.
        return sel("unclear", relevant[0], "conflicting pedestrian signals")

    # Prefer an explicit walk when present and not contradicted by red (sticky vehicle-green
    # heads must not silence a real walk). Otherwise keep relevance order.
    if "walk" in states:
        walks = [d for d in relevant if d.attrs.get("light_state") == "walk"]

        def dist_key(d: Detection) -> float:
            m = get_estimate(d, frame_w, frame_h, hfov, dcfg).m
            return m if m is not None else float("inf")

        walk_d = min(walks, key=dist_key)  # nearest walk head
        caution = walk_caution(walk_d, relevant, frame_w, frame_h, hfov, dcfg, unreadable)
        if caution is not None:
            return sel("unclear", walk_d, caution[1], caution[0])
        return sel("walk", walk_d, "walk among relevant")

    top = relevant[0]
    state = top.attrs.get("light_state", "unknown")
    if state == "countdown":
        state = "unclear"
    if state == "green":
        # Same distance checks as walk: a far/side/out-ranked green must not be announced as ours.
        caution = walk_caution(top, relevant, frame_w, frame_h, hfov, dcfg, unreadable)
        if caution is not None:
            return sel("unclear", top, caution[1], caution[0])
    return sel(state if state in _ALERTABLE_LIGHT else "unknown", top, "relevance")


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

    sel = select_crossing_signal(f.detections, f.width, f.height, f.crosswalk, hfov_deg=hfov_deg)
    if sel is not None and sel.decision in _ALERTABLE_LIGHT:
        state, kind = sel.decision, sel.attrs.get("signal_kind", "unknown")
        name = "pedestrian light" if kind == "pedestrian" else "traffic light"
        light = sel.detection
        where = spoken_distance(sel.distance)
        if state == "unclear":
            text = CAUTION_SPEECH.get(sel.caution or "", "crossing signals unclear, wait")
            prio = Priority.WARNING
        elif state == "walk":
            text = "walk signal on" + (f", {where}" if where else "") + (
                ", countdown running" if light and light.attrs.get("countdown") else "")
            prio = Priority.NOTICE
        else:
            text = f"{name} is " + {"red": "red", "yellow": "amber", "green": "green"}.get(state, state)
            text += f", {where}" if where else ""
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
                "distance_m": sel.distance.m if sel.distance else None,
                "caution": sel.caution,
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
