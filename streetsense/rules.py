"""Turn per-frame perception into candidate alerts (the manager decides what is spoken)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .crosswalk import CrosswalkResult
from .geometry import estimate_distance_m, horizontal_zone, in_walking_path, round_distance
from .surface import SurfaceResult
from .types import Alert, Detection, Priority

VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle", "train"}
OBSTACLES = {"bench", "fire hydrant", "parking meter", "chair", "potted plant", "stop sign", "dog", "suitcase"}


@dataclass
class FrameFacts:
    """Everything the rules need about one frame."""

    width: int
    height: int
    detections: list[Detection]
    crosswalk: CrosswalkResult | None = None
    surface: SurfaceResult | None = None
    approaching: set[int] = field(default_factory=set)  # track ids whose boxes are growing fast


def _where(zone: str) -> str:
    return {"left": "on your left", "right": "on your right", "ahead": "ahead"}[zone]


def build_alerts(f: FrameFacts, hfov_deg: float = 70.0) -> list[Alert]:
    out: list[Alert] = []
    for d in f.detections:
        zone = horizontal_zone(d.box, f.width)
        dist = estimate_distance_m(d.box, d.label, f.width, f.height, hfov_deg)
        d.attrs["zone"] = zone
        d.attrs["distance_m"] = dist
        path = in_walking_path(d.box, f.width, f.height)

        if d.label in VEHICLES:
            approaching = d.track_id is not None and d.track_id in f.approaching
            if dist is not None and dist < 8 and (path or approaching):
                word = "bicycle" if d.label == "bicycle" else ("motorbike" if d.label == "motorcycle" else d.label)
                verb = "approaching" if approaching else "close"
                out.append(Alert(f"vehicle:{zone}", f"{word} {verb} {_where(zone)}", Priority.CRITICAL,
                                 data={"label": d.label, "distance_m": dist}))
            elif approaching and dist is not None and dist < 20:
                out.append(Alert(f"vehicle_far:{zone}", f"{d.label} approaching {_where(zone)}", Priority.WARNING,
                                 data={"label": d.label, "distance_m": dist}))
        elif d.label == "person":
            if path and dist is not None and dist < 4:
                out.append(Alert("person:ahead" if zone == "ahead" else f"person:{zone}",
                                 f"person {_where(zone)}, {round_distance(dist)}", Priority.WARNING,
                                 data={"distance_m": dist}))
        elif d.label in OBSTACLES:
            if path and dist is not None and dist < 5:
                out.append(Alert(f"obstacle:{zone}", f"{d.label} {_where(zone)}, {round_distance(dist)}",
                                 Priority.WARNING, data={"label": d.label, "distance_m": dist}))
        elif d.label == "traffic light":
            state = d.attrs.get("light_state", "unknown")
            kind = d.attrs.get("signal_kind", "unknown")
            if state == "unknown":
                continue
            name = "pedestrian light" if kind == "pedestrian" else "traffic light"
            if state == "walk":
                text = "walk signal on" + (", countdown running" if d.attrs.get("countdown") else "")
            else:
                text = f"{name} is " + {"red": "red", "yellow": "amber", "green": "green"}[state]
            prio = Priority.NOTICE
            if f.crosswalk and f.crosswalk.found and kind != "vehicle":
                prio = Priority.WARNING  # at a crossing, the pedestrian light is what matters most
            out.append(Alert(f"light:{kind}:{state}", text, prio, data={"state": state, "kind": kind}))

    if f.crosswalk and f.crosswalk.found and f.crosswalk.box is not None:
        zone = horizontal_zone(f.crosswalk.box, f.width)
        near = f.crosswalk.box.y2 > f.height * 0.85
        out.append(Alert("crosswalk", f"pedestrian crossing {'here' if near else _where(zone)}", Priority.NOTICE,
                         data={"stripes": f.crosswalk.n_stripes}))

    if f.surface is not None:
        if f.surface.underfoot == "road":
            # Standing in the road with no crossing in view is the most dangerous state.
            on_crossing = bool(f.crosswalk and f.crosswalk.found)
            if not on_crossing:
                out.append(Alert("surface:road", "careful, road ahead of you", Priority.WARNING))
        if f.surface.pole_ahead:
            out.append(Alert("obstacle:pole", "pole ahead", Priority.WARNING))
    return out
