"""Turn per-frame perception into candidate alerts (the manager decides what is spoken)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .crosswalk import CrosswalkResult
from .geometry import estimate_distance_m, horizontal_zone, in_walking_path, round_distance
from .surface import SurfaceResult
from .types import Alert, Detection, Priority

VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle", "train"}
OBSTACLES = {"bench", "fire hydrant", "parking meter", "chair", "potted plant", "stop sign", "dog", "suitcase",
            "bollard", "traffic cone", "scooter", "construction barrier", "barrier",
            "stairs", "step", "pothole", "manhole", "kerb"}


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


def most_relevant_light(dets: list[Detection]) -> Detection | None:
    """Several heads are usually in view; speak about one. Pedestrian heads win, then the
    action-critical state (walk/red over green), then the biggest (nearest) box."""
    lights = [d for d in dets if d.label == "traffic light" and d.attrs.get("light_state", "unknown") != "unknown"]
    if not lights:
        return None
    kind_rank = {"pedestrian": 2, "unknown": 1, "vehicle": 0}
    # walk/red matter more than a sticky vehicle-green mislabelled as pedestrian
    state_rank = {"walk": 4, "red": 3, "yellow": 2, "green": 1}
    return max(
        lights,
        key=lambda d: (
            kind_rank.get(d.attrs.get("signal_kind", "unknown"), 1),
            state_rank.get(d.attrs.get("light_state", "unknown"), 0),
            d.box.area,
        ),
    )


def build_alerts(f: FrameFacts, hfov_deg: float = 70.0) -> list[Alert]:
    out: list[Alert] = []
    for d in f.detections:
        zone = horizontal_zone(d.box, f.width)
        dist = d.attrs.get("distance_m")
        if dist is None:
            dist = estimate_distance_m(d.box, d.label, f.width, f.height, hfov_deg)
            d.attrs["distance_m"] = dist
        d.attrs["zone"] = zone
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
                spoken = {
                    "bollard": "bollard", "traffic cone": "traffic cone", "scooter": "scooter",
                    "construction barrier": "construction barrier", "barrier": "barrier",
                    "stairs": "stairs", "step": "step", "pothole": "pothole",
                    "manhole": "open manhole", "kerb": "kerb",
                }.get(d.label, d.label)
                out.append(Alert(f"obstacle:{d.label}:{zone}", f"{spoken} {_where(zone)}, {round_distance(dist)}",
                                 Priority.WARNING, data={"label": d.label, "distance_m": dist}))
    light = most_relevant_light(f.detections)
    if light is not None:
        state, kind = light.attrs["light_state"], light.attrs.get("signal_kind", "unknown")
        name = "pedestrian light" if kind == "pedestrian" else "traffic light"
        if state == "walk":
            text = "walk signal on" + (", countdown running" if light.attrs.get("countdown") else "")
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
