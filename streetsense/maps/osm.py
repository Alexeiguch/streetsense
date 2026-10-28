"""Free, keyless provider: Valhalla (walking routes), Overpass (crossing tags), Nominatim (places).

OpenStreetMap data is ODbL, so it may be spoken aloud, cached and processed,
with attribution. The public servers are shared community resources: we send
at most about one request per second, set a real User-Agent, and record
fixtures so tests and demos do not hit them again.
"""

from __future__ import annotations

import os
import re
from typing import Any

from .geo import LatLon, decode_polyline, haversine_m, project_onto_line
from .http import Http
from .model import OSM_ATTRIBUTION, Crossing, Place, Route, Step

VALHALLA_URL = os.environ.get("STREETSENSE_VALHALLA_URL", "https://valhalla1.openstreetmap.de")
OVERPASS_URLS = [u for u in os.environ.get(
    "STREETSENSE_OVERPASS_URL",
    "https://overpass-api.de/api/interpreter,https://overpass.private.coffee/api/interpreter,"
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
).split(",") if u]
NOMINATIM_URL = os.environ.get("STREETSENSE_NOMINATIM_URL", "https://nominatim.openstreetmap.org")

# Valhalla maneuver type -> our step kind.
_KIND = {
    1: "depart", 2: "depart", 3: "depart", 4: "arrive", 5: "arrive", 6: "arrive", 7: "continue", 8: "continue",
    9: "slight_right", 10: "right", 11: "sharp_right", 12: "uturn", 13: "uturn", 14: "sharp_left", 15: "left",
    16: "slight_left", 22: "continue", 23: "slight_right", 24: "slight_left", 39: "elevator", 40: "stairs",
    41: "escalator",
}
_PASS_SIGNALS = re.compile(r"^Pass traffic signals on (.+?)\.?$", re.I)
_PASS = re.compile(r"^Pass (.+?)\.?$", re.I)
_SIGNAL_REFS = {"pelican", "puffin", "toucan", "pegasus", "tiger"}
_ROAD_HW = "^(trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|trunk_link|primary_link|secondary_link|tertiary_link)$"


def _yes(v: Any) -> bool | None:
    if v is None:
        return None
    v = str(v).lower()
    if v in {"yes", "true", "1", "incorrect"}:  # "incorrect" tactile paving still exists
        return True
    if v in {"no", "false", "0"}:
        return False
    return None


def crossing_from_tags(lat: float, lon: float, tags: dict, osm_id: int | None = None) -> Crossing:
    t = {k: str(v) for k, v in (tags or {}).items()}
    cr = t.get("crossing", "").lower()
    ref = t.get("crossing_ref", "").lower()
    signals = cr == "traffic_signals" or ref in _SIGNAL_REFS or t.get("highway") == "traffic_signals" or None
    if signals is None and cr in {"uncontrolled", "marked", "zebra", "unmarked", "no"}:
        signals = False
    if signals:
        kind = "traffic_signals"
    elif cr == "zebra" or ref == "zebra" or t.get("crossing:markings") == "zebra":
        kind = "zebra"
    elif cr in {"marked", "uncontrolled"}:
        kind = "marked"
    elif cr in {"unmarked", "informal"}:
        kind = "unmarked"
    elif t.get("crossing:markings", "no") not in {"no", ""}:
        kind = "marked"
    else:
        kind = "unknown"
    return Crossing(
        lat=lat, lon=lon, kind=kind, signals=signals,
        sound=_yes(t.get("traffic_signals:sound")),
        vibration=_yes(t.get("traffic_signals:vibration")),
        button=_yes(t.get("button_operated")),
        tactile=_yes(t.get("tactile_paving")),
        island=_yes(t.get("crossing:island")) or (True if t.get("traffic_calming") == "island" else None),
        source="osm", osm_id=osm_id,
    )


def parse_valhalla_trip(trip: dict, label: str = "") -> Route:
    leg = trip["legs"][0]
    pts = decode_polyline(leg["shape"], precision=6)
    steps: list[Step] = []
    crossings: list[Crossing] = []
    stairs = 0
    for m in leg.get("maneuvers", []):
        kind = _KIND.get(int(m.get("type", 8)), "continue")
        text = (m.get("instruction") or "").strip()
        street = (m.get("street_names") or [None])[0]
        idx = int(m.get("begin_shape_index", 0))
        landmark = None
        sig = _PASS_SIGNALS.match(text)
        if sig:
            kind = "pass"
            road = sig.group(1)
            p = pts[min(idx, len(pts) - 1)]
            crossings.append(Crossing(lat=p[0], lon=p[1], kind="traffic_signals", signals=True, road=road, source="valhalla"))
            landmark = f"traffic signals on {road}"
        elif kind == "continue" and _PASS.match(text):
            kind = "pass"
            landmark = _PASS.match(text).group(1)
        if kind == "stairs" or "stairs" in text.lower():
            kind = "stairs"
            stairs += 1
        steps.append(Step(instruction=text, distance_m=float(m.get("length", 0.0)) * 1000.0, start_index=idx,
                          kind=kind, street=street, landmark=landmark))
    summ = trip.get("summary", {})
    r = Route(points=pts, steps=steps, distance_m=float(summ.get("length", 0.0)) * 1000.0,
              duration_s=float(summ.get("time", 0.0)), source="osm", attribution=OSM_ATTRIBUTION,
              speakable=True, crossings=crossings, stairs=stairs, label=label)
    return r


def crossings_from_trace(route: Route, trace: dict) -> list[Crossing]:
    """Valhalla trace_attributes: every edge whose use is pedestrian_crossing is a street crossing."""
    out: list[Crossing] = []
    for e in trace.get("edges", []) or []:
        if e.get("use") != "pedestrian_crossing":
            continue
        i0, i1 = int(e.get("begin_shape_index", 0)), int(e.get("end_shape_index", 0))
        i0 = min(i0, len(route.points) - 1)
        i1 = min(max(i1, i0), len(route.points) - 1)
        a, b = route.points[i0], route.points[i1]
        out.append(Crossing(lat=(a[0] + b[0]) / 2, lon=(a[1] + b[1]) / 2, kind="unknown", source="valhalla"))
    return out


def crossings_from_overpass(data: dict) -> list[Crossing]:
    nodes = [e for e in data.get("elements", []) if e.get("type") == "node"]
    ways = [e for e in data.get("elements", []) if e.get("type") == "way"]
    road_of: dict[int, dict] = {}
    for w in ways:
        for nid in w.get("nodes", []):
            road_of.setdefault(nid, w.get("tags", {}))
    crossings: list[Crossing] = []
    signal_nodes: list[LatLon] = []
    for n in nodes:
        tags = n.get("tags", {})
        if tags.get("highway") == "traffic_signals" and "crossing" not in tags:
            signal_nodes.append((n["lat"], n["lon"]))
            continue
        c = crossing_from_tags(n["lat"], n["lon"], tags, n.get("id"))
        road = road_of.get(n.get("id"), {})
        c.road = road.get("name") or c.road
        try:
            c.lanes = int(str(road.get("lanes", "")).split(";")[0]) if road.get("lanes") else None
        except ValueError:
            c.lanes = None
        crossings.append(c)
    # A crossing with no crossing=* value next to a junction signal is very likely signalised.
    for c in crossings:
        if c.signals is None and any(haversine_m(c.point, s) < 15 for s in signal_nodes):
            c.signals, c.kind = True, "traffic_signals"
    return crossings


def attach_crossings(route: Route, candidates: list[Crossing], max_lateral_m: float = 12.0,
                     merge_m: float = 15.0) -> list[Crossing]:
    """Keep candidates on the route, order them by distance along it and merge duplicates.

    Overpass nodes carry the richest tags, so they win over Valhalla-derived ones.
    """
    rank = {"osm": 0, "valhalla": 1, "google": 2}
    on: list[Crossing] = []
    for c in candidates:
        lat_m, along, _, _ = project_onto_line(c.point, route.points, route.cum)
        if lat_m <= max_lateral_m:
            c.along_m = along
            on.append(c)
    on.sort(key=lambda c: (rank.get(c.source, 9), c.along_m))
    kept: list[Crossing] = []
    for c in on:
        twin = next((k for k in kept if abs((k.along_m or 0) - (c.along_m or 0)) < merge_m), None)
        if twin is None:
            kept.append(c)
        else:  # fill gaps in the richer record
            if twin.signals is None and c.signals is not None:
                twin.signals = c.signals
                if c.signals:
                    twin.kind = "traffic_signals"
            twin.road = twin.road or c.road
    kept.sort(key=lambda c: c.along_m or 0.0)
    return kept


def overpass_query(points: list[LatLon], pad_deg: float = 0.0004) -> str:
    s = min(p[0] for p in points) - pad_deg
    w = min(p[1] for p in points) - pad_deg
    n = max(p[0] for p in points) + pad_deg
    e = max(p[1] for p in points) + pad_deg
    bb = f"{s:.5f},{w:.5f},{n:.5f},{e:.5f}"
    return (
        "[out:json][timeout:25];"
        f"(node({bb})[highway=crossing];node({bb})[footway=crossing];node({bb})[highway=traffic_signals];)->.c;"
        ".c out body;"
        f'way(bn.c)[highway~"{_ROAD_HW}"];out body;'
    )


class OSMProvider:
    name = "osm"
    speakable = True

    def __init__(self, http: Http | None = None, valhalla_url: str = VALHALLA_URL,
                 overpass_urls: list[str] | None = None, nominatim_url: str = NOMINATIM_URL):
        self.http = http or Http()
        self.valhalla_url = valhalla_url.rstrip("/")
        self.overpass_urls = list(overpass_urls or OVERPASS_URLS)
        self.nominatim_url = nominatim_url.rstrip("/")
        self.warnings: list[str] = []

    # Routing ---------------------------------------------------------------
    def routes(self, origin: LatLon, dest: LatLon, alternates: int = 2) -> list[Route]:
        body = {
            "locations": [{"lat": round(origin[0], 6), "lon": round(origin[1], 6)},
                          {"lat": round(dest[0], 6), "lon": round(dest[1], 6)}],
            "costing": "pedestrian",
            # "blind" turns on Valhalla's guidance for blind users: landmarks, signals, bollards, stairs.
            "costing_options": {"pedestrian": {"type": "blind"}},
            "alternates": int(alternates),
            "directions_options": {"units": "kilometers", "language": "en-US"},
        }
        data = self.http.request("POST", f"{self.valhalla_url}/route", json_body=body)
        trips = [data["trip"]] + [a["trip"] for a in data.get("alternates", []) or []]
        routes = [parse_valhalla_trip(t, label=f"route {i + 1}") for i, t in enumerate(trips)]
        self.enrich_crossings(routes)
        return routes

    def enrich_crossings(self, routes: list[Route]) -> None:
        """One Overpass query for all candidate routes, then per-route matching."""
        tagged: list[Crossing] = []
        q = overpass_query([p for r in routes for p in r.points])
        for url in self.overpass_urls:  # public mirrors are often busy; try each once
            try:
                tagged = crossings_from_overpass(self.http.request("POST", url, form={"data": q}))
                break
            except Exception as e:
                self.warnings.append(f"overpass {url.split('/')[2]}: {type(e).__name__}")
        for r in routes:
            self._enrich_one(r, tagged)

    def _enrich_one(self, route: Route, tagged: list[Crossing]) -> None:
        import copy

        cands = [copy.copy(c) for c in tagged] + list(route.crossings)
        try:
            body = {"encoded_polyline": _encode6(route.points), "costing": "pedestrian", "shape_match": "edge_walk",
                    "filters": {"attributes": ["edge.use", "edge.begin_shape_index", "edge.end_shape_index"],
                                "action": "include"}}
            trace = self.http.request("POST", f"{self.valhalla_url}/trace_attributes", json_body=body)
            cands += crossings_from_trace(route, trace)
        except Exception as e:  # degrade, never fail the walk
            self.warnings.append(f"trace_attributes: {type(e).__name__}")
        route.crossings = attach_crossings(route, cands)

    # Places ------------------------------------------------------------------
    def search(self, query: str, near: LatLon | None = None, limit: int = 5) -> list[Place]:
        params: dict[str, Any] = {"format": "jsonv2", "q": query[:120], "limit": limit, "extratags": 1}
        if near:
            d = 0.01  # about 1 km box to prefer nearby results
            params["viewbox"] = f"{near[1] - d:.4f},{near[0] + d:.4f},{near[1] + d:.4f},{near[0] - d:.4f}"
            params["bounded"] = 1
        rows = self.http.request("GET", f"{self.nominatim_url}/search", params=params) or []
        out = []
        for r in rows:
            p = Place(name=(r.get("name") or r.get("display_name", "")).split(",")[0], lat=float(r["lat"]),
                      lon=float(r["lon"]), source="osm", address=r.get("display_name"),
                      wheelchair=(r.get("extratags") or {}).get("wheelchair"))
            if near:
                p.distance_m = haversine_m(near, p.point)
            out.append(p)
        if near:
            out.sort(key=lambda p: p.distance_m or 0)
        return out

    def reverse(self, p: LatLon) -> dict:
        r = self.http.request("GET", f"{self.nominatim_url}/reverse",
                              params={"format": "jsonv2", "lat": round(p[0], 5), "lon": round(p[1], 5), "zoom": 18})
        a = (r or {}).get("address", {})
        return {"road": a.get("road") or a.get("pedestrian") or a.get("footway"), "name": (r or {}).get("name"),
                "suburb": a.get("suburb") or a.get("neighbourhood"), "city": a.get("city") or a.get("town")}


def _encode6(points: list[LatLon]) -> str:
    from .geo import encode_polyline

    return encode_polyline(points, precision=6)
