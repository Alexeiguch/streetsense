"""MapSession: the walking state shared by the CLI, the web app and the Wonder guide tools."""

from __future__ import annotations

from typing import Any

from .geo import LatLon, bearing_deg, clock_face, haversine_m, relative_bearing, round_m
from .model import Place, Route, choose_safer_route
from .navigator import Fix, NavConfig, Navigator

MAP_TOOLS: list[dict] = [
    {
        "type": "function",
        "name": "plan_route",
        "description": (
            "Plan a walking route from the user's position to a destination named in words, e.g. 'pharmacy' "
            "or 'Renwick Gallery'. Picks the most careful of several routes (signalised crossings, no stairs). "
            "Returns length, minutes, crossings and why it was chosen. Free (OpenStreetMap)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"destination": {"type": "string", "description": "Place name or type, max 80 characters."}},
            "required": ["destination"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "route_status",
        "description": (
            "Where the user is on the current route: metres left, next turn, next crossing with clock-face "
            "direction and map features, whether they seem off route, GPS accuracy. Free."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "find_nearby",
        "description": "Find up to 3 nearby places of a type or name, with distance and clock-face direction. Free (OpenStreetMap).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "e.g. 'pharmacy', 'bus stop', max 80 characters."}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]

GOOGLE_TOOL: dict = {
    "type": "function",
    "name": "google_maps_search",
    "description": (
        "Ask Google Maps about places: opening hours, wheelchair-accessible entrance, what a place is. "
        "Use only when OpenStreetMap lacks the fact. Costs money. Max 1 per question."
    ),
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Specific question with the place name, max 120 characters."}},
        "required": ["query"],
        "additionalProperties": False,
    },
}

class MapSession:
    def __init__(self, provider: Any, google: Any = None, nav_cfg: NavConfig | None = None):
        self.provider = provider
        self.google = google
        self.nav_cfg = nav_cfg
        self.nav: Navigator | None = None
        self.position: LatLon | None = None
        self.heading: float | None = None
        self.accuracy_m: float = 10.0
        self.routes: list[Route] = []
        self.google_calls = 0
        self.sources: list[dict] = []  # Google Maps sources for the companion transcript

    @property
    def tools(self) -> list[dict]:
        return MAP_TOOLS + ([GOOGLE_TOOL] if self.google is not None and getattr(self.google, "enabled", False) else [])

    def set_position(self, lat: float, lon: float, heading: float | None = None, accuracy_m: float = 10.0, t: float = 0.0) -> list:
        self.position, self.heading, self.accuracy_m = (lat, lon), heading, accuracy_m
        if self.nav is not None:
            return self.nav.update(Fix(t=t, lat=lat, lon=lon, heading=heading, accuracy_m=accuracy_m))
        return []

    def _clock(self, p: LatLon) -> int | None:
        if self.position is None or self.heading is None:
            return None
        return clock_face(relative_bearing(bearing_deg(self.position, p), self.heading))

    def start_route(self, dest: Place) -> dict:
        if self.position is None:
            return {"error": "no GPS position yet"}
        self.routes = self.provider.routes(self.position, dest.point)
        best, why = choose_safer_route(self.routes)
        self.nav = Navigator(best, dest.name, cfg=self.nav_cfg, reverse=getattr(self.provider, "reverse", None))
        self.nav.update(Fix(t=0.0, lat=self.position[0], lon=self.position[1], heading=self.heading,
                            accuracy_m=self.accuracy_m))
        start = self.nav.start()
        return {"destination": dest.name, "chosen": best.summary(), "why": why,
                "alternatives": [r.summary() for r in self.routes if r is not best],
                "first_cue": start.text, "attribution": best.attribution}

    # Wonder tool handlers -------------------------------------------------------
    def plan_route(self, destination: str) -> dict:
        places = self.find_places(destination)
        if not places:
            return {"error": f"no place found for '{destination[:40]}' nearby"}
        return self.start_route(places[0])

    def find_places(self, query: str) -> list[Place]:
        return self.provider.search(query, near=self.position)[:3]

    def find_nearby(self, query: str) -> dict:
        out = []
        for p in self.find_places(query):
            d = haversine_m(self.position, p.point) if self.position else None
            out.append({"name": p.name, "distance_m": round_m(d) if d else None, "clock": self._clock(p.point),
                        "wheelchair": p.wheelchair})
        return {"query": query, "places": out, "source": "OpenStreetMap"}

    def route_status(self) -> dict:
        if self.nav is None:
            return {"note": "no active route; call plan_route first"}
        return self.nav.status()

    def google_search(self, query: str) -> dict:
        if self.google is None or not getattr(self.google, "enabled", False):
            return {"error": "Google Maps not enabled"}
        if self.google_calls >= 1:
            return {"error": "Google Maps limit reached for this question"}
        self.google_calls += 1
        res = self.google.search_places(query, near=self.position)
        self.sources.extend(s.to_dict() for s in res.sources)
        return {"google_maps": res.data, "sources_shown_in_app": len(res.sources)}

    def dispatch(self, name: str, args: dict) -> dict | None:
        """Returns None when the tool is not a map tool."""
        if name == "plan_route":
            dest = str(args.get("destination", "")).strip()
            if not dest or len(dest) > 80:
                return {"error": "destination must be 1 to 80 characters"}
            return self.plan_route(dest)
        if name == "route_status":
            return self.route_status()
        if name == "find_nearby":
            q = str(args.get("query", "")).strip()
            if not q or len(q) > 80:
                return {"error": "query must be 1 to 80 characters"}
            return self.find_nearby(q)
        if name == "google_maps_search":
            q = str(args.get("query", "")).strip()
            if not q or len(q) > 120:
                return {"error": "query must be 1 to 120 characters"}
            return self.google_search(q)
        return None

    def new_question(self) -> None:
        self.google_calls = 0
