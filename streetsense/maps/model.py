"""Plain data types shared by every map provider."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .geo import LatLon, cumulative_m

OSM_ATTRIBUTION = "Map data (c) OpenStreetMap contributors, ODbL. Routing by Valhalla (FOSSGIS)."
GOOGLE_ATTRIBUTION = "Google Maps"

# How careful a crossing is for a blind walker, lower is better. Used only to rank
# routes before the walk. It never says anything about whether to cross now.
CROSSING_RISK_M = {
    "signals_accessible": 20.0,  # signals plus sound, vibration or a button
    "signals": 30.0,
    "zebra": 45.0,
    "marked": 60.0,
    "uncontrolled": 70.0,
    "unknown": 80.0,
    "unmarked": 100.0,
}
STAIRS_PENALTY_M = 40.0


@dataclass
class Step:
    instruction: str
    distance_m: float
    start_index: int
    kind: str = "continue"  # depart, left, right, slight_left, slight_right, sharp_*, uturn, pass, stairs, arrive, continue
    street: str | None = None
    landmark: str | None = None
    along_m: float = 0.0


@dataclass
class Crossing:
    lat: float
    lon: float
    kind: str = "unknown"  # traffic_signals, zebra, marked, uncontrolled, unmarked, unknown
    signals: bool | None = None
    sound: bool | None = None
    vibration: bool | None = None
    button: bool | None = None
    tactile: bool | None = None
    island: bool | None = None
    lanes: int | None = None
    road: str | None = None
    along_m: float | None = None
    source: str = "osm"
    osm_id: int | None = None

    @property
    def point(self) -> LatLon:
        return (self.lat, self.lon)

    @property
    def risk_class(self) -> str:
        if self.signals:
            return "signals_accessible" if (self.sound or self.vibration or self.button) else "signals"
        return self.kind if self.kind in CROSSING_RISK_M else "unknown"

    def features(self) -> list[str]:
        f = []
        if self.signals:
            f.append("signals")
        elif self.kind == "zebra":
            f.append("zebra, no signals")
        elif self.kind in ("marked", "uncontrolled"):
            f.append("marked, no signals")
        elif self.kind == "unmarked":
            f.append("unmarked")
        if self.button:
            f.append("push button")
        if self.sound:
            f.append("sound")
        elif self.signals and self.sound is False:
            f.append("no sound")
        if self.vibration:
            f.append("vibrating button")
        if self.tactile:
            f.append("tactile paving")
        if self.island:
            f.append("island in the middle")
        if self.lanes:
            f.append(f"{self.lanes} lane" + ("s" if self.lanes != 1 else ""))
        return f

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = self.features()
        return d


@dataclass
class Place:
    name: str
    lat: float
    lon: float
    source: str = "osm"
    address: str | None = None
    wheelchair: str | None = None  # yes, limited, no, or None when unknown
    link: str | None = None
    distance_m: float | None = None

    @property
    def point(self) -> LatLon:
        return (self.lat, self.lon)


@dataclass
class Route:
    points: list[LatLon]
    steps: list[Step]
    distance_m: float
    duration_s: float
    source: str = "osm"
    attribution: str = OSM_ATTRIBUTION
    # False for content whose licence forbids text to speech (raw Google Routes or Places).
    speakable: bool = True
    crossings: list[Crossing] = field(default_factory=list)
    stairs: int = 0
    label: str = ""
    cum: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.cum and self.points:
            self.cum = cumulative_m(self.points)
        for s in self.steps:
            if 0 <= s.start_index < len(self.cum):
                s.along_m = self.cum[s.start_index]

    @property
    def length_m(self) -> float:
        return self.cum[-1] if self.cum else self.distance_m

    def risk_score(self) -> float:
        return self.length_m + STAIRS_PENALTY_M * self.stairs + sum(
            CROSSING_RISK_M[c.risk_class] for c in self.crossings)

    def crossing_summary(self) -> str:
        n = len(self.crossings)
        if n == 0:
            return "no street crossings on the map"
        sig = sum(1 for c in self.crossings if c.signals)
        if sig == n:
            return f"{n} crossing{'s' if n > 1 else ''}, all with signals" if n > 1 else "1 crossing, with signals"
        return f"{n} crossing{'s' if n > 1 else ''}, {sig} with signals"

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label, "source": self.source, "length_m": round(self.length_m),
            "duration_min": round(self.duration_s / 60.0, 1), "crossings": len(self.crossings),
            "signalised": sum(1 for c in self.crossings if c.signals), "stairs": self.stairs,
            "risk_score": round(self.risk_score()),
        }


def choose_safer_route(routes: list[Route], max_detour: float = 1.4) -> tuple[Route, str]:
    """Pick the lowest risk route within max_detour of the shortest.

    Returns (route, reason). The reason compares against the router's first choice.
    """
    if not routes:
        raise ValueError("no routes")
    default = routes[0]
    shortest = min(routes, key=lambda r: r.length_m)
    ok = [r for r in routes if r.length_m <= shortest.length_m * max_detour]
    best = min(ok, key=lambda r: (r.risk_score(), r.length_m))
    if best is default:
        return best, "the router's first route is also the most careful"

    def unsignalised(r: Route) -> int:
        return sum(1 for c in r.crossings if not c.signals)

    why = []
    if unsignalised(best) < unsignalised(default):
        why.append(f"{unsignalised(best)} crossings without signals instead of {unsignalised(default)}")
    if len(best.crossings) < len(default.crossings):
        why.append(f"{len(best.crossings)} crossings instead of {len(default.crossings)}")
    if best.stairs < default.stairs:
        why.append("no stairs" if best.stairs == 0 else "fewer stairs")
    delta = round(best.length_m - default.length_m)
    why.append(f"{abs(delta)} m {'longer' if delta > 0 else 'shorter'}")
    return best, ", ".join(why)
