"""Turns a walking route plus noisy GPS fixes into short spoken cues.

Design rules:
  * Map data is context, never permission. No cue here says when to cross;
    the go rule stays local reader walk AND Muse Spark yes (answers.py).
  * GPS in cities is off by 5 to 15 m, worse near tall buildings. Thresholds
    scale with the reported accuracy, distances are rounded, and off-route
    needs several fixes in a row before we say anything.
  * Directions use clock faces relative to the walker's heading, which is
    what orientation and mobility trainers teach: 12 is straight ahead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

from .geo import LatLon, bearing_deg, clock_face, haversine_m, point_at, project_onto_line, relative_bearing, round_m, side_word
from .model import Crossing, Route, Step

TURN_KINDS = {"left", "right", "slight_left", "slight_right", "sharp_left", "sharp_right", "uturn"}
TURN_WORDS = {
    "left": "turn left", "right": "turn right", "slight_left": "bear left", "slight_right": "bear right",
    "sharp_left": "turn sharp left", "sharp_right": "turn sharp right", "uturn": "turn around",
}
SHORT_TURN = {
    "left": "Left", "right": "Right", "slight_left": "Bear left", "slight_right": "Bear right",
    "sharp_left": "Sharp left", "sharp_right": "Sharp right", "uturn": "Turn around",
}


@dataclass
class Fix:
    t: float
    lat: float
    lon: float
    heading: float | None = None  # compass degrees from the phone or glasses, None if unknown
    accuracy_m: float = 10.0

    @property
    def point(self) -> LatLon:
        return (self.lat, self.lon)


@dataclass
class Cue:
    t: float
    kind: str  # route_start, turn, landmark, crossing_preview, at_crossing, off_route, back_on_route, gps_weak, arrive, camera
    text: str
    short: str  # five words max, for the glasses
    speak: bool = True
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NavConfig:
    turn_far_m: float = 40.0
    turn_now_m: float = 10.0
    crossing_preview_m: float = 45.0
    at_crossing_m: float = 12.0
    arrive_m: float = 12.0
    off_route_m: float = 20.0
    off_route_fixes: int = 3
    back_on_route_m: float = 12.0
    gps_weak_m: float = 25.0


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def crossing_phrase(c: Crossing) -> str:
    feats = c.features()
    road = f"{c.road}, " if c.road else ""
    return road + (", ".join(feats) if feats else "no details on the map")


class Navigator:
    def __init__(self, route: Route, dest_name: str = "your destination", cfg: NavConfig | None = None,
                 reverse: Callable[[LatLon], dict] | None = None):
        self.route = route
        self.dest_name = dest_name
        self.cfg = cfg or NavConfig()
        self.reverse = reverse
        self.along = 0.0
        self.seg = 0
        self.lateral = 0.0
        self.heading: float | None = None
        self.last_fix: Fix | None = None
        self._prev_pts: list[LatLon] = []
        self._said: set[str] = set()
        self._off_count = 0
        self.off_route = False
        self.arrived = False
        self.at_crossing: Crossing | None = None
        self.passed_crossings = 0
        self._landmarks: dict[str, float] = {}

    # Helpers --------------------------------------------------------------
    def _speak(self, cue: Cue) -> Cue:
        cue.speak = cue.speak and self.route.speakable
        return cue

    def remaining_m(self) -> float:
        return max(0.0, self.route.length_m - self.along)

    def next_turn(self) -> Step | None:
        for s in self.route.steps:
            if s.kind in TURN_KINDS | {"stairs", "elevator", "escalator"} and s.along_m > self.along + 2:
                return s
        return None

    def next_crossing(self) -> Crossing | None:
        for c in self.route.crossings:
            if (c.along_m or 0) > self.along - 3:
                return c
        return None

    def current_street(self) -> str | None:
        name = None
        for s in self.route.steps:
            if s.along_m <= self.along + 1 and s.street:
                name = s.street
        return name

    def clock_to(self, p: LatLon) -> int | None:
        if self.last_fix is None or self.heading is None:
            return None
        return clock_face(relative_bearing(bearing_deg(self.last_fix.point, p), self.heading))

    def _update_heading(self, fix: Fix) -> None:
        if fix.heading is not None:
            self.heading = fix.heading % 360.0
            return
        # Course over ground from the last few fixes, needs real movement to beat GPS noise.
        self._prev_pts = (self._prev_pts + [fix.point])[-6:]
        if len(self._prev_pts) >= 3 and haversine_m(self._prev_pts[0], fix.point) > 6:
            self.heading = bearing_deg(self._prev_pts[0], fix.point)
        elif self.heading is None:
            self.heading = point_at(self.route.points, self.route.cum, self.along)[1]

    # Main entry ----------------------------------------------------------------
    def start(self, t: float = 0.0) -> Cue:
        r = self.route
        mins = max(1, round(r.duration_s / 60.0))
        first = next((s for s in r.steps if s.kind == "depart"), None)
        head = f" {first.instruction}" if first and first.instruction else ""
        text = (f"Route to {self.dest_name}: {round_m(r.length_m)} metres, about {_plural(mins, 'minute')}, "
                f"{r.crossing_summary()}.{head}")
        if r.stairs:
            text += f" {_plural(r.stairs, 'flight')} of stairs."
        return self._speak(Cue(t, "route_start", text, f"{round_m(r.length_m)} m, {len(r.crossings)} crossings",
                               data=r.summary()))

    def update(self, fix: Fix) -> list[Cue]:
        cfg, r = self.cfg, self.route
        cues: list[Cue] = []
        self.last_fix = fix
        if self.arrived:
            return cues
        # Search a window ahead of the last snapped segment so a noisy fix cannot jump backwards.
        lat_m, along, seg, _ = project_onto_line(fix.point, r.points, r.cum, max(0, self.seg - 3), self.seg + 40)
        self.lateral = lat_m
        if along >= self.along - 5:
            self.along, self.seg = max(self.along, along), seg
        self._update_heading(fix)

        if fix.accuracy_m > cfg.gps_weak_m and "gps_weak" not in self._said:
            self._said.add("gps_weak")
            cues.append(Cue(fix.t, "gps_weak", "GPS is weak here, so my directions may be off by a lot. Use your cane and the camera.",
                            "GPS weak"))

        off_m = max(cfg.off_route_m, 1.5 * fix.accuracy_m)
        if lat_m > off_m:
            self._off_count += 1
        else:
            self._off_count = 0
        if not self.off_route and self._off_count >= cfg.off_route_fixes:
            self.off_route = True
            _, _, _, snap = project_onto_line(fix.point, r.points, r.cum)
            clk = self.clock_to(snap)
            where = f" at {clk} o'clock" if clk else ""
            cues.append(self._speak(Cue(
                fix.t, "off_route",
                f"You may be off the route. It is about {round_m(lat_m)} metres away{where}. "
                f"GPS can be off by {round_m(max(5.0, fix.accuracy_m))} metres, so stop and check.",
                f"Off route, {clk or '?'} o'clock", data={"lateral_m": round(lat_m, 1)})))
            return cues
        if self.off_route:
            if lat_m <= cfg.back_on_route_m:
                self.off_route = False
                cues.append(self._speak(Cue(fix.t, "back_on_route", "Back on the route.", "Back on route")))
            else:
                return cues

        # Turns and stairs: one early cue and one at the turn. A turn that comes
        # after a crossing waits until the walker is across, so the order stays clear.
        blocking = self.next_crossing()
        for s in r.steps:
            ahead = s.along_m - self.along
            if s.kind not in TURN_KINDS | {"stairs", "elevator", "escalator"} or ahead < -3:
                continue
            if blocking is not None and (blocking.along_m or 0) <= s.along_m + 5:
                break
            onto = f" onto {s.street}" if s.street else ""
            if s.kind in TURN_KINDS:
                verb, short = TURN_WORDS[s.kind], SHORT_TURN[s.kind]
            else:
                verb, short = f"take the {s.kind}", s.kind.capitalize()
            key_far, key_now = f"far:{s.start_index}", f"now:{s.start_index}"
            if cfg.turn_now_m < ahead <= cfg.turn_far_m and key_far not in self._said and key_now not in self._said:
                self._said.add(key_far)
                cues.append(self._speak(Cue(fix.t, "turn", f"In {round_m(ahead)} metres, {verb}{onto}.",
                                            f"{short} in {round_m(ahead)} m", data={"step": s.instruction})))
            elif ahead <= cfg.turn_now_m and key_now not in self._said:
                self._said.add(key_now)
                cues.append(self._speak(Cue(fix.t, "turn", f"{verb.capitalize()} now{onto}.", f"{short} now",
                                            data={"step": s.instruction})))
            break  # only the nearest turn

        # Landmarks that help a cane user: bollards, gates.
        for s in r.steps:
            ahead = s.along_m - self.along
            if s.kind == "pass" and s.landmark and not s.landmark.lower().startswith("traffic signals") \
                    and 0 <= ahead <= 15 and f"lm:{s.start_index}" not in self._said:
                lm = s.landmark.lower().removeprefix("the ").strip()
                if any(w in lm for w in ("bollard", "gate", "barrier")):
                    self._said.add(f"lm:{s.start_index}")
                    recent = self._landmarks.get(lm)
                    if recent is None or s.along_m - recent > 30:
                        self._landmarks[lm] = s.along_m
                        cues.append(self._speak(Cue(fix.t, "landmark", f"{lm.capitalize()} ahead.", lm.capitalize()[:30])))

        # Crossing preview, then arrival at the kerb.
        c = self.next_crossing()
        if c is not None:
            ahead = (c.along_m or 0) - self.along
            key = f"x:{round(c.along_m or 0)}"
            clk = self.clock_to(c.point)
            at = f" at {clk} o'clock" if clk else ""
            if cfg.at_crossing_m < ahead <= cfg.crossing_preview_m and f"pre:{key}" not in self._said:
                self._said.add(f"pre:{key}")
                cues.append(self._speak(Cue(
                    fix.t, "crossing_preview",
                    f"Crossing in {round_m(ahead)} metres{at}: {crossing_phrase(c)}. Stop at the kerb and ask me to check.",
                    f"Crossing {round_m(ahead)} m", data=c.to_dict())))
            elif ahead <= cfg.at_crossing_m and f"at:{key}" not in self._said:
                self._said.add(f"at:{key}")
                self.at_crossing = c
                hint = "The map shows signals. " if c.signals else "The map shows no signals here. " if c.signals is False else ""
                if c.button:
                    hint += "Feel for the push button. "
                cues.append(self._speak(Cue(
                    fix.t, "at_crossing",
                    f"At the crossing{(' for ' + c.road) if c.road else ''}. {hint}Wait for my check before you step out.",
                    "At crossing, wait", data=c.to_dict())))
            if self.at_crossing is not None and (self.at_crossing.along_m or 0) < self.along - 8:
                self.at_crossing = None
                self.passed_crossings += 1

        if self.remaining_m() <= cfg.arrive_m and not self.arrived:
            self.arrived = True
            clk = self.clock_to(r.points[-1])
            d = haversine_m(fix.point, r.points[-1])
            where = f", about {round_m(d)} metres at {clk} o'clock" if clk else ""
            cues.append(self._speak(Cue(fix.t, "arrive", f"Arriving at {self.dest_name}{where}.",
                                        "Arriving")))
        return cues

    # Camera fusion -----------------------------------------------------------------
    def camera_hint(self, crosswalk: dict | None, local_signal: str | None, t: float = 0.0) -> Cue | None:
        """Combine the map's crossing direction with what the camera sees. Never a go."""
        c = self.at_crossing or self.next_crossing()
        if c is None or self.last_fix is None or self.heading is None:
            return None
        if haversine_m(self.last_fix.point, c.point) > 30:
            return None
        rel = relative_bearing(bearing_deg(self.last_fix.point, c.point), self.heading)
        map_zone = "left" if rel < -25 else "right" if rel > 25 else "ahead"
        clk = clock_face(rel)
        cw = crosswalk or {}
        lines = []
        if cw.get("found"):
            zone = cw.get("zone", "ahead")
            if zone == map_zone:
                lines.append(f"The crosswalk you need is {side_word(rel)}.")
            else:
                lines.append(f"I see a crosswalk {'ahead' if zone == 'ahead' else 'on your ' + zone}, "
                             f"but the map puts your crossing at {clk} o'clock. Check with your cane.")
        else:
            lines.append(f"The map puts your crossing at {clk} o'clock. I can't see the crosswalk yet.")
        if c.signals and (local_signal or "unknown") in {"unknown", "none"}:
            lines.append(f"The map shows a signal here but I can't see it yet. Turn slowly toward {clk} o'clock.")
        return self._speak(Cue(t, "camera", " ".join(lines), f"Crossing {clk} o'clock",
                               data={"map_zone": map_zone, "camera": cw, "clock": clk}))

    # Status for Wonder and "where am I" -----------------------------------------------
    def status(self) -> dict:
        nt, nc = self.next_turn(), self.next_crossing()
        return {
            "destination": self.dest_name,
            "remaining_m": round_m(self.remaining_m()),
            "off_route": self.off_route,
            "arrived": self.arrived,
            "gps_accuracy_m": self.last_fix.accuracy_m if self.last_fix else None,
            "street": self.current_street(),
            "next_turn": None if nt is None else {"in_m": round_m(nt.along_m - self.along), "instruction": nt.instruction},
            "next_crossing": None if nc is None else {
                "in_m": round_m(max(0.0, (nc.along_m or 0) - self.along)),
                "clock": self.clock_to(nc.point), "map_features": crossing_phrase(nc),
                "note": "map context only, the app checks the light itself"},
        }

    def where_am_i(self) -> str:
        st = self.status()
        street = st["street"]
        if not street and self.reverse and self.last_fix:
            try:
                street = (self.reverse(self.last_fix.point) or {}).get("road")
            except Exception:
                street = None
        near = None
        if not street:
            done = [c for c in self.route.crossings if c.road and (c.along_m or 0) <= self.along + 15]
            near = done[-1].road if done else None
        parts = [f"On {street}" if street else f"Near {near}" if near else "I don't know the street name here"]
        if self.heading is not None:
            parts[0] += f", heading {compass_word(self.heading)}"
        parts[0] += "."
        parts.append(f"{st['remaining_m']} metres to {self.dest_name}.")
        nc = st["next_crossing"]
        if nc:
            parts.append(f"Next crossing in {nc['in_m']} metres.")
        if self.last_fix and self.last_fix.accuracy_m > 15:
            parts.append(f"GPS is rough, about {round_m(self.last_fix.accuracy_m)} metres.")
        return " ".join(parts)


def compass_word(h: float) -> str:
    return ["north", "north east", "east", "south east", "south", "south west", "west", "north west"][int(((h % 360) + 22.5) // 45) % 8]
