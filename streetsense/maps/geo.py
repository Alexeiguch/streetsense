"""Small, dependency-free geodesy helpers for walking-scale distances.

Equirectangular maths is accurate to well under a metre over a few hundred
metres, which is far below phone GPS error (5 to 15 m in cities).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

EARTH_R = 6_371_008.8
LatLon = tuple[float, float]


def haversine_m(a: LatLon, b: LatLon) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_R * math.asin(min(1.0, math.sqrt(h)))


def bearing_deg(a: LatLon, b: LatLon) -> float:
    """Initial compass bearing from a to b, 0 = north, clockwise."""
    la1, la2 = math.radians(a[0]), math.radians(b[0])
    dlo = math.radians(b[1] - a[1])
    y = math.sin(dlo) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlo)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def destination(a: LatLon, bearing: float, dist_m: float) -> LatLon:
    la1, lo1 = math.radians(a[0]), math.radians(a[1])
    br, d = math.radians(bearing), dist_m / EARTH_R
    la2 = math.asin(math.sin(la1) * math.cos(d) + math.cos(la1) * math.sin(d) * math.cos(br))
    lo2 = lo1 + math.atan2(math.sin(br) * math.sin(d) * math.cos(la1), math.cos(d) - math.sin(la1) * math.sin(la2))
    return (math.degrees(la2), (math.degrees(lo2) + 540.0) % 360.0 - 180.0)


def relative_bearing(target_bearing: float, heading: float) -> float:
    """Signed angle in (-180, 180]: negative is left, positive is right."""
    r = (target_bearing - heading + 540.0) % 360.0 - 180.0
    return 180.0 if r == -180.0 else r


def clock_face(rel_deg: float) -> int:
    """Relative bearing to a clock hour: 12 straight ahead, 3 right, 9 left."""
    h = int(round(((rel_deg % 360.0) + 360.0) % 360.0 / 30.0)) % 12
    return 12 if h == 0 else h


def side_word(rel_deg: float) -> str:
    a = abs(rel_deg)
    if a <= 20:
        return "ahead"
    if a >= 150:
        return "behind you"
    side = "right" if rel_deg > 0 else "left"
    return f"on your {side}" if a >= 60 else f"ahead on your {side}"


def decode_polyline(encoded: str, precision: int = 5) -> list[LatLon]:
    """Google/OSRM use precision 5, Valhalla uses 6."""
    coords: list[LatLon] = []
    index = lat = lng = 0
    factor = 10.0 ** precision
    while index < len(encoded):
        for which in (0, 1):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if which == 0:
                lat += delta
            else:
                lng += delta
        coords.append((lat / factor, lng / factor))
    return coords


def encode_polyline(points: Iterable[LatLon], precision: int = 5) -> str:
    factor = 10 ** precision
    out = []
    plat = plng = 0
    for la, lo in points:
        ila, ilo = int(round(la * factor)), int(round(lo * factor))
        for d in (ila - plat, ilo - plng):
            v = ~(d << 1) if d < 0 else d << 1
            while v >= 0x20:
                out.append(chr((0x20 | (v & 0x1F)) + 63))
                v >>= 5
            out.append(chr(v + 63))
        plat, plng = ila, ilo
    return "".join(out)


def _local_xy(origin: LatLon, p: LatLon) -> tuple[float, float]:
    k = math.cos(math.radians(origin[0]))
    return (math.radians(p[1] - origin[1]) * EARTH_R * k, math.radians(p[0] - origin[0]) * EARTH_R)


def cumulative_m(line: Sequence[LatLon]) -> list[float]:
    out = [0.0]
    for i in range(1, len(line)):
        out.append(out[-1] + haversine_m(line[i - 1], line[i]))
    return out


def project_onto_line(p: LatLon, line: Sequence[LatLon], cum: Sequence[float] | None = None,
                      start_seg: int = 0, end_seg: int | None = None) -> tuple[float, float, int, LatLon]:
    """Closest point of a polyline to p.

    Returns (lateral_m, along_m, segment_index, snapped_point).
    """
    if len(line) < 2:
        q = line[0]
        return haversine_m(p, q), 0.0, 0, q
    cum = cum if cum is not None else cumulative_m(line)
    end = len(line) - 1 if end_seg is None else min(end_seg, len(line) - 1)
    best = (float("inf"), 0.0, start_seg, line[start_seg])
    for i in range(max(0, start_seg), end):
        a, b = line[i], line[i + 1]
        bx, by = _local_xy(a, b)
        px, py = _local_xy(a, p)
        seg2 = bx * bx + by * by
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, (px * bx + py * by) / seg2))
        dx, dy = px - t * bx, py - t * by
        d = math.hypot(dx, dy)
        if d < best[0]:
            q = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            best = (d, cum[i] + t * (cum[i + 1] - cum[i]), i, q)
    return best


def point_at(line: Sequence[LatLon], cum: Sequence[float], along_m: float) -> tuple[LatLon, float]:
    """Point at a distance along the line and the local bearing there."""
    along_m = max(0.0, min(cum[-1], along_m))
    for i in range(len(line) - 1):
        if cum[i + 1] >= along_m:
            seg = cum[i + 1] - cum[i]
            t = 0.0 if seg == 0 else (along_m - cum[i]) / seg
            a, b = line[i], line[i + 1]
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), bearing_deg(a, b)
    return line[-1], bearing_deg(line[-2], line[-1]) if len(line) > 1 else 0.0


def round_m(d: float) -> int:
    """Spoken distances: 5 m steps under 50 m, 10 m steps above. GPS is not better than that."""
    if d < 50:
        return max(5, int(round(d / 5.0)) * 5)
    return int(round(d / 10.0)) * 10
