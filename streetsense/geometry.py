"""Geometry helpers: where is an object relative to the walker, and roughly how far."""

from __future__ import annotations

import math

from .types import Box

# Typical real-world heights in metres, used for a pinhole-camera distance estimate.
# These are coarse priors: the estimate is "about 2 metres", not a measurement.
REAL_HEIGHT_M = {
    "person": 1.7,
    "bicycle": 1.1,
    "motorcycle": 1.2,
    "car": 1.5,
    "bus": 3.2,
    "truck": 3.0,
    "traffic light": 0.9,
    "stop sign": 0.75,
    "fire hydrant": 0.75,
    "bench": 0.9,
    "dog": 0.6,
    "chair": 0.9,
    "potted plant": 0.8,
    "parking meter": 1.3,
    # open-vocab sidewalk hazards (coarse priors)
    "bollard": 0.9,
    "traffic cone": 0.7,
    "scooter": 1.1,
    "construction barrier": 1.2,
    "barrier": 1.2,
    "stairs": 1.5,
    "step": 0.2,
    "pothole": 0.15,
    "manhole": 0.1,
    "kerb": 0.15,
}


def focal_length_px(image_width: int, hfov_deg: float) -> float:
    """Focal length in pixels for a camera with the given horizontal field of view."""
    if image_width <= 0 or not 0 < hfov_deg < 180:
        raise ValueError("invalid image width or field of view")
    return (image_width / 2) / math.tan(math.radians(hfov_deg) / 2)


def truncation_flags(box: Box, image_width: int, image_height: int, margin: float = 1.0) -> dict:
    """Which frame edges the box touches. Single-edge clips make pinhole distance unreliable."""
    top = box.y1 <= margin
    bottom = box.y2 >= image_height - margin
    left = box.x1 <= margin
    right = box.x2 >= image_width - margin
    both_vertical = top and bottom
    any_vertical = top or bottom
    uncertain = (any_vertical and not both_vertical) or left or right
    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "both_vertical": both_vertical,
        "uncertain": uncertain,
    }


def estimate_distance_m(
    box: Box, label: str, image_width: int, image_height: int, hfov_deg: float = 70.0
) -> float | None:
    """Distance from apparent height: d = f * H_real / h_px.

    Returns None for unknown classes. Full vertical fill → 0.5 m. Single-edge (top XOR bottom)
    truncation returns None so callers use a conservative proximity fallback instead of a
    misleading far estimate from the visible partial height.
    """
    real_h = REAL_HEIGHT_M.get(label)
    if real_h is None or box.h < 2:
        return None
    flags = truncation_flags(box, image_width, image_height)
    if flags["both_vertical"]:
        return 0.5  # fills the whole frame vertically: right in front
    if flags["top"] or flags["bottom"]:
        return None  # partial height → do not claim a precise distance
    f = focal_length_px(image_width, hfov_deg)
    return f * real_h / box.h


def horizontal_zone(box: Box, image_width: int, center_frac: float = 0.34) -> str:
    """'left' | 'ahead' | 'right' from the box centre. The middle band is `center_frac` wide."""
    x = box.cx / image_width
    half = center_frac / 2
    if x < 0.5 - half:
        return "left"
    if x > 0.5 + half:
        return "right"
    return "ahead"


def clock_direction(box: Box, image_width: int, hfov_deg: float = 70.0) -> int:
    """Clock-face direction (12 = straight ahead), a convention many blind travellers use."""
    angle = (box.cx / image_width - 0.5) * hfov_deg  # degrees, negative = left
    hour = round(angle / 30)  # 30 degrees per clock hour
    return 12 if hour == 0 else (12 + hour - 1) % 12 + 1


def in_walking_path(box: Box, image_width: int, image_height: int, corridor_frac: float = 0.4) -> bool:
    """True when the object's footprint overlaps the central corridor in the lower part of the frame."""
    left = image_width * (0.5 - corridor_frac / 2)
    right = image_width * (0.5 + corridor_frac / 2)
    overlaps = box.x2 > left and box.x1 < right
    return overlaps and box.y2 > image_height * 0.45


def round_distance(d: float) -> str:
    """Speakable distance: '1 metre', '3 metres', 'about 10 metres'."""
    if d < 1.5:
        return "1 metre"
    if d < 6:
        return f"{round(d)} metres"
    return f"about {int(round(d / 5) * 5)} metres"


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
