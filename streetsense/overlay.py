"""Debug/demo overlay: boxes, light states, crossing stripes, surface tint, alert banner."""

from __future__ import annotations

import cv2
import numpy as np

from .crosswalk import CrosswalkResult
from .surface import PALETTE, SurfaceResult
from .types import Alert, Detection, Priority

COLORS = {
    "person": (255, 180, 0), "car": (0, 0, 255), "bus": (0, 0, 255), "truck": (0, 0, 255),
    "motorcycle": (0, 0, 255), "bicycle": (0, 120, 255), "traffic light": (0, 255, 255),
}
LIGHT_BGR = {"red": (0, 0, 255), "yellow": (0, 200, 255), "green": (0, 220, 0), "walk": (255, 255, 255), "unknown": (160, 160, 160)}
_LUT = np.zeros((256, 3), np.uint8)
_LUT_ON = np.zeros(256, bool)
for _c, _col in PALETTE.items():
    _LUT[_c] = _col
    _LUT_ON[_c] = True
PRIO_BGR = {Priority.CRITICAL: (0, 0, 255), Priority.WARNING: (0, 140, 255), Priority.NOTICE: (0, 220, 255), Priority.INFO: (220, 220, 220)}


def draw(frame: np.ndarray, dets: list[Detection], cw: CrosswalkResult | None, surf: SurfaceResult | None,
         recent_alerts: list[Alert], stats: str = "", vlm_text: str | None = None) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    if surf is not None:
        tint = _LUT[surf.labels]  # one gather instead of a boolean mask per class
        m = _LUT_ON[surf.labels]
        blended = cv2.addWeighted(out, 0.65, tint, 0.35, 0)
        out = np.where(m[..., None], blended, out)
    if cw is not None and cw.found:
        cv2.polylines(out, cw.stripes, True, (255, 255, 255), 2)
        b = cw.box
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (255, 255, 255), 2)
        cv2.putText(out, f"CROSSING ({cw.n_stripes} stripes)", (int(b.x1), max(20, int(b.y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for d in dets:
        b = d.box
        col = COLORS.get(d.label, (200, 200, 200))
        label = d.label
        if d.label == "traffic light":
            st = d.attrs.get("light_state", "unknown")
            col = LIGHT_BGR[st]
            label = f"{d.attrs.get('signal_kind', '?')[:3]} light: {st}"
        dist = d.attrs.get("distance_m")
        if dist is not None and d.label != "traffic light":
            label += f" ~{dist:.1f}m"
        if d.track_id is not None:
            label = f"#{d.track_id} " + label
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), col, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y = max(th + 4, int(b.y1))
        cv2.rectangle(out, (int(b.x1), y - th - 4), (int(b.x1) + tw + 4, y), col, -1)
        cv2.putText(out, label, (int(b.x1) + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    # Walking corridor guide.
    cv2.line(out, (int(W * 0.3), H - 1), (int(W * 0.45), int(H * 0.45)), (90, 90, 90), 1)
    cv2.line(out, (int(W * 0.7), H - 1), (int(W * 0.55), int(H * 0.45)), (90, 90, 90), 1)
    # Alert banner.
    y = 34
    for a in recent_alerts[:3]:
        txt = f"[{a.priority.name}] {a.text}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(out, (8, y - th - 8), (16 + tw, y + 6), (0, 0, 0), -1)
        cv2.putText(out, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, PRIO_BGR[a.priority], 2)
        y += th + 18
    if vlm_text:
        cv2.putText(out, f"Muse Spark: {vlm_text[:90]}", (12, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if surf is not None:
        stats = f"underfoot: {surf.underfoot} | " + stats
    cv2.putText(out, stats, (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out
