"""Local signal diagnostics. Contains no model responses, credentials or camera images."""
from __future__ import annotations

from .rules import select_crossing_signal


def summarise_signals(out, width: int, height: int) -> dict:
    sel = select_crossing_signal(out.detections, width, height, out.crosswalk)
    heads = []
    for d in out.detections:
        if d.label != "traffic light":
            continue
        a = d.attrs
        heads.append({
            "track_id": d.track_id,
            "source": a.get("source") or ("tile" if "tile" in a else "zoom" if "zoom" in a else "local"),
            "box": [d.box.x1, d.box.y1, d.box.x2, d.box.y2],
            "confidence": float(d.conf),
            "selected": bool(sel is not None and sel.detection is d),
            "kind": a.get("signal_kind"),
            "distance": {k: v for k, v in a.items() if k.startswith("signal_distance_")
                         or k in ("signal_bearing_deg", "signal_band", "signal_depth_rel")},
            "heuristic": a.get("light_heuristic"),
            "cnn_probs": a.get("signal_cnn_probs"),
            "cnn_t_walk": a.get("signal_cnn_t_walk"),
            "cnn_t_veto": a.get("signal_cnn_t_veto"),
            "raw": a.get("light_raw"),
            "smoothed": a.get("light_smoothed"),
            "cnn_veto": a.get("light_cnn_veto", False),
            "walk_blocked": a.get("light_walk_blocked", False),
            "reset_on_veto": a.get("light_reset_on_veto", False),
            "state": a.get("light_state"),
        })
    return {
        "frame": out.index, "t": float(out.t),
        "shape": [height, width],
        "crosswalk": {
            "found": bool(out.crosswalk is not None and out.crosswalk.found),
            "box": ([out.crosswalk.box.x1, out.crosswalk.box.y1, out.crosswalk.box.x2, out.crosswalk.box.y2]
                    if out.crosswalk is not None and out.crosswalk.box is not None else None),
        },
        "local": sel.decision if sel is not None else "unknown",
        "reason": sel.reason if sel is not None else "no readable signal",
        "caution": sel.caution if sel is not None else None,
        "heads": heads,
    }
