#!/usr/bin/env python3
"""Compare smoother changes on identical recorded inputs. No detector, images or API calls.

Requires a trace from scripts/trace_signals.py. This replays local signal selection;
it does not measure detection/classifier accuracy or fused GO decisions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streetsense.crosswalk import CrosswalkResult
from streetsense.signal_cnn import CLASSES
from streetsense.signal_trace import summarise_signals
from streetsense.traffic_light import LightStateSmoother, cnn_hold_veto, cnn_rejects_walk
from streetsense.types import Box, Detection


def replay(records, *, reset_on_veto=False):
    smoother = LightStateSmoother(reset_walk_on_veto=reset_on_veto)
    focus_smoother = LightStateSmoother(reset_walk_on_veto=True)
    previous = None
    for rec in records:
        height, width = rec["shape"]
        cw = rec["crosswalk"]
        crosswalk = CrosswalkResult(cw["found"], Box(*cw["box"]) if cw["box"] else None, 0, 0.0, [])
        dets = []
        for head in rec["heads"]:
            probs = tuple(head["cnn_probs"][c] for c in CLASSES) if head["cnn_probs"] else None
            threshold = head.get("cnn_t_veto") or 0.8
            focused = head.get("focus_parent_box") is not None
            blocked = cnn_rejects_walk(probs, threshold) or (focused and head.get("focus_parent_veto", False))
            active = focus_smoother if focused else smoother
            box = Box(*head["box"])
            st = active.update(head["track_id"], head["raw"], box.clip(width, height),
                                 t=rec["t"], walk_blocked=blocked)
            final = cnn_hold_veto(st, head["raw"], probs, threshold)
            attrs = {
                "signal_kind": head["kind"], "light_raw": head["raw"], "light_state": final,
                "light_heuristic": head["heuristic"], "signal_cnn_probs": head["cnn_probs"],
                "signal_cnn_t_walk": head.get("cnn_t_walk"), "signal_cnn_t_veto": threshold,
                "light_smoothed": st, "light_cnn_veto": final != st,
                "light_walk_blocked": blocked, "light_reset_on_veto": active.reset_walk_on_veto,
                "focus_parent_box": head.get("focus_parent_box"),
                "focus_parent_veto": head.get("focus_parent_veto", False),
                "source": head["source"],
            }
            attrs.update(head.get("distance") or {})
            dets.append(Detection("traffic light", box, head["confidence"], head["track_id"], attrs))
        out = SimpleNamespace(index=rec["frame"], t=rec["t"], detections=dets, crosswalk=crosswalk)
        result = summarise_signals(out, width, height)
        result["transition"] = previous is not None and previous != result["local"]
        previous = result["local"]
        yield result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reset-on-veto", action="store_true")
    args = ap.parse_args()
    if args.trace.resolve() == args.out.resolve():
        ap.error("output must differ from the input trace")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = transitions = walk = 0
    with args.trace.open() as source, args.out.open("w") as dest:
        for rec in replay(map(json.loads, source), reset_on_veto=args.reset_on_veto):
            dest.write(json.dumps(rec) + "\n")
            n += 1
            transitions += rec["transition"]
            walk += rec["local"] == "walk"
    print(json.dumps({"frames": n, "transitions": transitions, "walk_frames": walk, "meta_calls": 0}))


if __name__ == "__main__":
    main()
