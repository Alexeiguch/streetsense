#!/usr/bin/env python3
"""Score SAM / SAM+Muse boxes against ground truth on frames that have it (PTLD train/valid, commons)."""
import collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sam_muse_label import iou  # noqa: E402


def score(path, thr=0.5):
    R = [json.loads(l) for l in open(path)]
    R = [r for r in R if r.get("gt")]
    c = collections.Counter(); ious = []
    for r in R:
        gt = [g[:4] for g in r["gt"]]; c["frames"] += 1; c["gt"] += len(gt)
        sam = [b["box"] for b in r["boxes"]]; kept = [b["box"] for b in r["boxes"] if b["muse"] == "yes"]
        c["sam_boxes"] += len(sam); c["kept_boxes"] += len(kept)
        c["muse_unclear"] += sum(b["muse"] == "unclear" for b in r["boxes"])
        for g in gt:
            best = max([iou(g, s) for s in sam] or [0]); ious.append(best)
            c["gt_hit_sam"] += best >= thr; c["gt_hit_sam_03"] += best >= 0.3
            c["gt_hit_kept"] += max([iou(g, s) for s in kept] or [0]) >= thr
        for s in sam:
            m = max([iou(s, g) for g in gt] or [0]) >= thr; c["sam_match"] += m
        for b in r["boxes"]:
            m = max([iou(b["box"], g) for g in gt] or [0]) >= thr
            c[("muse_yes" if b["muse"] == "yes" else "muse_no") + ("_on_gt" if m else "_off_gt")] += 1
    out = dict(c)
    f = lambda a, b: round(out.get(a, 0) / out[b], 3) if out.get(b) else None
    out.update(sam_recall=f("gt_hit_sam", "gt"), sam_recall_iou03=f("gt_hit_sam_03", "gt"), sam_precision=f("sam_match", "sam_boxes"),
               kept_recall=f("gt_hit_kept", "gt"),
               kept_precision=round(out.get("muse_yes_on_gt", 0) / max(1, out.get("kept_boxes", 0)), 3),
               muse_agree_with_gt=round((out.get("muse_yes_on_gt", 0) + out.get("muse_no_off_gt", 0)) / max(1, out.get("sam_boxes", 0)), 3),
               median_best_iou=round(sorted(ious)[len(ious) // 2], 3) if ious else None)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, json.dumps(score(p)))
