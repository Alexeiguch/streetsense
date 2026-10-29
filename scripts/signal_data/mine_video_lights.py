#!/usr/bin/env python3
"""Mine traffic-light boxes from videos with our detector (YOLO full frame + zoom-out + 2x3 tiles).

For each clip in a clips CSV (columns id,file) sample `--fps` frames, detect lights, merge by NMS,
write one row per box: clip,t,x1,y1,x2,y2,conf,how(full|zoom|tile). Crops are saved to
<out_dir>/crops/<clip>_<t>_<i>.jpg (outside git) for contact-sheet review.
  python scripts/signal_data/mine_video_lights.py --clips /workspace/datasets/commons_video/clips.csv
"""
import argparse, csv, sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from streetsense.config import DetectorConfig  # noqa: E402
from streetsense.detector import YoloDetector, nms_lights  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base = Path(a.clips).parent
    out = Path(a.out or base / "boxes.csv")
    (base / "crops").mkdir(exist_ok=True)
    det = YoloDetector(DetectorConfig(track=False, conf=a.conf, imgsz=960))
    rows = []
    for c in csv.DictReader(open(a.clips)):
        cap = cv2.VideoCapture(str(base / c["file"]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, round(fps / a.fps)); k = 0; got = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if k % step == 0:
                t = round(k / fps, 2)
                d = [x for x in det(f) if x.label == "traffic light"]
                d += det.zoomed_lights(f, (0.5,), a.conf) if not d else []
                d += det.tiled_lights(f, (2, 3), max(a.conf, 0.4))
                for i, x in enumerate(nms_lights(d)):
                    b = x.box
                    if b.w < 6 or b.h < 6:
                        continue
                    how = "tile" if "tile" in (x.attrs or {}) else "zoom" if "zoom" in (x.attrs or {}) else "full"
                    rows.append(dict(clip=c["id"], t=t, x1=int(b.x1), y1=int(b.y1), x2=int(b.x2), y2=int(b.y2),
                                     conf=round(x.conf, 3), how=how, name=f"{c['id']}_{t:07.2f}_{i}"))
                    cv2.imwrite(str(base / "crops" / f"{c['id']}_{t:07.2f}_{i}.jpg"),
                                f[int(b.y1):int(b.y2), int(b.x1):int(b.x2)])
                    got += 1
            k += 1
        print(c["id"], "frames", n, "boxes", got, flush=True)
    w = csv.DictWriter(open(out, "w"), fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
