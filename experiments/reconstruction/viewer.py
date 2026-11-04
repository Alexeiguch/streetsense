"""Self-contained point-cloud viewer; no server, CDN, or model needed to open it."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np


def write_viewer(path: Path, data, report):
    points, colors = data["points"][data["mask"]], data["rgb"][data["mask"]]
    if not len(points):
        raise ValueError("Cannot display an empty reconstruction")
    # Display-only clipping of extreme tails; evaluations use every valid point.
    center = np.median(points, axis=0)
    distance = np.linalg.norm(points-center, axis=1)
    keep = distance <= np.percentile(distance, 98)
    points, colors = points[keep], colors[keep]
    ids = np.random.default_rng(42).choice(len(points), min(len(points), 24000), replace=False)
    scale = float(np.percentile(np.linalg.norm(points-center, axis=1), 75)) or 1.
    cloud = np.column_stack([(points[ids]-center)/scale, colors[ids]])
    cameras = (data["cameras"][:, :3, 3] - center) / scale
    images = []
    for rgb in data["rgb"]:
        _, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 78])
        images.append("data:image/jpeg;base64," + base64.b64encode(encoded).decode())
    payload = {"clouds": {"map": np.round(cloud, 4).tolist()}, "cameras": np.round(cameras, 4).tolist()}
    template = Path(__file__).with_name("viewer.html").read_text()
    checks = report["multiview_metrics"]
    matches = sum(m["directional_checks"] for m in checks)
    agreement = sum(m["directional_checks"]*m["within_3px_fraction"] for m in checks) / matches if matches else 0
    comparison = ""
    if "depth_baseline" in report and (path.parent / "baseline.npz").exists():
        baseline = np.load(path.parent / "baseline.npz")["points"][data["mask"]][keep][ids]
        good = np.isfinite(baseline).all(-1)
        payload["clouds"]["depth"] = np.round(np.column_stack([(baseline[good]-center)/scale, colors[ids][good]]), 4).tolist()
        metrics = report["depth_baseline"]["metrics"]
        n = sum(m["directional_checks"] for m in metrics)
        ratio = sum(m["directional_checks"]*m["within_3px_fraction"] for m in metrics) / n if n else 0
        comparison = ('<label>Geometry <select id="geometry"><option value="map">MapAnything multi-view</option>'
                      '<option value="depth">Existing relative depth</option></select></label>')
        payload["baseline_agreement"] = f'{ratio:.0%} of {n:,} checks within 3 px · cameras and per-view scale borrowed from MapAnything'
    labels = [f'{r["decoded_s"]:.2f} s' for r in report["frames"]]
    source_images = "".join(f'<figure><img src="{src}" alt="Source view {i+1} at {labels[i]}"><figcaption>{labels[i]}</figcaption></figure>' for i, src in enumerate(images))
    replacements = {"PAYLOAD": json.dumps(payload, separators=(",", ":")),
                    "IMAGES": source_images, "SOURCE": html.escape(Path(report["source"]).name),
                    "STATS": f'{len(images)} views · {len(ids):,} displayed points · MapAnything {report["performance"]["inference_s"]:.1f} s/batch',
                    "AGREEMENT": f'{agreement:.0%} of {matches:,} correspondence checks within 3 px',
                    "COMPARISON": comparison,
                    "GATE": html.escape(report["gate"]["status"] + (": " + ", ".join(r.replace("_", " ") for r in report["gate"]["reasons"]) if report["gate"]["reasons"] else ""))}
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    path.write_text(template)
