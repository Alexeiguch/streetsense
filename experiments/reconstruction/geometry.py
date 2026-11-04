"""CPU-only geometry checks, independent of learned model imports."""
from __future__ import annotations

import cv2
import numpy as np


def backproject(depth: np.ndarray, intrinsics: np.ndarray, camera: np.ndarray) -> np.ndarray:
    """Camera Z depth -> OpenCV cam2world points (not distance along the ray)."""
    h, w = depth.shape
    y, x = np.mgrid[:h, :w]
    rays = np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(intrinsics).T
    local = rays * depth[..., None]
    return local @ camera[:3, :3].T + camera[:3, 3]


def project(points: np.ndarray, intrinsics: np.ndarray, camera: np.ndarray):
    local = (points - camera[:3, 3]) @ camera[:3, :3]
    z = local[..., 2]
    uvh = local @ intrinsics.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvh[..., :2] / uvh[..., 2:3]
    return uv, z


def feature_matches(a: np.ndarray, b: np.ndarray, mask_a=None, mask_b=None):
    """Mutual SIFT ratio matches; no model geometry used to select matches."""
    sift = cv2.SIFT_create(nfeatures=2500)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), mask_a)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), mask_b)
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    bf = cv2.BFMatcher()
    forward = {m.queryIdx: m.trainIdx for pair in bf.knnMatch(da, db, k=2)
               if len(pair) == 2 for m, n in [pair] if m.distance < .7 * n.distance}
    reverse = {m.queryIdx: m.trainIdx for pair in bf.knnMatch(db, da, k=2)
               if len(pair) == 2 for m, n in [pair] if m.distance < .7 * n.distance}
    pairs = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
    return (np.array([ka[i].pt for i, _ in pairs], dtype=float).reshape(-1, 2),
            np.array([kb[j].pt for _, j in pairs], dtype=float).reshape(-1, 2))


def correspondence_metrics(points, intrinsics, cameras, masks, a, b, uv_a, uv_b):
    """Symmetric reprojection checks. Missing/behind-camera points count as failures."""
    errors, angles = [], []
    for i, j, source_uv, target_uv in [(a, b, uv_a, uv_b), (b, a, uv_b, uv_a)]:
        xy = np.rint(source_uv).astype(int)
        target_xy = np.rint(target_uv).astype(int)
        h, w = masks[i].shape
        inside = ((xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
                  & (target_xy[:, 0] >= 0) & (target_xy[:, 0] < w)
                  & (target_xy[:, 1] >= 0) & (target_xy[:, 1] < h))
        err = np.full(len(xy), np.inf)
        ids = np.flatnonzero(inside)
        x, y = xy[ids].T
        tx, ty = target_xy[ids].T
        valid = masks[i][y, x] & masks[j][ty, tx] & np.isfinite(points[i][y, x]).all(axis=-1)
        ids = ids[valid]; x, y = xy[ids].T
        pred, z = project(points[i][y, x], intrinsics[j], cameras[j])
        finite = np.isfinite(pred).all(axis=-1) & (z > 0)
        err[ids[finite]] = np.linalg.norm(pred[finite] - target_uv[ids[finite]], axis=-1)
        errors.extend(err.tolist())
        if len(ids):
            ray_a = points[i][y, x] - cameras[i, :3, 3]
            ray_b = points[i][y, x] - cameras[j, :3, 3]
            denom = np.linalg.norm(ray_a, axis=-1) * np.linalg.norm(ray_b, axis=-1)
            cosine = np.sum(ray_a * ray_b, axis=-1) / np.maximum(denom, 1e-12)
            angles.extend(np.degrees(np.arccos(np.clip(cosine, -1, 1))).tolist())
    errors = np.asarray(errors)
    finite_errors = errors[np.isfinite(errors)]
    return {
        "correspondences": len(uv_a), "directional_checks": len(errors),
        "valid_fraction": float(len(finite_errors) / len(errors)) if len(errors) else 0.,
        "within_3px_fraction": float(np.mean(errors <= 3)) if len(errors) else 0.,
        "median_reprojection_px": float(np.median(finite_errors)) if len(finite_errors) else None,
        "p95_reprojection_px": float(np.percentile(finite_errors, 95)) if len(finite_errors) else None,
        "median_ray_angle_deg": float(np.median(angles)) if angles else None,
    }


def align_relative_depth(relative, reference_z, valid):
    """Optimistic per-view scale alignment; uses reconstruction, NOT ground truth."""
    good = valid & np.isfinite(relative) & (relative > 1e-8) & np.isfinite(reference_z) & (reference_z > 0)
    if not np.any(good):
        raise ValueError("No valid pixels for relative-depth alignment")
    scale = float(np.median(relative[good] * reference_z[good]))
    depth = np.full(relative.shape, np.nan, dtype=np.float32)
    depth[good] = scale / relative[good]
    return depth, scale


def quality_gate(metrics, frame_hashes):
    """Conservative demo eligibility only. Passing never authorizes navigation."""
    reasons = []
    if len(set(frame_hashes)) != len(frame_hashes):
        reasons.append("duplicate_frames")
    if not metrics or any(m["correspondences"] < 30 for m in metrics):
        reasons.append("too_few_matches")
    if any((m["median_ray_angle_deg"] or 0) < .5 for m in metrics):
        reasons.append("low_parallax")
    if any(m["within_3px_fraction"] < .5 for m in metrics):
        reasons.append("poor_view_consistency")
    return {"status": "inspect" if not reasons else "reject", "reasons": reasons,
            "navigation_validated": False, "metric_accuracy_validated": False}
