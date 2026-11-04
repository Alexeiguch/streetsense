"""Geometry and isolation tests; no checkpoint download or accelerator required."""
import ast
from pathlib import Path

import numpy as np
import pytest

from experiments.reconstruction.geometry import (align_relative_depth, backproject,
    correspondence_metrics, feature_matches, project, quality_gate)


def scene():
    z = np.full((40, 60), 5., dtype=np.float32)
    k = np.array([[100., 0, 30], [0, 100., 20], [0, 0, 1]])
    c = np.eye(4)
    return z, k, c


def test_backproject_uses_z_not_radial_distance():
    z, k, c = scene()
    pts = backproject(z, k, c)
    assert pts[0, 0, 2] == 5
    assert np.linalg.norm(pts[0, 0]) > 5
    uv, depth = project(pts, k, c)
    y, x = np.mgrid[:40, :60]
    np.testing.assert_allclose(uv, np.stack([x, y], -1), atol=1e-6)
    np.testing.assert_allclose(depth, z)


def test_camera_pose_is_cam_to_world_with_rotation():
    z, k, c = scene()
    c[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    c[:3, 3] = [2, 3, 4]
    points = backproject(z, k, c)
    uv, depth = project(points, k, c)
    np.testing.assert_allclose(uv[20, 30], [30, 20])
    np.testing.assert_allclose(depth, z)


def test_symmetric_correspondences_with_known_camera_translation():
    z, k, c = scene()
    other = c.copy(); other[0, 3] = .25
    cameras = np.stack([c, other])
    points = np.stack([backproject(z, k, camera) for camera in cameras])
    masks = np.ones((2, 40, 60), dtype=bool)
    a = np.array([[20, 10], [40, 30]])
    b = a - [5, 0]
    result = correspondence_metrics(points, [k, k], cameras, masks, 0, 1, a, b)
    assert result["within_3px_fraction"] == 1
    assert result["median_reprojection_px"] < 1e-6
    assert result["median_ray_angle_deg"] > 1
    # A behind-camera point is a failed check, not silently dropped from denominator.
    points[0, 10, 20] = [0, 0, -5]
    result = correspondence_metrics(points, [k, k], cameras, masks, 0, 1, a, b)
    assert result["valid_fraction"] == .75
    assert result["within_3px_fraction"] == .75


def test_invalid_and_out_of_image_correspondences_are_failures():
    z, k, c = scene(); points = np.stack([backproject(z, k, c)]*2)
    result = correspondence_metrics(points, [k, k], np.stack([c, c]),
        np.zeros((2, 40, 60), dtype=bool), 0, 1,
        np.array([[10, 10], [-1, 2]]), np.array([[10, 10], [-1, 2]]))
    assert result["valid_fraction"] == 0
    assert result["within_3px_fraction"] == 0
    assert result["median_reprojection_px"] is None


def test_relative_alignment_is_inverse_and_ignores_bad_pixels():
    ref = np.array([[2., 4., 8., np.nan]])
    relative = np.array([[8., 4., 2., 0.]])
    aligned, scale = align_relative_depth(relative, ref, np.ones_like(ref, dtype=bool))
    assert scale == 16
    np.testing.assert_allclose(aligned[0, :3], ref[0, :3])
    assert np.isnan(aligned[0, 3])
    with pytest.raises(ValueError):
        align_relative_depth(relative, ref, np.zeros_like(ref, dtype=bool))


def test_static_duplicate_and_low_texture_sequences_rejected():
    row = {"correspondences": 100, "median_ray_angle_deg": 0., "within_3px_fraction": 1.}
    result = quality_gate([row], ["same", "same"])
    assert result["status"] == "reject"
    assert set(result["reasons"]) == {"duplicate_frames", "low_parallax"}
    blank = np.zeros((40, 60, 3), dtype=np.uint8)
    a, b = feature_matches(blank, blank)
    assert a.shape == b.shape == (0, 2)
    assert quality_gate([], ["a", "b"])["status"] == "reject"


def test_quality_gate_does_not_claim_metric_or_navigation_validation():
    result = quality_gate([{"correspondences": 50, "median_ray_angle_deg": 2.,
                            "within_3px_fraction": .9}], ["a", "b"])
    assert result["status"] == "inspect"
    assert not result["navigation_validated"]
    assert not result["metric_accuracy_validated"]


def test_production_never_imports_experiment():
    root = Path(__file__).resolve().parents[1]
    for source in (root / "streetsense").glob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not n.name.startswith("experiments") for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("experiments")
