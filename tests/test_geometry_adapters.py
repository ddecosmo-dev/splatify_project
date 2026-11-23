"""Tests for reconstruction-output point adapters."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from utils.evaluation.adapters import adapt_geometry


def test_mesh_adapter_samples_surface_deterministically(tmp_path):
    path = tmp_path / "box.ply"
    trimesh.creation.box().export(path)

    first = adapt_geometry(path, sample_count=200, seed=7)
    second = adapt_geometry(path, sample_count=200, seed=7)

    assert first.kind == "mesh"
    assert first.points.shape == (200, 3)
    assert np.array_equal(first.points, second.points)
    assert first.metadata["faces"] == 12


def test_point_cloud_adapter_retains_native_points(tmp_path):
    path = tmp_path / "cloud.ply"
    points = np.arange(90, dtype=float).reshape(30, 3)
    trimesh.PointCloud(points).export(path)

    sample = adapt_geometry(path, sample_count=10, seed=3)

    assert sample.kind == "point_cloud"
    assert sample.points.shape == (10, 3)
    assert sample.metadata["input_points"] == 30
    assert sample.metadata["sampled_points"] == 10


def _write_gaussian_ply(path: Path) -> None:
    path.write_text(
        """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
0 0 0 -4 0 0 0 1 0 0 0
1 0 0 0 0 0 0 1 0 0 0
2 0 0 4 0 0 0 1 0 0 0
""",
        encoding="ascii",
    )


def test_gaussian_adapter_activates_and_filters_opacity(tmp_path):
    path = tmp_path / "gaussians.ply"
    _write_gaussian_ply(path)

    sample = adapt_geometry(path, min_opacity=0.5)

    assert sample.kind == "gaussian_splat"
    assert sample.points.shape == (2, 3)
    assert set(sample.points[:, 0]) == {1.0, 2.0}
    assert sample.metadata["input_gaussians"] == 3
    assert sample.metadata["visible_gaussians"] == 2


def test_explicit_wrong_kind_fails_clearly(tmp_path):
    path = tmp_path / "cloud.ply"
    trimesh.PointCloud(np.eye(3)).export(path)

    with pytest.raises(ValueError, match="no faces"):
        adapt_geometry(path, kind="mesh")


@pytest.mark.parametrize("sample_count", [0, -1, 1.5])
def test_invalid_sample_count(sample_count, tmp_path):
    with pytest.raises(ValueError, match="sample_count"):
        adapt_geometry(tmp_path / "missing.ply", sample_count=sample_count)
