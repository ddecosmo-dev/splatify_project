"""Convert reconstruction outputs into comparable 3D point samples.

Adapters preserve the native reconstruction representation. Meshes are sampled
on their surfaces, point clouds retain their points, and 3D Gaussian Splatting
outputs retain Gaussian centers whose activated opacity passes a configured
threshold. No adapter attempts to reconstruct a mesh from a point-based output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh


GeometryKind = Literal["auto", "mesh", "point_cloud", "gaussian_splat"]


@dataclass(frozen=True)
class GeometrySample:
    """A deterministic point sample produced from one reconstruction output."""

    points: np.ndarray
    kind: str
    source_path: Path
    metadata: dict[str, int | float | str] = field(default_factory=dict)


def _validate_sample_count(sample_count: int | None) -> None:
    if sample_count is not None and (
        not isinstance(sample_count, int) or sample_count < 1
    ):
        raise ValueError("sample_count must be a positive integer or None")


def _clean_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError("Geometry contains no finite 3D points")
    return points


def _subsample(
    points: np.ndarray,
    sample_count: int | None,
    seed: int,
) -> np.ndarray:
    if sample_count is None or len(points) <= sample_count:
        return points.copy()

    generator = np.random.default_rng(seed)
    selected = generator.choice(len(points), sample_count, replace=False)
    return points[selected]


def _vertex_data(geometry: trimesh.Trimesh | trimesh.points.PointCloud):
    raw = geometry.metadata.get("_ply_raw")
    if not raw or "vertex" not in raw:
        return None
    return raw["vertex"].get("data")


def _vertex_fields(vertex_data) -> set[str]:
    if isinstance(vertex_data, dict):
        return set(vertex_data)
    if vertex_data.dtype.names is None:
        return set()
    return set(vertex_data.dtype.names)


def _vertex_column(vertex_data, name: str) -> np.ndarray:
    return np.asarray(vertex_data[name]).reshape(-1)


def _vertex_count(vertex_data) -> int:
    fields = _vertex_fields(vertex_data)
    if not fields:
        return 0
    return len(_vertex_column(vertex_data, next(iter(fields))))


def _is_gaussian_splat(
    geometry: trimesh.Trimesh | trimesh.points.PointCloud,
) -> bool:
    vertex_data = _vertex_data(geometry)
    if vertex_data is None:
        return False

    fields = _vertex_fields(vertex_data)
    required = {
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    return required <= fields


def _activated_opacity(logits: np.ndarray) -> np.ndarray:
    # Vanilla 3DGS stores pre-sigmoid opacity values in its PLY output.
    logits = np.clip(np.asarray(logits, dtype=np.float64), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _adapt_mesh(
    geometry: trimesh.Trimesh,
    path: Path,
    sample_count: int | None,
    seed: int,
) -> GeometrySample:
    if len(geometry.faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")
    if sample_count is None:
        raise ValueError("Mesh adaptation requires a finite sample_count")

    points, _ = trimesh.sample.sample_surface(
        geometry,
        count=sample_count,
        seed=seed,
    )
    points = _clean_points(points)
    return GeometrySample(
        points=points,
        kind="mesh",
        source_path=path,
        metadata={
            "vertices": len(geometry.vertices),
            "faces": len(geometry.faces),
            "sampled_points": len(points),
        },
    )


def _adapt_point_cloud(
    geometry: trimesh.Trimesh | trimesh.points.PointCloud,
    path: Path,
    sample_count: int | None,
    seed: int,
) -> GeometrySample:
    original_count = len(geometry.vertices)
    points = _clean_points(geometry.vertices)
    finite_count = len(points)
    points = _subsample(points, sample_count, seed)
    return GeometrySample(
        points=points,
        kind="point_cloud",
        source_path=path,
        metadata={
            "input_points": original_count,
            "finite_points": finite_count,
            "sampled_points": len(points),
        },
    )


def _adapt_gaussian_splat(
    geometry: trimesh.Trimesh | trimesh.points.PointCloud,
    path: Path,
    sample_count: int | None,
    seed: int,
    min_opacity: float,
) -> GeometrySample:
    if not 0.0 <= min_opacity <= 1.0:
        raise ValueError("min_opacity must be between 0 and 1")

    vertex_data = _vertex_data(geometry)
    if vertex_data is None or not _is_gaussian_splat(geometry):
        raise ValueError(f"Not a recognized 3DGS PLY: {path}")

    points = np.column_stack(
        (
            _vertex_column(vertex_data, "x"),
            _vertex_column(vertex_data, "y"),
            _vertex_column(vertex_data, "z"),
        )
    )
    opacity = _activated_opacity(_vertex_column(vertex_data, "opacity"))
    valid = np.isfinite(points).all(axis=1) & np.isfinite(opacity)
    visible = valid & (opacity >= min_opacity)
    points = _clean_points(points[visible])
    visible_count = len(points)
    points = _subsample(points, sample_count, seed)

    return GeometrySample(
        points=points,
        kind="gaussian_splat",
        source_path=path,
        metadata={
            "input_gaussians": _vertex_count(vertex_data),
            "visible_gaussians": visible_count,
            "sampled_points": len(points),
            "min_opacity": min_opacity,
        },
    )


def adapt_geometry(
    path: str | Path,
    kind: GeometryKind = "auto",
    *,
    sample_count: int | None = 100_000,
    seed: int = 42,
    min_opacity: float = 0.1,
) -> GeometrySample:
    """Load a mesh, point cloud, or 3DGS PLY as a point sample.

    Args:
        path: Input geometry path supported by ``trimesh``.
        kind: Explicit input type or ``"auto"`` to inspect the loaded file.
        sample_count: Surface sample count for meshes and maximum retained count
            for point-based outputs. ``None`` retains every point but is invalid
            for meshes, whose continuous surfaces require a finite count.
        seed: Random seed used for surface sampling or point subsampling.
        min_opacity: Minimum activated opacity for Gaussian centers. Vanilla
            3DGS stores opacity logits, so the adapter applies a sigmoid first.

    Returns:
        GeometrySample: Finite 3D points and adaptation metadata.
    """
    _validate_sample_count(sample_count)
    if kind not in ("auto", "mesh", "point_cloud", "gaussian_splat"):
        raise ValueError(f"Unsupported geometry kind: {kind}")

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    geometry = trimesh.load(source_path, process=False)
    if isinstance(geometry, trimesh.Scene):
        geometry = geometry.to_mesh()
    if not isinstance(geometry, (trimesh.Trimesh, trimesh.points.PointCloud)):
        raise TypeError(f"Unsupported loaded geometry type: {type(geometry)}")

    detected_kind = kind
    if detected_kind == "auto":
        if _is_gaussian_splat(geometry):
            detected_kind = "gaussian_splat"
        elif isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0:
            detected_kind = "mesh"
        else:
            detected_kind = "point_cloud"

    if detected_kind == "mesh":
        if not isinstance(geometry, trimesh.Trimesh):
            raise ValueError(f"Requested mesh input has no faces: {source_path}")
        return _adapt_mesh(geometry, source_path, sample_count, seed)
    if detected_kind == "gaussian_splat":
        return _adapt_gaussian_splat(
            geometry,
            source_path,
            sample_count,
            seed,
            min_opacity,
        )
    return _adapt_point_cloud(geometry, source_path, sample_count, seed)
