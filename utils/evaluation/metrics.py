"""Scale and geometric accuracy metrics for standardized point samples."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class AlignmentResult:
    """Prediction points after similarity alignment to a reference sample."""

    points: np.ndarray
    scale_correction: float
    scale_error_percent: float
    rotation: np.ndarray
    translation: np.ndarray
    prediction_center: np.ndarray
    alignment_cost: float


@dataclass(frozen=True)
class GeometryMetrics:
    """Directional and symmetric point-to-surface approximation metrics."""

    scale_correction: float
    scale_error_percent: float
    accuracy: float
    completeness: float
    chamfer_l1: float
    normalized_accuracy: float
    normalized_completeness: float
    normalized_chamfer_l1: float
    reference_diagonal: float


def _points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
    if len(points) < 3:
        raise ValueError(f"{name} must contain at least three points")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values")
    return points


def _subsample(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    generator = np.random.default_rng(seed)
    selected = generator.choice(len(points), limit, replace=False)
    return points[selected]


def _center_and_radius(
    points: np.ndarray,
    radius_quantile: float,
) -> tuple[np.ndarray, float]:
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    radius = float(np.quantile(distances, radius_quantile))
    if not np.isfinite(radius) or radius <= np.finfo(float).eps:
        raise ValueError("Cannot estimate scale from degenerate geometry")
    return center, radius


def _principal_axes(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / len(points)
    _, axes = np.linalg.eigh(covariance)
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes


def _proper_axis_mappings() -> list[np.ndarray]:
    mappings = []
    for order in permutations(range(3)):
        permutation = np.eye(3)[:, order]
        for signs in product((-1.0, 1.0), repeat=3):
            mapping = permutation @ np.diag(signs)
            if np.linalg.det(mapping) > 0:
                mappings.append(mapping)
    return mappings


def _rigid_fit(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1] *= -1
        rotation = right_transpose.T @ left.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _trimmed_mean(values: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(len(values) * fraction)))
    return float(np.mean(np.partition(values, count - 1)[:count]))


def _refine_rigid(
    source: np.ndarray,
    reference: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    iterations: int,
    trim_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_tree = cKDTree(reference)
    previous_cost = np.inf

    for _ in range(iterations):
        transformed = source @ rotation.T + translation
        distances, indices = reference_tree.query(transformed, workers=-1)
        keep_count = max(3, int(np.ceil(len(distances) * trim_fraction)))
        keep = np.argpartition(distances, keep_count - 1)[:keep_count]
        delta_rotation, delta_translation = _rigid_fit(
            transformed[keep],
            reference[indices[keep]],
        )
        rotation = delta_rotation @ rotation
        translation = translation @ delta_rotation.T + delta_translation

        cost = float(np.mean(distances[keep]))
        if np.isfinite(previous_cost) and (
            abs(previous_cost - cost)
            <= 1e-8 * max(1.0, previous_cost)
        ):
            break
        previous_cost = cost

    transformed = source @ rotation.T + translation
    return transformed, rotation, translation


def _symmetric_cost(
    prediction: np.ndarray,
    reference: np.ndarray,
    trim_fraction: float,
) -> float:
    prediction_distances = cKDTree(reference).query(
        prediction, workers=-1
    )[0]
    reference_distances = cKDTree(prediction).query(
        reference, workers=-1
    )[0]
    return 0.5 * (
        _trimmed_mean(prediction_distances, trim_fraction)
        + _trimmed_mean(reference_distances, trim_fraction)
    )


def align_prediction(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    radius_quantile: float = 0.95,
    alignment_sample_count: int = 10_000,
    icp_iterations: int = 30,
    trim_fraction: float = 0.9,
    seed: int = 42,
) -> AlignmentResult:
    """Estimate scale, then rigidly align a prediction to its reference.

    The reported scale correction is the multiplier applied to the prediction.
    A value of one is perfect; two means the prediction needed to be doubled.
    The symmetric percentage error makes factors of two and one-half equally
    wrong. PCA axis proposals followed by trimmed ICP handle unknown coordinate
    frames while reducing the influence of reconstruction outliers.
    """
    prediction = _points(prediction, "prediction")
    reference = _points(reference, "reference")
    if not 0.5 <= radius_quantile < 1.0:
        raise ValueError("radius_quantile must be in [0.5, 1.0)")
    if not 0.5 < trim_fraction <= 1.0:
        raise ValueError("trim_fraction must be in (0.5, 1.0]")
    if alignment_sample_count < 3:
        raise ValueError("alignment_sample_count must be at least three")
    if icp_iterations < 1:
        raise ValueError("icp_iterations must be positive")

    prediction_center, prediction_radius = _center_and_radius(
        prediction, radius_quantile
    )
    reference_center, reference_radius = _center_and_radius(
        reference, radius_quantile
    )
    scale_correction = reference_radius / prediction_radius
    scale_error_percent = (
        np.exp(abs(np.log(scale_correction))) - 1.0
    ) * 100.0

    scaled = (prediction - prediction_center) * scale_correction
    source_sample = _subsample(scaled, alignment_sample_count, seed)
    reference_sample = _subsample(
        reference, alignment_sample_count, seed + 1
    )
    prediction_axes = _principal_axes(source_sample)
    reference_axes = _principal_axes(reference_sample)

    best_cost = np.inf
    best_rotation = np.eye(3)
    best_translation = reference_center.copy()
    for mapping in _proper_axis_mappings():
        initial_rotation = reference_axes @ mapping @ prediction_axes.T
        transformed, rotation, translation = _refine_rigid(
            source_sample,
            reference_sample,
            initial_rotation,
            reference_center.copy(),
            icp_iterations,
            trim_fraction,
        )
        cost = _symmetric_cost(
            transformed,
            reference_sample,
            trim_fraction,
        )
        if cost < best_cost:
            best_cost = cost
            best_rotation = rotation
            best_translation = translation

    aligned = scaled @ best_rotation.T + best_translation
    return AlignmentResult(
        points=aligned,
        scale_correction=float(scale_correction),
        scale_error_percent=float(scale_error_percent),
        rotation=best_rotation,
        translation=best_translation,
        prediction_center=prediction_center,
        alignment_cost=float(best_cost),
    )


def geometric_metrics(
    aligned_prediction: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    """Compute directional distances and their symmetric L1 Chamfer mean."""
    prediction = _points(aligned_prediction, "aligned_prediction")
    reference = _points(reference, "reference")

    accuracy = float(
        np.mean(cKDTree(reference).query(prediction, workers=-1)[0])
    )
    completeness = float(
        np.mean(cKDTree(prediction).query(reference, workers=-1)[0])
    )
    chamfer_l1 = 0.5 * (accuracy + completeness)
    reference_diagonal = float(
        np.linalg.norm(np.ptp(reference, axis=0))
    )
    if reference_diagonal <= np.finfo(float).eps:
        raise ValueError("Reference bounding box is degenerate")

    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "chamfer_l1": chamfer_l1,
        "normalized_accuracy": accuracy / reference_diagonal,
        "normalized_completeness": completeness / reference_diagonal,
        "normalized_chamfer_l1": chamfer_l1 / reference_diagonal,
        "reference_diagonal": reference_diagonal,
    }


def evaluate_geometry(
    prediction: np.ndarray,
    reference: np.ndarray,
    **alignment_options,
) -> GeometryMetrics:
    """Align prediction points and return the agreed output-only metrics."""
    alignment = align_prediction(
        prediction,
        reference,
        **alignment_options,
    )
    distances = geometric_metrics(alignment.points, reference)
    return GeometryMetrics(
        scale_correction=alignment.scale_correction,
        scale_error_percent=alignment.scale_error_percent,
        **distances,
    )
