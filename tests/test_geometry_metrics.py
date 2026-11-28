"""Tests for output-only scale and geometry metrics."""

import numpy as np
import pytest

from utils.evaluation.metrics import (
    align_prediction,
    evaluate_geometry,
    geometric_metrics,
)


def _asymmetric_points(seed=8, count=2_000):
    generator = np.random.default_rng(seed)
    points = generator.normal(size=(count, 3))
    points[:, 0] *= 3.0
    points[:, 1] *= 1.5
    points[:, 2] += 0.2 * points[:, 0] ** 2
    return points


def test_alignment_recovers_scale_rotation_and_translation():
    reference = _asymmetric_points()
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    prediction = reference @ rotation.T * 0.4 + np.array([7.0, -3.0, 2.0])

    result = align_prediction(prediction, reference)
    distances = geometric_metrics(result.points, reference)

    assert result.scale_correction == pytest.approx(2.5, rel=0.02)
    assert result.scale_error_percent == pytest.approx(150.0, rel=0.03)
    assert distances["normalized_chamfer_l1"] < 1e-4


def test_accuracy_and_completeness_remain_directional():
    reference = np.column_stack(
        (np.arange(10, dtype=float), np.zeros(10), np.zeros(10))
    )
    prediction = reference[:5]

    metrics = geometric_metrics(prediction, reference)

    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["completeness"] > 0.0
    assert metrics["chamfer_l1"] == pytest.approx(
        metrics["completeness"] / 2.0
    )


def test_evaluate_geometry_returns_normalized_metrics():
    reference = _asymmetric_points(count=500)
    result = evaluate_geometry(reference.copy(), reference)

    assert result.scale_correction == pytest.approx(1.0)
    assert result.scale_error_percent == pytest.approx(0.0)
    assert result.normalized_accuracy == pytest.approx(0.0, abs=1e-12)
    assert result.normalized_completeness == pytest.approx(0.0, abs=1e-12)
    assert result.normalized_chamfer_l1 == pytest.approx(0.0, abs=1e-12)
    assert result.noise_error_percent == 0.0


def test_noise_error_counts_only_prediction_points_beyond_the_tolerance():
    reference = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    prediction = np.array([[0.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 2.0, 0.0]])

    metrics = geometric_metrics(prediction, reference, workers=1)

    assert metrics["noise_threshold_fraction"] == 0.05
    assert metrics["noise_error_percent"] == pytest.approx(100 / 3)
    assert metrics["completeness"] > 0.0
    assert geometric_metrics(
        prediction, reference, noise_threshold_fraction=0.1, workers=1,
    )["noise_error_percent"] == 0.0


def test_metrics_reject_nonfinite_points():
    reference = np.eye(3)
    prediction = reference.copy()
    prediction[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        evaluate_geometry(prediction, reference)
