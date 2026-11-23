"""Shared output adaptation and geometry evaluation utilities."""

from .adapters import GeometrySample, adapt_geometry
from .metrics import (
    AlignmentResult,
    GeometryMetrics,
    align_prediction,
    evaluate_geometry,
    geometric_metrics,
)

__all__ = [
    "AlignmentResult",
    "GeometryMetrics",
    "GeometrySample",
    "adapt_geometry",
    "align_prediction",
    "evaluate_geometry",
    "geometric_metrics",
]
