"""Tests for VGGT rankings and aggregate reports."""

import pytest

from app.evaluate_vggt import SCORES, build_tables


def test_overall_summary_weights_each_product_once():
    runs = [
        {
            "variant": variant,
            "category": "CHAIR",
            "spin_id": spin_id,
            "views": views,
            "noise_threshold_fraction": 0.05,
            **dict.fromkeys(SCORES, score),
        }
        for variant, spin_id, views, score in (
            ("masked", "a", 18, 2),
            ("masked", "a", 36, 4),
            ("masked", "b", 18, 9),
            ("raw", "a", 18, 15),
        )
    ]

    tables = build_tables(runs)
    masked_overall = next(
        row for row in tables["summary.csv"]
        if row["variant"] == "masked" and row["views"] == "all"
    )
    assert masked_overall["object_count"] == 2
    assert masked_overall["run_count"] == 3
    assert masked_overall["mean_shape_error_percent"] == pytest.approx(6)
    assert masked_overall["median_noise_error_percent"] == pytest.approx(6)
    ranked = [row for row in tables["per_object.csv"] if row["variant"] == "masked"]
    assert [(row["shape_rank"], row["spin_id"]) for row in ranked] == [(1, "a"), (2, "b")]
    assert ranked[0]["views_included"] == "18,36"
