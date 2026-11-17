import json

import numpy as np
import pytest

import app.pipeline as pipeline
from app.pipeline import (
    QualityMetrics,
    QualityThresholds,
    evaluate_quality,
    measure_quality,
    select_frame_indices,
)


def test_select_frame_indices_is_equidistant_and_includes_endpoints():
    assert select_frame_indices(total_frames=10, number_of_frames=4) == [0, 3, 6, 9]


def test_select_frame_indices_returns_exactly_n_unique_indices():
    indices = select_frame_indices(total_frames=101, number_of_frames=18)

    assert len(indices) == 18
    assert len(set(indices)) == 18
    assert indices[0] == 0
    assert indices[-1] == 100


@pytest.mark.parametrize(
    ("total", "requested", "message"),
    [
        (10, 0, "greater than zero"),
        (0, 1, "no decodable frames"),
        (2, 3, "only has 2 frames"),
    ],
)
def test_select_frame_indices_rejects_invalid_counts(total, requested, message):
    with pytest.raises(ValueError, match=message):
        select_frame_indices(total, requested)


def test_measure_quality_identifies_a_flat_frame():
    frame = np.full((16, 16, 3), 128, dtype=np.uint8)

    metrics = measure_quality(frame)

    assert metrics.brightness == pytest.approx(128.0)
    assert metrics.contrast == pytest.approx(0.0)
    assert metrics.sharpness == pytest.approx(0.0)


def test_evaluate_quality_reports_each_failure():
    thresholds = QualityThresholds(
        min_brightness=35,
        max_brightness=220,
        min_contrast=18,
        min_sharpness=40,
    )

    assert evaluate_quality(QualityMetrics(20, 5, 2), thresholds) == (
        "too_dark",
        "low_contrast",
        "blurry",
    )
    assert evaluate_quality(QualityMetrics(230, 30, 100), thresholds) == (
        "too_bright",
    )
    assert evaluate_quality(QualityMetrics(100, 30, 100), thresholds) == ()


def test_process_video_writes_not_ready_request_when_a_frame_fails(
    tmp_path, monkeypatch
):
    video_path = tmp_path / "input.mp4"
    video_path.touch()
    dark_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    detailed_frame = np.indices((8, 8)).sum(axis=0) % 2 * 255
    detailed_frame = np.repeat(detailed_frame[:, :, None], 3, axis=2).astype(
        np.uint8
    )

    monkeypatch.setattr(pipeline, "_video_frame_count", lambda _path: 2)
    monkeypatch.setattr(
        pipeline,
        "_decode_selected_frames",
        lambda _path, _indices: (
            [(0, 0.0, dark_frame), (1, 0.1, detailed_frame)],
            2,
        ),
    )

    result = pipeline.process_video(video_path, 2, tmp_path / "output")
    request = json.loads(
        (tmp_path / "output" / "reconstruction_request.json").read_text()
    )

    assert result.decoded_frames == 2
    assert result.accepted_frames == 1
    assert result.rejected_frames == 1
    assert request["status"] == "insufficient_quality_frames"
    assert request["requested_frames"] == 2
    assert request["accepted_frames"] == 1
    assert request["image_paths"] == [result.frames[1].path]
