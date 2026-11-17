"""Core steps for preparing video frames for a reconstruction model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityThresholds:
    """Simple, intentionally configurable image-quality thresholds."""

    min_brightness: float = 35.0
    max_brightness: float = 220.0
    min_contrast: float = 18.0
    min_sharpness: float = 40.0


@dataclass(frozen=True)
class QualityMetrics:
    brightness: float
    contrast: float
    sharpness: float


@dataclass(frozen=True)
class FrameResult:
    sample_number: int
    source_frame_index: int
    timestamp_seconds: float
    path: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    metrics: QualityMetrics


@dataclass(frozen=True)
class PipelineResult:
    input_video: str
    requested_frames: int
    decoded_frames: int
    accepted_frames: int
    rejected_frames: int
    frames: tuple[FrameResult, ...]
    manifest_path: str
    reconstruction_request_path: str


class ReconstructionModel(Protocol):
    """Interface that an existing reconstruction model should implement."""

    def reconstruct(
        self, image_paths: Sequence[Path], output_dir: Path
    ) -> Path:
        """Create a 3D artifact and return its path."""


def select_frame_indices(total_frames: int, number_of_frames: int) -> list[int]:
    """Return exactly N approximately equidistant, unique frame indices."""

    if number_of_frames <= 0:
        raise ValueError("number_of_frames must be greater than zero")
    if total_frames <= 0:
        raise ValueError("the video contains no decodable frames")
    if number_of_frames > total_frames:
        raise ValueError(
            f"requested {number_of_frames} frames, but the video only has "
            f"{total_frames} frames"
        )

    return np.linspace(
        0, total_frames - 1, num=number_of_frames, dtype=np.int64
    ).tolist()


def measure_quality(rgb_frame: np.ndarray) -> QualityMetrics:
    """Measure luminance, contrast, and a Laplacian-like sharpness score."""

    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        raise ValueError("expected an RGB image with shape (height, width, 3)")

    frame = rgb_frame.astype(np.float32)
    gray = (
        0.2126 * frame[:, :, 0]
        + 0.7152 * frame[:, :, 1]
        + 0.0722 * frame[:, :, 2]
    )

    # The variance of this discrete Laplacian is a useful blur heuristic.
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * gray
    )

    return QualityMetrics(
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        sharpness=float(laplacian.var()),
    )


def evaluate_quality(
    metrics: QualityMetrics, thresholds: QualityThresholds
) -> tuple[str, ...]:
    """Return human-readable reasons why a frame should be rejected."""

    reasons: list[str] = []
    if metrics.brightness < thresholds.min_brightness:
        reasons.append("too_dark")
    if metrics.brightness > thresholds.max_brightness:
        reasons.append("too_bright")
    if metrics.contrast < thresholds.min_contrast:
        reasons.append("low_contrast")
    if metrics.sharpness < thresholds.min_sharpness:
        reasons.append("blurry")
    return tuple(reasons)


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Video support requires OpenCV. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return cv2


def _validate_inputs(video_path: Path, output_dir: Path) -> None:
    if video_path.suffix.lower() != ".mp4":
        raise ValueError("input video must have an .mp4 extension")
    if not video_path.is_file():
        raise FileNotFoundError(f"input video does not exist: {video_path}")
    if output_dir.resolve() == video_path.resolve():
        raise ValueError("output directory cannot be the input video")


def _decode_selected_frames(
    video_path: Path, selected_indices: Sequence[int]
) -> tuple[list[tuple[int, float, np.ndarray]], int]:
    """Decode selected frames in a single sequential pass through the video."""

    cv2 = _load_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 0.0

    wanted = set(selected_indices)
    frames: list[tuple[int, float, np.ndarray]] = []
    frame_index = 0
    try:
        while True:
            ok, bgr_frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_index / fps if fps else 0.0
                frames.append((frame_index, timestamp, rgb_frame))
            frame_index += 1
    finally:
        capture.release()

    return frames, frame_index


def _video_frame_count(video_path: Path) -> int:
    cv2 = _load_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def process_video(
    video_path: str | Path,
    number_of_frames: int,
    output_dir: str | Path,
    thresholds: QualityThresholds | None = None,
) -> PipelineResult:
    """Extract, assess, and prepare frames for reconstruction.

    The reconstruction itself is deliberately not invoked. Instead, this writes
    ``reconstruction_request.json`` as the stable handoff point for a future
    model adapter.
    """

    source = Path(video_path)
    destination = Path(output_dir)
    _validate_inputs(source, destination)
    thresholds = thresholds or QualityThresholds()

    total_frames = _video_frame_count(source)
    selected_indices = select_frame_indices(total_frames, number_of_frames)
    decoded, actual_total = _decode_selected_frames(source, selected_indices)
    if len(decoded) != number_of_frames:
        # Some codecs report an inaccurate frame count. A clear failure is safer
        # than silently handing an incomplete image set to reconstruction.
        raise ValueError(
            f"video metadata reported {total_frames} frames, but only "
            f"{actual_total} were decodable; extracted {len(decoded)} of "
            f"{number_of_frames} requested frames"
        )

    frames_dir = destination / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_results: list[FrameResult] = []
    for sample_number, (source_index, timestamp, rgb_frame) in enumerate(
        decoded, start=1
    ):
        filename = f"frame_{sample_number:04d}_source_{source_index:08d}.jpg"
        frame_path = frames_dir / filename
        Image.fromarray(rgb_frame).save(frame_path, format="JPEG", quality=95)

        metrics = measure_quality(rgb_frame)
        reasons = evaluate_quality(metrics, thresholds)
        frame_results.append(
            FrameResult(
                sample_number=sample_number,
                source_frame_index=source_index,
                timestamp_seconds=round(timestamp, 6),
                path=str(frame_path.resolve()),
                accepted=not reasons,
                rejection_reasons=reasons,
                metrics=metrics,
            )
        )

    accepted_paths = [frame.path for frame in frame_results if frame.accepted]
    manifest_path = destination / "frame_manifest.json"
    request_path = destination / "reconstruction_request.json"

    manifest_payload = {
        "input_video": str(source.resolve()),
        "requested_frames": number_of_frames,
        "quality_thresholds": asdict(thresholds),
        "frames": [asdict(frame) for frame in frame_results],
    }
    _write_json(manifest_path, manifest_payload)
    accepted_count = len(accepted_paths)
    _write_json(
        request_path,
        {
            "status": (
                "ready"
                if accepted_count == number_of_frames
                else "insufficient_quality_frames"
            ),
            "model": None,
            "requested_frames": number_of_frames,
            "accepted_frames": accepted_count,
            "image_paths": accepted_paths,
            "output_dir": str((destination / "reconstruction").resolve()),
            "todo": "Connect these inputs to a ReconstructionModel adapter.",
        },
    )

    return PipelineResult(
        input_video=str(source.resolve()),
        requested_frames=number_of_frames,
        decoded_frames=len(frame_results),
        accepted_frames=accepted_count,
        rejected_frames=len(frame_results) - accepted_count,
        frames=tuple(frame_results),
        manifest_path=str(manifest_path.resolve()),
        reconstruction_request_path=str(request_path.resolve()),
    )


def run_reconstruction(
    model: ReconstructionModel,
    image_paths: Sequence[Path],
    output_dir: Path,
) -> Path:
    """Future reconstruction boundary; intentionally not called by the CLI."""

    # TODO: Configure and invoke the selected existing reconstruction model.
    return model.reconstruct(image_paths, output_dir)
