"""Command-line interface for the naive Splatify preparation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import QualityThresholds, process_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample an MP4 into N frames, run basic quality checks, and "
            "prepare a reconstruction request."
        )
    )

    parser.add_argument("video", type=Path, help="path to the input .mp4 file")
    parser.add_argument(
        "-n",
        "--number-of-frames",
        type=int,
        required=True,
        help="number of approximately equidistant frames to extract",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="directory for frames and JSON manifests (default: output)",
    )
    
    parser.add_argument("--min-brightness", type=float, default=35.0)
    parser.add_argument("--max-brightness", type=float, default=220.0)
    parser.add_argument("--min-contrast", type=float, default=18.0)
    parser.add_argument("--min-sharpness", type=float, default=40.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = QualityThresholds(
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
        min_contrast=args.min_contrast,
        min_sharpness=args.min_sharpness,
    )

    try:
        result = process_video(
            video_path=args.video,
            number_of_frames=args.number_of_frames,
            output_dir=args.output_dir,
            thresholds=thresholds,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        build_parser().error(str(exc))

    print(
        json.dumps(
            {
                "decoded_frames": result.decoded_frames,
                "accepted_frames": result.accepted_frames,
                "rejected_frames": result.rejected_frames,
                "manifest": result.manifest_path,
                "reconstruction_request": result.reconstruction_request_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
