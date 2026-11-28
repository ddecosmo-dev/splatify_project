import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Project and official VGGT source
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = Path("/projects/bjwl/zkang2/vggt/official")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VGGT_ROOT))

from utils.ablation_dataset import ABODataset
from VGGT.mask_utils import load_mask
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def export_ply(predictions, output_path, percentile=10, max_points=400000, foreground_mask=None):
    """Export colored VGGT world points to binary PLY."""

    points = predictions["world_points"].detach().float().cpu().numpy()
    confidence = predictions["world_points_conf"].detach().float().cpu().numpy()
    images = predictions["images"].detach().float().cpu().numpy()

    # Remove batch dimension
    if points.ndim == 5:
        points = points[0]
    if confidence.ndim == 4:
        confidence = confidence[0]
    if images.ndim == 5:
        images = images[0]

    # Images: [S, 3, H, W] -> [S, H, W, 3]
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))

    colors = (np.clip(images, 0, 1) * 255).astype(np.uint8)

    points = points.reshape(-1, 3)
    confidence = confidence.reshape(-1)
    colors = colors.reshape(-1, 3)

    if not (len(points) == len(confidence) == len(colors)):
        raise ValueError("Point, confidence and color counts do not match")

    total_points = len(points)

    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(confidence)
        & (confidence > 1e-5)
    )

    if foreground_mask is not None:
        if foreground_mask.size != len(valid):
            raise ValueError('Mask and points do not match')
        valid &= foreground_mask.reshape(-1)
    points = points[valid]
    confidence = confidence[valid]
    colors = colors[valid]

    valid_points = len(points)

    if valid_points == 0:
        raise RuntimeError("No valid 3D points found")

    threshold = np.percentile(confidence, percentile)
    selected = confidence >= threshold

    points = points[selected]
    colors = colors[selected]

    filtered_points = len(points)

    if len(points) > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    # Binary PLY with RGB colors
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )

    for i, axis in enumerate(("x", "y", "z")):
        vertices[axis] = points[:, i]

    for i, channel in enumerate(("red", "green", "blue")):
        vertices[channel] = colors[:, i]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)

    return {
        "total_predicted_points": total_points,
        "valid_points": valid_points,
        "after_confidence_filter": filtered_points,
        "exported_points": len(vertices),
        "confidence_percentile": percentile,
        "max_export_points": max_points,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="CHAIR")
    parser.add_argument("--spin-id", default="0b35058e")
    parser.add_argument("--views", type=int, default=18)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("PyTorch:", torch.__version__, flush=True)

    # Load team-standard image selection
    dataset = ABODataset()
    sample = dataset.get_product(args.spin_id, num_images=args.views)

    paths = [str(p) for p in sample["image_paths"]]

    if len(paths) != args.views:
        raise RuntimeError("Unexpected number of images")

    output_dir = (
        Path("/work/hdd/bjwl/zkang2/results/VGGT_all20")
        / args.category.upper()
        / args.spin_id
        / f"{args.views}views"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Category:", args.category, flush=True)
    print("Spin ID:", args.spin_id, flush=True)
    print("Views:", len(paths), flush=True)
    print("Output:", output_dir, flush=True)

    # Load pretrained model
    print("Loading VGGT-1B...", flush=True)
    t0 = time.perf_counter()

    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.eval().to("cuda")

    model_load_seconds = time.perf_counter() - t0
    print("Model load time:", round(model_load_seconds, 2), "s", flush=True)

    # Preprocess images
    t0 = time.perf_counter()
    images = load_and_preprocess_images(paths).to("cuda")
    preprocessing_seconds = time.perf_counter() - t0

    print("Image shape:", tuple(images.shape), flush=True)

    # Run inference
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print("Starting inference...", flush=True)
    start = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(images)

    torch.cuda.synchronize()

    inference_seconds = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("Inference successful!", flush=True)
    print("Inference time:", round(inference_seconds, 2), "s", flush=True)
    print("Peak GPU allocated:", round(peak_gib, 2), "GiB", flush=True)

    # Export point cloud
    print("Exporting point cloud...", flush=True)
    t0 = time.perf_counter()

    ply_path = output_dir / "pointcloud_raw.ply"
    point_stats = export_ply(predictions, ply_path)
    masks = np.stack([load_mask(p, images.shape[-2:]) for p in paths])
    masked_stats = export_ply(predictions, output_dir / 'pointcloud_masked.ply', foreground_mask=masks)

    export_seconds = time.perf_counter() - t0

    # Save reproducibility information
    metrics = {
        "model": "VGGT-1B",
        "category": args.category.upper(),
        "spin_id": args.spin_id,
        "views": args.views,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "precision": "BF16 autocast",
        "model_load_seconds": model_load_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        "inference_seconds": inference_seconds,
        "export_seconds": export_seconds,
        "peak_allocated_gib": peak_gib,
        "image_shape": list(images.shape),
        "input_images": paths,
        "ground_truth_mesh": str(sample["mesh_path"]),
        "pointcloud": str(ply_path),
        "masked": masked_stats,
        **point_stats,
    }

    with open(output_dir / "masked_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Exported points:", point_stats["exported_points"], flush=True)
    print("PLY saved:", ply_path, flush=True)
    print("Metrics saved:", output_dir / "masked_metrics.json", flush=True)
    print("Benchmark complete!", flush=True)


if __name__ == "__main__":
    main()
