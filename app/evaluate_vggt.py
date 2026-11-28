"""Evaluate raw and masked VGGT point clouds against the ABO reference meshes."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from utils.ablation_dataset import ABODataset
from utils.evaluation import adapt_geometry, evaluate_geometry
from utils.evaluation.metrics import DEFAULT_NOISE_THRESHOLD


VARIANTS = ("raw", "masked")
SCORES = (
    "shape_error_percent",
    "scale_error_percent",
    "noise_error_percent",
    "accuracy_percent",
    "completeness_percent",
)


@dataclass(frozen=True)
class ProductRuns:
    category: str
    spin_id: str
    reference: Path
    predictions: dict[tuple[int, str], Path]


def load_runs(root: Path, dataset: ABODataset) -> list[ProductRuns]:
    """Resolve every expected raw and masked cloud from the run manifest."""
    groups = defaultdict(dict)
    with (root / "manifest.tsv").open(newline="", encoding="utf-8") as file:
        for category, spin_id, num_views in csv.reader(file, delimiter="\t"):
            views = int(num_views)
            predictions = groups[category, spin_id]
            for variant in VARIANTS:
                key = (views, variant)
                path = (
                    root / category / spin_id / f"{views}views"
                    / f"pointcloud_{variant}.ply"
                )
                if key in predictions:
                    raise ValueError(f"Duplicate VGGT run: {path}")
                if not path.is_file():
                    raise FileNotFoundError(path)
                predictions[key] = path
    if not groups:
        raise ValueError(f"No VGGT runs in {root / 'manifest.tsv'}")

    return [
        ProductRuns(category, spin_id, dataset.get_mesh_path(spin_id), predictions)
        for (category, spin_id), predictions in sorted(groups.items())
    ]


def evaluate_product(
    product: ProductRuns,
    sample_count: int,
    seed: int,
    noise_threshold_fraction: float,
) -> list[dict]:
    """Score one product using the same reference surface sample for all runs."""
    reference = adapt_geometry(
        product.reference,
        kind="mesh",
        sample_count=sample_count,
        seed=seed,
    )
    rows = []
    for (views, variant), path in sorted(product.predictions.items()):
        try:
            prediction = adapt_geometry(
                path,
                kind="point_cloud",
                sample_count=sample_count,
                seed=seed,
            )
            metrics = evaluate_geometry(
                prediction.points,
                reference.points,
                seed=seed,
                noise_threshold_fraction=noise_threshold_fraction,
                workers=1,
            )
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"Could not evaluate {path}: {error}") from error
        rows.append({
            "variant": variant,
            "views": views,
            "category": product.category,
            "spin_id": product.spin_id,
            "shape_error_percent": 100 * metrics.normalized_chamfer_l1,
            "scale_error_percent": metrics.scale_error_percent,
            "noise_error_percent": metrics.noise_error_percent,
            "accuracy_percent": 100 * metrics.normalized_accuracy,
            "completeness_percent": 100 * metrics.normalized_completeness,
            "scale_correction": metrics.scale_correction,
            "reference_diagonal": metrics.reference_diagonal,
            "noise_threshold_fraction": metrics.noise_threshold_fraction,
            "prediction_sample_count": len(prediction.points),
            "reference_sample_count": len(reference.points),
            "seed": seed,
            "prediction_path": str(path),
            "reference_mesh": str(product.reference),
        })
    return rows


def _rank(rows: list[dict]) -> list[dict]:
    ordered = sorted(
        rows, key=lambda row: (row["shape_error_percent"], row["spin_id"]),
    )
    return [{"shape_rank": rank, **row} for rank, row in enumerate(ordered, start=1)]


def build_tables(rows: list[dict]) -> dict[str, list[dict]]:
    """Rank by shape error and summarize view budgets and per-product averages."""
    by_views = defaultdict(list)
    by_product = defaultdict(list)
    for row in rows:
        by_views[row["variant"], row["views"]].append(row)
        by_product[row["variant"], row["category"], row["spin_id"]].append(row)

    per_run = []
    summary = []
    by_variant = defaultdict(list)

    def summarize(variant, views, group):
        result = {
            "variant": variant,
            "views": views,
            "object_count": len({row["spin_id"] for row in group}),
            "run_count": sum(row.get("run_count", 1) for row in group),
            "noise_threshold_fraction": group[0]["noise_threshold_fraction"],
        }
        for score in SCORES:
            values = [row[score] for row in group]
            result[f"mean_{score}"] = mean(values)
            result[f"median_{score}"] = median(values)
        return result

    for (variant, views), group in sorted(by_views.items()):
        per_run.extend(_rank(group))
        summary.append(summarize(variant, views, group))

    for (variant, category, spin_id), group in sorted(by_product.items()):
        by_variant[variant].append({
            "variant": variant,
            "category": category,
            "spin_id": spin_id,
            "views_included": ",".join(map(str, sorted(row["views"] for row in group))),
            "run_count": len(group),
            **{score: mean(row[score] for row in group) for score in SCORES},
            "noise_threshold_fraction": group[0]["noise_threshold_fraction"],
        })

    per_object = []
    for variant, group in sorted(by_variant.items()):
        per_object.extend(_rank(group))
        summary.append(summarize(variant, "all", group))

    return {"per_run.csv": per_run, "per_object.csv": per_object, "summary.csv": summary}


def main(argv=None) -> int:
    """Run the VGGT batch evaluation and write CSV reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="extracted VGGT folder with manifest.tsv")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/abo-selected-20"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=100_000)
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=DEFAULT_NOISE_THRESHOLD,
        help="outlier distance as a fraction of reference diagonal (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.sample_count < 3 or args.jobs < 1 or args.seed < 0:
        parser.error("--sample-count must be at least 3, --jobs positive, and --seed nonnegative")
    if not math.isfinite(args.noise_threshold) or args.noise_threshold <= 0:
        parser.error("--noise-threshold must be positive and finite")

    try:
        products = load_runs(args.input_dir, ABODataset(args.dataset_dir))
        results = []
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            jobs = {
                executor.submit(
                    evaluate_product,
                    product,
                    args.sample_count,
                    args.seed,
                    args.noise_threshold,
                ): product
                for product in products
            }
            for count, job in enumerate(as_completed(jobs), start=1):
                product = jobs[job]
                results.extend(job.result())
                print(
                    f"[{count}/{len(products)}] {product.category}/{product.spin_id}",
                    flush=True,
                )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, table in build_tables(results).items():
            path = args.output_dir / filename
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
            print(f"Saved {len(table)} rows to {path}")
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
