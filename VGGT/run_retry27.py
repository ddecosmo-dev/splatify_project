import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/work/hdd/bjwl/zkang2/results/VGGT_all20")
PROJECT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT / "VGGT" / "benchmark_vggt.py"

FIELDS = [
    "category", "spin_id", "views", "status",
    "total_runtime_seconds", "model_load_seconds",
    "preprocessing_seconds", "inference_seconds",
    "export_seconds", "peak_allocated_gib",
    "raw_valid_points", "masked_valid_points",
    "masked_filtered_points", "exported_points",
    "gpu", "precision", "error", "completed_utc"
]

manifest = ROOT / "manifest_retry.tsv"
summary = ROOT / "benchmark_retry.csv"

with manifest.open() as f:
    experiments = [line.strip().split("\t") for line in f if line.strip()]

assert len(experiments) == 27

with summary.open("w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writeheader()

for index, (category, spin_id, views) in enumerate(experiments, 1):
    out = ROOT / category / spin_id / f"{views}views"
    out.mkdir(parents=True, exist_ok=True)

    row = dict.fromkeys(FIELDS, "")
    row.update(category=category, spin_id=spin_id, views=int(views))

    cmd = [
        sys.executable, "-u", str(BENCHMARK),
        "--category", category,
        "--spin-id", spin_id,
        "--views", views
    ]

    print(f"[{index}/27] {category} {spin_id} {views} views", flush=True)
    start = time.perf_counter()

    try:
        with (out / "run.log").open("w") as log:
            result = subprocess.run(
                cmd, cwd=PROJECT, stdout=log,
                stderr=subprocess.STDOUT, check=False
            )

        if result.returncode != 0:
            raise RuntimeError(f"Exit code {result.returncode}")

        metrics_path = out / "masked_metrics.json"
        metrics = json.loads(metrics_path.read_text())

        if not (out / "pointcloud_masked.ply").is_file():
            raise RuntimeError("Masked PLY missing")

        raw = metrics
        masked = metrics["masked"]
        row.update(
            status="success",
            model_load_seconds=metrics.get("model_load_seconds"),
            preprocessing_seconds=metrics.get("preprocessing_seconds"),
            inference_seconds=metrics.get("inference_seconds"),
            export_seconds=metrics.get("export_seconds"),
            peak_allocated_gib=metrics.get("peak_allocated_gib"),
            raw_valid_points=raw.get("valid_points"),
            masked_valid_points=masked.get("valid_points"),
            masked_filtered_points=masked.get("after_confidence_filter"),
            exported_points=masked.get("exported_points"),
            gpu=metrics.get("gpu"),
            precision=metrics.get("precision")
        )
    except Exception as exc:
        row.update(status="failed", error=str(exc))

    row["total_runtime_seconds"] = round(time.perf_counter() - start, 3)
    row["completed_utc"] = datetime.now(timezone.utc).isoformat()

    with summary.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(row)

    print(
        f"  {row['status']} | {row['total_runtime_seconds']} s",
        flush=True
    )

print("All 27 retries attempted.", flush=True)
