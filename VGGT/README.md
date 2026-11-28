# VGGT Reconstruction Benchmark

This directory contains the VGGT reconstruction pipeline used for
the Splatify project.

## Overview

We evaluated the pretrained facebook/VGGT-1B model on 20 selected
Amazon Berkeley Objects (ABO) products.

- Categories: CHAIR, DRESSER, LAMP, SOFA, VASE
- Products: 4 per category
- Input views: 18, 36, and 72
- Total experiments: 60
- Completed successfully: 60/60

## Pipeline

1. Load the selected product images using ABODataset.
2. Preprocess input images using the official VGGT pipeline.
3. Run pretrained VGGT inference.
4. Export the raw colored point cloud.
5. Apply precomputed foreground segmentation masks.
6. Export the foreground-masked point cloud.
7. Record runtime, GPU memory, and point-cloud statistics.

Point clouds are exported in PLY format.

## Files

- test_vggt.py: Initial model testing
- export_pointcloud.py: Point-cloud export utility
- benchmark_vggt.py: Reconstruction and benchmark pipeline
- mask_utils.py: Foreground mask preprocessing
- run_all20.py: Batch runner for the 60 experiments
- run_all20.sbatch: Slurm job configuration
- run_retry27.py: Retry runner for previously failed experiments
- run_retry27.sbatch: Slurm retry job configuration

## Benchmark Results

See results/benchmark_summary_final.csv.

The CSV contains all 60 successful experiments, including:

- Total runtime
- Model loading time
- Image preprocessing time
- Inference time
- Point-cloud export time
- Peak allocated GPU memory
- Raw and masked point counts

The experiment manifests are also included in results/.

## Environment

Experiments were run on NCSA Delta with an NVIDIA A100 GPU.

The pipeline uses PyTorch, the official VGGT implementation,
the project's ABO dataset loader, and pretrained VGGT-1B weights.

The batch scripts currently contain Delta-specific absolute paths.
Update these paths for another machine or account.

## Reproduction Notes

The input dataset and segmentation masks are required to reproduce
the experiments. Dataset files and pretrained model weights are not
included in this repository.

The scripts use the original Delta directory structure. Check
paths, environment variables, and Slurm account settings before use.

Running the batch scripts again may overwrite existing benchmark
summary files. Use a separate output directory for new experiments.

## Outputs

Each experiment produces:

- pointcloud_raw.ply
- pointcloud_masked.ply
- masked_metrics.json
- run.log

The complete point clouds and execution logs are distributed
separately from the Git repository.

## Notes

The 27 unsuccessful attempts from the initial batch were caused
by mismatched mask and image preprocessing dimensions.

After correcting mask alignment, all 27 experiments were
successfully rerun, yielding 60/60 successful experiments.
