#!/bin/bash
#SBATCH --job-name=vggt-retry27
#SBATCH --account=bjwl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bjwl/zkang2/results/VGGT_all20/logs/slurm-retry-%j.out

set -e

cd /projects/bjwl/zkang2/splatify_project
export HF_HOME=/projects/bjwl/zkang2/hf_cache
export PYTHONUNBUFFERED=1

/projects/bjwl/zkang2/envs/vggt/bin/python -u VGGT/run_retry27.py
