#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -J biomegpt_ovr
#SBATCH -o logs/finetune_ovr_%j.out
#SBATCH -e logs/finetune_ovr_%j.err

echo "Welcome to LANTA"
echo "Job: $SLURM_JOB_NAME   ID: $SLURM_JOB_ID   Node: $(hostname)"
echo "Started: $(date)"
nvidia-smi

ml Mamba/23.11.0-0
conda activate microbiome-gnn-disease

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

echo "=== Fine-tuning (OvR; internal CV is sample-level sanity metric) ==="
python experiments/phase1_biomegpt_baseline/finetune_ovr.py

echo "Finished: $(date)"
