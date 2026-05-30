#!/bin/bash -l
#SBATCH -p gpu                     #specify partition (GPU nodes)
#SBATCH -N 1                       #specify number of nodes
#SBATCH --ntasks-per-node=1        #specify number of tasks per node
#SBATCH --gpus-per-node=1          #specify number of GPUs (1x A100)
#SBATCH --cpus-per-task=4         #specify number of cpus
#SBATCH -t 12:00:00                #job time limit <hr:min:sec>
#SBATCH -J biomegpt_train          #job name
#SBATCH -o logs/train_%j.out       #stdout log (%j = job id)
#SBATCH -e logs/train_%j.err       #stderr log

echo "Welcome to LANTA"
echo "Job: $SLURM_JOB_NAME   ID: $SLURM_JOB_ID   Node: $(hostname)"
echo "Started: $(date)"
nvidia-smi

# --- Environment ------------------------------------------------------------
ml Mamba/23.11.0-0                             # LANTA conda/mamba module
conda activate microbiome-gnn-disease              # env from environment.yml

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Run --------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# 1) Self-supervised pretraining → results/pretrain_checkpoint.pt
echo "=== Pretraining ==="
python experiments/phase1_biomegpt_baseline/pretrain.py

# 2) Fine-tune the frozen encoder. finetune.py runs first (writes the [CLS]
#    embedding cache); finetune_ovr.py reuses it. task.mode in config.yaml
#    decides whether the OvR step actually does work.
echo "=== Fine-tuning (binary) ==="
python experiments/phase1_biomegpt_baseline/finetune.py

echo "=== Fine-tuning (OvR) ==="
python experiments/phase1_biomegpt_baseline/finetune_ovr.py

echo "Finished: $(date)"
