#!/bin/bash -l
#SBATCH -p compute                 #specify partition (CPU-only baselines)
#SBATCH -N 1                       #specify number of nodes
#SBATCH --ntasks-per-node=1        #specify number of tasks per node
#SBATCH --cpus-per-task=4         #specify number of cpus (sklearn/XGBoost threads)
#SBATCH -t 12:00:00                 #job time limit <hr:min:sec>
#SBATCH -J baselines               #job name
#SBATCH -o logs/baselines_%j.out   #stdout log (%j = job id)
#SBATCH -e logs/baselines_%j.err   #stderr log

echo "Welcome to LANTA"
echo "Job: $SLURM_JOB_NAME   ID: $SLURM_JOB_ID   Node: $(hostname)"
echo "Started: $(date)"

# --- Environment ------------------------------------------------------------
ml Mamba/23.11.0-0                             # LANTA conda/mamba module
conda activate microbiome-gnn-disease              # env from environment.yml

# Keep BLAS/OpenMP thread counts in sync with the allocation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Run --------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# RF / logistic regression / XGBoost baselines.
# task.mode in config.yaml selects binary vs. OvR. Results are written after
# each feature/classifier/disease combo so timed-out jobs can be rerun.
python experiments/phase1_biomegpt_baseline/baselines.py

echo "Finished: $(date)"
