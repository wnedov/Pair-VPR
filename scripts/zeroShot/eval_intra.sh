#!/bin/bash
#SBATCH --job-name=wx-intra
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --account=OD-236362
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# -------------------------------------------------------------------------
# 1. Define Sequences (Intra-Sequence only requires the list)
# -------------------------------------------------------------------------
# We have 8 sequences total: 4 Venman (V) and 4 Karawatha (K)
SEQS=("V-01" "V-02" "V-03" "V-04" "K-01" "K-02" "K-03" "K-04")

# Select current sequence based on Array ID
CURRENT_SEQ=${SEQS[$SLURM_ARRAY_TASK_ID]}

# Safety check
NUM_SEQS=${#SEQS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_SEQS" ]; then
    echo "Error: Task ID $SLURM_ARRAY_TASK_ID exceeds number of sequences ($NUM_SEQS)."
    exit 1
fi

# Define paths
CSV_OUTPUT="logs/intra_results/viz_intra_frames_${CURRENT_SEQ}.csv"

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Intra-Sequence: $CURRENT_SEQ"
echo "Saving CSV to: $CSV_OUTPUT"
echo "=================================================="

# -------------------------------------------------------------------------
# 2. Initialize Environment
# -------------------------------------------------------------------------
eval "$(mamba shell hook --shell bash)"
mamba activate pairvpr_v2

# 3. Setup Directories
cd /scratch3/ned007/eai-walter-nedov-research/PairVPR/Pair-VPR

# Create specific intra results folder
mkdir -p logs/intra_results

export PYTHONPATH="${PYTHONPATH}:."
mkdir -p /scratch3/ned007/torch_cache
export TORCH_HOME="/scratch3/ned007/torch_cache"

# -------------------------------------------------------------------------
# 4. Run Evaluation
# -------------------------------------------------------------------------

python pairvpr/eval/eval_wildcross_intra.py \
    --seq_dir "pairvpr/datasets/datasetfiles/WildCross/${CURRENT_SEQ}" \
    --trained_ckpt trained_models/pairvpr-vitB.pth \
    --output_csv "$CSV_OUTPUT" \
    --time_thresh 600.0 \
    --subsample_step 5 \
    --top_k 100 \
    2>&1