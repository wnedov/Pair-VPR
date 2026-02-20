#!/bin/bash
#SBATCH --job-name=wx-pairVPR-perf
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --account=OD-236362
#
# Inter-sequence WildCross evaluation in "performance" mode:
# - ViT-G checkpoint
# - pairvpr_performance.yaml (refinetopcands=500, memoryeffmode=false)
# - explicitly rerank Top-500 candidates (can change below)
#
# This script generates *within-environment* pairs only (K↔K and V↔V), excluding self-pairs.
# That matches the typical 24-pair setup for 4 K sequences + 4 V sequences.
#SBATCH --array=0-23
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# -------------------------------------------------------------------------
# 1. Define Sequences & Generate Pairs (within-env only)
# -------------------------------------------------------------------------
SEQS=("V-01" "V-02" "V-03" "V-04" "K-01" "K-02" "K-03" "K-04")

DB_LIST=()
QUERY_LIST=()

for db in "${SEQS[@]}"; do
    for query in "${SEQS[@]}"; do
        # Only compare within the same environment, and skip self-pairs
        if [[ "${db:0:1}" == "${query:0:1}" && "$db" != "$query" ]]; then
            DB_LIST+=("$db")
            QUERY_LIST+=("$query")
        fi
    done
done

CURRENT_DB=${DB_LIST[$SLURM_ARRAY_TASK_ID]}
CURRENT_QUERY=${QUERY_LIST[$SLURM_ARRAY_TASK_ID]}

# Safety check
NUM_PAIRS=${#DB_LIST[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_PAIRS" ]; then
    echo "Error: Task ID $SLURM_ARRAY_TASK_ID exceeds number of pairs ($NUM_PAIRS)."
    exit 1
fi

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing (PERF): $CURRENT_QUERY (Query) vs $CURRENT_DB (Database)"
echo "=================================================="

# -------------------------------------------------------------------------
# 2. Initialize Environment
# -------------------------------------------------------------------------
eval "$(mamba shell hook --shell bash)"
mamba activate pairvpr_v2

# -------------------------------------------------------------------------
# 3. Setup Directories
# -------------------------------------------------------------------------
cd /scratch3/ned007/eai-walter-nedov-research/PairVPR/Pair-VPR
mkdir -p logs
export PYTHONPATH=".:${PYTHONPATH:-}"
mkdir -p /scratch3/ned007/torch_cache
export TORCH_HOME="/scratch3/ned007/torch_cache"
# -------------------------------------------------------------------------
# 4. Run Evaluation
# -------------------------------------------------------------------------
CSV_OUTPUT="logs/csv_results_perf/viz_frames_${CURRENT_QUERY}_vs_${CURRENT_DB}.csv"
echo "Saving CSV results to: $CSV_OUTPUT"
mkdir -p "$(dirname "$CSV_OUTPUT")"

python pairvpr/eval/eval_wildcross.py \
    --db_seq "pairvpr/datasets/datasetfiles/WildCross/${CURRENT_DB}" \
    --query_seq "pairvpr/datasets/datasetfiles/WildCross/${CURRENT_QUERY}" \
    --trained_ckpt trained_models/pairvpr-vitG.pth \
    --config pairvpr/configs/pairvpr_performance.yaml \
    --output_csv "$CSV_OUTPUT" \
    --subsample_step 5 \
    --top_k 500

