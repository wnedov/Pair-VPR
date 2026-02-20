#!/bin/bash
#SBATCH --job-name=wx-pairVPR
#SBATCH --time=23:00:00          
#SBATCH --mem=32GB               
#SBATCH --gres=gpu:1             
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8              
#SBATCH --account=OD-236362
#SBATCH --array=6-11      
#SBATCH --output=logs/%x_%A_%a.out 
#SBATCH --error=logs/%x_%A_%a.err

# -------------------------------------------------------------------------
# 1. Define Sequences & Generate Pairs
# -------------------------------------------------------------------------
SEQS=("K-01" "K-02" "K-03" "K-04")

DB_LIST=()
QUERY_LIST=()

for db in "${SEQS[@]}"; do
    for query in "${SEQS[@]}"; do
        if [ "$db" != "$query" ]; then
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
echo "Processing: $CURRENT_QUERY (Query) vs $CURRENT_DB (Database)"
echo "=================================================="

# -------------------------------------------------------------------------
# 2. Initialize Environment
# -------------------------------------------------------------------------
eval "$(mamba shell hook --shell bash)"
mamba activate pairvpr_v2

# 3. Setup Directories
cd /scratch3/ned007/eai-walter-nedov-research/PairVPR/Pair-VPR

# Create the logs directory HERE (inside the repo), so 'tee logs/...' works
mkdir -p logs

export PYTHONPATH="${PYTHONPATH}:."
mkdir -p /scratch3/ned007/torch_cache
export TORCH_HOME="/scratch3/ned007/torch_cache"

# Define CSV Output Path
CSV_OUTPUT="logs/csv_results/viz_frames_${CURRENT_QUERY}_vs_${CURRENT_DB}.csv"

echo "Saving CSV results to: $CSV_OUTPUT"

# Ensure output directory exists
mkdir -p "$(dirname "$CSV_OUTPUT")"

python pairvpr/eval/eval_wildcross.py \
    --db_seq "pairvpr/datasets/datasetfiles/WildCross/${CURRENT_DB}" \
    --query_seq "pairvpr/datasets/datasetfiles/WildCross/${CURRENT_QUERY}" \
    --trained_ckpt trained_models/pairvpr-vitB.pth \
    --output_csv "$CSV_OUTPUT" \
    --subsample_step 5 \
    --top_k 100