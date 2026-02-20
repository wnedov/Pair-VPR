#!/bin/bash
#SBATCH --job-name=eval_venman
#SBATCH --output=runs/pairvpr_stagetwo_wildcross/logs/venman_%A_%a.out
#SBATCH --error=runs/pairvpr_stagetwo_wildcross/logs/venman_%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --account=OD-236362

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${TASK_ID}" ]]; then exit 1; fi

# Math magic to map 0-11 to a split (0-3) and a DB index (0-2)
SPLIT_IDX=$((TASK_ID / 3))
DB_IDX=$((TASK_ID % 3))

REPO_DIR="/scratch3/ned007/eai-walter-nedov-research/PairVPR/Pair-VPR"
DSETROOT="/scratch3/ned007/eai-walter-nedov-research"

SUBSAMPLE_STEP=5
TOP_K=100

CKPT="${OUTPUT_ROOT}/split_${SPLIT_IDX}/finetuned_last.pth"
INTER_DIR="${OUTPUT_ROOT}/split_${SPLIT_IDX}/eval/inter"
mkdir -p "${INTER_DIR}" "${OUTPUT_ROOT}/logs" /scratch3/ned007/torch_cache
export TORCH_HOME="/scratch3/ned007/torch_cache"

VENMAN=(V-01 V-02 V-03 V-04)
HOLDOUT_V="${VENMAN[$SPLIT_IDX]}"

# Build a list of the 3 databases that are NOT the holdout
DBS=()
for db in "${VENMAN[@]}"; do
  if [[ "${db}" != "${HOLDOUT_V}" ]]; then
    DBS+=("${db}")
  fi
done

DB_TO_EVAL="${DBS[$DB_IDX]}"

echo "Split ${SPLIT_IDX} | Query: ${HOLDOUT_V} | DB: ${DB_TO_EVAL}"

source ~/.bashrc
mamba activate pairvpr_v2
cd "${REPO_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:."

python pairvpr/eval/eval_wildcross.py \
  --db_seq "${DSETROOT}/WildCross-Replication/data/${DB_TO_EVAL}" \
  --query_seq "${DSETROOT}/WildCross-Replication/data/${HOLDOUT_V}" \
  --trained_ckpt "${CKPT}" \
  --output_csv "${INTER_DIR}/${HOLDOUT_V}_vs_${DB_TO_EVAL}.csv" \
  --subsample_step "${SUBSAMPLE_STEP}" \
  --top_k "${TOP_K}"