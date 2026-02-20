#!/bin/bash
#SBATCH --job-name=eval_karawatha
#SBATCH --output=runs/pairvpr_stagetwo_wildcross/logs/karawatha_%A_%a.out
#SBATCH --error=runs/pairvpr_stagetwo_wildcross/logs/karawatha_%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --account=OD-236362

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${TASK_ID}" ]]; then exit 1; fi

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

KARAWATHA=(K-01 K-02 K-03 K-04)
HOLDOUT_K="${KARAWATHA[$SPLIT_IDX]}"

DBS=()
for db in "${KARAWATHA[@]}"; do
  if [[ "${db}" != "${HOLDOUT_K}" ]]; then
    DBS+=("${db}")
  fi
done

DB_TO_EVAL="${DBS[$DB_IDX]}"

echo "Split ${SPLIT_IDX} | Query: ${HOLDOUT_K} | DB: ${DB_TO_EVAL}"

source ~/.bashrc
mamba activate pairvpr_v2
cd "${REPO_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:."

python pairvpr/eval/eval_wildcross.py \
  --db_seq "${DSETROOT}/WildCross-Replication/data/${DB_TO_EVAL}" \
  --query_seq "${DSETROOT}/WildCross-Replication/data/${HOLDOUT_K}" \
  --trained_ckpt "${CKPT}" \
  --output_csv "${INTER_DIR}/${HOLDOUT_K}_vs_${DB_TO_EVAL}.csv" \
  --subsample_step "${SUBSAMPLE_STEP}" \
  --top_k "${TOP_K}"