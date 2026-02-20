#!/bin/bash
#SBATCH --job-name=eval_intra
#SBATCH --output=runs/logs/intra_%A_%a.out
#SBATCH --error=runs/logs/intra_%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --account=OD-236362

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${TASK_ID}" ]]; then exit 1; fi

# Math magic: Map 0-7 to a Split (0-3) and an Environment (0=Venman, 1=Karawatha)
SPLIT_IDX=$((TASK_ID / 2))
ENV_IDX=$((TASK_ID % 2))

REPO_DIR="/scratch3/ned007/eai-walter-nedov-research/PairVPR/Pair-VPR"
DSETROOT="/scratch3/ned007/eai-walter-nedov-research"
SUBSAMPLE_STEP=5

CKPT="${OUTPUT_ROOT}/split_${SPLIT_IDX}/finetuned_last.pth"
INTRA_DIR="${OUTPUT_ROOT}/split_${SPLIT_IDX}/eval/intra"
mkdir -p "${INTRA_DIR}" "${OUTPUT_ROOT}/logs" /scratch3/ned007/torch_cache
export TORCH_HOME="/scratch3/ned007/torch_cache"

VENMAN=(V-01 V-02 V-03 V-04)
KARAWATHA=(K-01 K-02 K-03 K-04)

if [[ ${ENV_IDX} -eq 0 ]]; then
  HOLDOUT="${VENMAN[$SPLIT_IDX]}"
else
  HOLDOUT="${KARAWATHA[$SPLIT_IDX]}"
fi

echo "Split ${SPLIT_IDX} | Intra-Eval: ${HOLDOUT}"

source ~/.bashrc
mamba activate pairvpr_v2
cd "${REPO_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:."

python pairvpr/eval/eval_wildcross_intra.py \
  --seq_dir "${DSETROOT}/WildCross-Replication/data/${HOLDOUT}" \
  --trained_ckpt "${CKPT}" \
  --output_csv "${INTRA_DIR}/${HOLDOUT}.csv" \
  --subsample_step "${SUBSAMPLE_STEP}"