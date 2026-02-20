#!/bin/bash
#SBATCH --job-name=pairvpr_eval
#SBATCH --output=runs/pairvpr_stagetwo_wildcross/logs/eval_%A_%a.out
#SBATCH --error=runs/pairvpr_stagetwo_wildcross/logs/eval_%A_%a.err
#SBATCH --time=21:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --account=OD-236362



# -------------------------
# Array Setup
# -------------------------
# Slurm automatically sets $SLURM_ARRAY_TASK_ID to the current index (0, 1, 2, or 3).
SPLIT_IDX="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${SPLIT_IDX}" ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set."
  exit 1
fi
if ! [[ "${SPLIT_IDX}" =~ ^[0-3]$ ]]; then
  echo "split index must be 0,1,2,3. Got: ${SPLIT_IDX}"
  exit 1
fi

# -------------------------
# Hardcoded defaults
# -------------------------
REPO_DIR="/scratch3/ned007/eai-walter-nedov-research/PairVPR/Pair-VPR"
DSETROOT="/scratch3/ned007/eai-walter-nedov-research"
OUTPUT_ROOT="runs/pairvpr_stagetwo_wildcross"
SUBSAMPLE_STEP=5
TOP_K=100

CKPT="${OUTPUT_ROOT}/split_${SPLIT_IDX}/finetuned_last.pth"
OUT_DIR="${OUTPUT_ROOT}/split_${SPLIT_IDX}/eval"
INTER_DIR="${OUT_DIR}/inter"
INTRA_DIR="${OUT_DIR}/intra"

mkdir -p "${INTER_DIR}" "${INTRA_DIR}"
# Also create the log directory so Slurm doesn't fail writing the .out files
mkdir -p "${OUTPUT_ROOT}/logs"
mkdir -p /scratch3/ned007/torch_cache
export TORCH_HOME="/scratch3/ned007/torch_cache"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}"
  exit 1
fi

VENMAN=(V-01 V-02 V-03 V-04)
KARAWATHA=(K-01 K-02 K-03 K-04)

HOLDOUT_V="${VENMAN[$SPLIT_IDX]}"
HOLDOUT_K="${KARAWATHA[$SPLIT_IDX]}"

echo "Split ${SPLIT_IDX} | holdout: ${HOLDOUT_V}, ${HOLDOUT_K}"
echo "Checkpoint: ${CKPT}"
echo "Output: ${OUT_DIR}"

source ~/.bashrc
mamba activate pairvpr_v2

cd "${REPO_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:."

# Inter-sequence: held-out query vs the other 3 in same environment.
for db in "${VENMAN[@]}"; do
  if [[ "${db}" != "${HOLDOUT_V}" ]]; then
    python pairvpr/eval/eval_wildcross.py \
      --db_seq "${DSETROOT}/WildCross-Replication/data/${db}" \
      --query_seq "${DSETROOT}/WildCross-Replication/data/${HOLDOUT_V}" \
      --trained_ckpt "${CKPT}" \
      --output_csv "${INTER_DIR}/${HOLDOUT_V}_vs_${db}.csv" \
      --subsample_step "${SUBSAMPLE_STEP}" \
      --top_k "${TOP_K}"
  fi
done

for db in "${KARAWATHA[@]}"; do
  if [[ "${db}" != "${HOLDOUT_K}" ]]; then
    python pairvpr/eval/eval_wildcross.py \
      --db_seq "${DSETROOT}/WildCross-Replication/data/${db}" \
      --query_seq "${DSETROOT}/WildCross-Replication/data/${HOLDOUT_K}" \
      --trained_ckpt "${CKPT}" \
      --output_csv "${INTER_DIR}/${HOLDOUT_K}_vs_${db}.csv" \
      --subsample_step "${SUBSAMPLE_STEP}" \
      --top_k "${TOP_K}"
  fi
done

# Intra-sequence on the 2 held-out routes.
python pairvpr/eval/eval_wildcross_intra.py \
  --seq_dir "${DSETROOT}/WildCross-Replication/data/${HOLDOUT_V}" \
  --trained_ckpt "${CKPT}" \
  --output_csv "${INTRA_DIR}/${HOLDOUT_V}.csv" \
  --subsample_step "${SUBSAMPLE_STEP}"

python pairvpr/eval/eval_wildcross_intra.py \
  --seq_dir "${DSETROOT}/WildCross-Replication/data/${HOLDOUT_K}" \
  --trained_ckpt "${CKPT}" \
  --output_csv "${INTRA_DIR}/${HOLDOUT_K}.csv" \
  --subsample_step "${SUBSAMPLE_STEP}"

echo "Done."