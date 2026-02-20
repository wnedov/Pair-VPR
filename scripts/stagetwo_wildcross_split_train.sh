#!/bin/bash
#SBATCH --job-name=pairvpr-wx-s2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --gres=gpu:4               # Requesting 4 GPUs here
#SBATCH --array=0-3                # Running splits 0, 1, 2, 3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=OD-236362

# ==========================================
# 1. Environment & GPU Setup
# ==========================================
source ~/.bashrc
mamba activate pairvpr_v2

# Safe handling of Split Index
SPLIT_IDX="${SLURM_ARRAY_TASK_ID}"
if [[ -z "${SPLIT_IDX}" ]]; then
    echo "Error: SPLIT_IDX not found. Are you running this with sbatch?"
    exit 1
fi

# Auto-detect GPU count from Slurm allocation
# This works regardless of whether you ask for 2, 4, 5, or 8 GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ "$NUM_GPUS" -lt 2 ]]; then
  echo "Error: PairVPR stage-2 needs at least 2 GPUs. Found: ${NUM_GPUS}"
  exit 1
fi

echo "--- Job Setup ---"
echo "Job ID: ${SLURM_JOB_ID} Array: ${SPLIT_IDX}"
echo "Detected GPUs: ${NUM_GPUS}"

# ==========================================
# 2. Configuration & Paths
# ==========================================
# Use absolute paths or standard variables to avoid directory confusion
REPO_DIR="/scratch3/ned007/eai-walter-nedov-research/PairVPR/Pair-VPR"
DSETROOT="/scratch3/ned007/eai-walter-nedov-research"
PRETRAINED_CKPT="${REPO_DIR}/trained_models/pairvpr-vitB.pth"
OUTPUT_ROOT="${REPO_DIR}/runs/pairvpr_stagetwo_wildcross_stage2"
OUTPUT_DIR="${OUTPUT_ROOT}/split_${SPLIT_IDX}"

# Hyperparameters
EPOCHS=100
BATCH_SIZE_PER_GPU=25
NUM_WORKERS=12
IMG_PER_PLACE=4
MIN_IMG_PER_PLACE=4
PLACE_CELL_SIZE_M=10.0
LOCAL_LOSS_WEIGHT=2.0

# ==========================================
# 3. Dataset Split Logic
# ==========================================
VENMAN=(V-01 V-02 V-03 V-04)
KARAWATHA=(K-01 K-02 K-03 K-04)

HOLDOUT_V="${VENMAN[$SPLIT_IDX]}"
HOLDOUT_K="${KARAWATHA[$SPLIT_IDX]}"

TRAIN_ROUTES=()
for seq in "${VENMAN[@]}"; do
  [[ "$seq" != "$HOLDOUT_V" ]] && TRAIN_ROUTES+=("$seq")
done
for seq in "${KARAWATHA[@]}"; do
  [[ "$seq" != "$HOLDOUT_K" ]] && TRAIN_ROUTES+=("$seq")
done

echo "Split ${SPLIT_IDX} | Holdout: ${HOLDOUT_V}, ${HOLDOUT_K}"
echo "Train routes: ${TRAIN_ROUTES[*]}"
echo "Output dir: ${OUTPUT_DIR}"



wandb offline
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}" || exit 1
export PYTHONPATH="${PYTHONPATH:-}:."
export WANDB_NAME="split_${SPLIT_IDX}"
export WANDB_RUN_GROUP="wildcross_stagetwo"
mkdir -p /scratch3/ned007/torch_cache
export TORCH_HOME="/scratch3/ned007/torch_cache"
export WANDB_API_KEY="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"

# ==========================================
# 4. Training Execution
# ==========================================
python pairvpr/training/train_stagetwo.py \
  --config-file-finetuned "pairvpr/configs/stagetwo_default_config.yaml" \
  --dsetroot "${DSETROOT}" \
  --pretrained_ckpt "${PRETRAINED_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_dataset wildcross \
  --wildcross_routes "${TRAIN_ROUTES[@]}" \
  --skip_validation \
  --usewandb \
  train.num_gpus="${NUM_GPUS}" \
  train.batch_size_per_gpu="${BATCH_SIZE_PER_GPU}" \
  train.num_workers="${NUM_WORKERS}" \
  optim.epochs="${EPOCHS}" \
  optim.locallossweight="${LOCAL_LOSS_WEIGHT}" \
  wildcross.img_per_place="${IMG_PER_PLACE}" \
  wildcross.min_img_per_place="${MIN_IMG_PER_PLACE}" \
  wildcross.place_cell_size_m="${PLACE_CELL_SIZE_M}"

# ==========================================
# 5. Post-Processing (Save Alias)
# ==========================================
LATEST_CKPT="$(python - "${OUTPUT_DIR}" <<'PY'
import glob, os, re, sys
out = sys.argv[1]
paths = glob.glob(os.path.join(out, "locas_*.pth"))
if not paths:
    # Don't fail the job if no checkpoints, just warn (optional)
    sys.exit(1)
def key(p):
    m = re.search(r"locas_(\d+)\.pth$", os.path.basename(p))
    return int(m.group(1)) if m else -1
print(sorted(paths, key=key)[-1])
PY
)"

if [[ -n "$LATEST_CKPT" ]]; then
    cp "${LATEST_CKPT}" "${OUTPUT_DIR}/finetuned_last.pth"
    echo "Saved stable alias: ${OUTPUT_DIR}/finetuned_last.pth"
else
    echo "Warning: No checkpoints found to alias."
fi