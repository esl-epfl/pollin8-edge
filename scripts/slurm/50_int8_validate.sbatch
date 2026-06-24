#!/usr/bin/env bash
#SBATCH --job-name=insect-int8val
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Quantify INT8 (ONNX-Runtime PTQ) accuracy vs FP32 and the degradation, for a SENSEI-arch run.
# ORT static PTQ is a runnable PROXY for the on-device NE16/nntool quantiser (same INT8/PTQ family;
# rounding differs) -> treat the delta as representative, not bit-identical to silicon.
#
#   RUN=base_320_s0 sbatch --account=$ACCOUNT --partition=mig12gb scripts/slurm/50_int8_validate.sbatch
# Or sweep all configs:
#   for a in base focal focal_noiw nwd; do for s in 192 320 512; do for d in 0 1 2; do
#     RUN=${a}_${s}_s${d} sbatch --account=$ACCOUNT --partition=mig12gb scripts/slurm/50_int8_validate.sbatch
#   done; done; done
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/slurm/config.sh"
module purge; module load $PY_MODULES; source "$VENV/bin/activate"
cd "$CODE"; export PYTHONPATH="$CODE/src:$CODE/scripts" WANDB_MODE=disabled

# onnxruntime stack (idempotent; pinned loosely to stay compatible with the env's numpy/onnx)
python -c "import onnx, onnxruntime" 2>/dev/null || pip -q install onnx onnxruntime

YOLOV5_REPO="${YOLOV5_REPO:-$WORK/yolov5}"
RUN="${RUN:?set RUN=<arm>_<size>_s<seed>, e.g. RUN=base_320_s0}"
ARM="$(echo "$RUN" | sed -E 's/_[0-9]+_s[0-9]+$//')"
SIZE="$(echo "$RUN" | sed -E 's/.*_([0-9]+)_s[0-9]+$/\1/')"
SEED="$(echo "$RUN" | sed -E 's/.*_s([0-9]+)$/\1/')"
WEIGHTS="$RUNS/sensei_arch/$RUN/weights/best.pt"
[ -f "$WEIGHTS" ] || { echo "[int8val] missing $WEIGHTS"; exit 1; }

# calibration tiles: prefer the grid-tiled val split (matches eval); fall back to val images
CALIB="$DATA/val_tiled/images"; [ -d "$CALIB" ] || CALIB="$DATA/insects1201/val/images"
echo "[int8val] RUN=$RUN arm=$ARM size=$SIZE seed=$SEED weights=$WEIGHTS calib=$CALIB"

python "$CODE/scripts/validate_int8.py" \
  --weights "$WEIGHTS" --imgsz "$SIZE" --arm "$ARM" --seed "$SEED" \
  --data "$DATA/tiled.yaml" --calib-dir "$CALIB" --yolov5-repo "$YOLOV5_REPO" \
  --n-calib 300 --out results/metrics/int8_validation.csv

echo "[int8val] done -> results/metrics/int8_validation.csv"
