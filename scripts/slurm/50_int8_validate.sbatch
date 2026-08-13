#!/usr/bin/env bash
#SBATCH --job-name=insect-int8val
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Quantify INT8 detection/counting accuracy vs FP32 (and the degradation) for a trained SENSEI-arch
# run. INT8 here is ONNX-Runtime static PTQ -- a runnable PROXY for the on-device NE16/nntool
# quantiser (same INT8/PTQ family, different rounding) -> treat the delta as representative, not
# bit-identical to silicon. The eval uses the CLASSIC yolov5 repo (like 49_sensei_eval, since modern
# ultralytics mis-parses the anchor-based head): val.py gives mAP and monitor_metrics
# --backend yolov5-classic gives the centre-matched counting metrics; FP32 and INT8 are both
# evaluated as ONNX for an apples-to-apples delta.
#
# PREREQUISITES (training protocol, docs/training_kuma.md): data tiled (10_data_tile.sbatch) and the
# arch sweep trained (47_sensei_arch_sweep.sbatch), so weights exist at
#   $RUNS/sensei_arch/<arm>_<size>_s<seed>/weights/best.pt
#
#
# NOTE: submit to a FULL GPU (--partition=l40s), NOT a MIG slice -- the counting-metrics eval
#   caches all test-set predictions (~32 GB / 6 cores, like 49_sensei_eval); MIG slices cap at 2 cores.
#
# RUN -- deployed point:
#   RUN=base_320_s0 sbatch --account=$ACCOUNT --partition=l40s scripts/slurm/50_int8_validate.sbatch
#
# RUN -- full sweep (4 arms x 3 sizes x 3 seeds = 36 jobs; all APPEND to one CSV):
#   for a in base focal focal_noiw nwd; do for s in 192 320 512; do for d in 0 1 2; do
#     RUN=${a}_${s}_s${d} sbatch --account=$ACCOUNT --partition=l40s scripts/slurm/50_int8_validate.sbatch
#   done; done; done
#
# OUTPUT -> results/metrics/int8_validation.csv : per run THREE rows (precision in
#   {fp32, int8, delta(int8-fp32)}) with map50, map5095, micro_f1 (centre-matched F1), recall,
#   count_mae_pct, net_bias_pct, model_mb.
#
# FINAL FP32-vs-INT8 COMPARISON (login node, after the sweep finishes):
#   PYTHONPATH=src python scripts/int8_summary.py
#     -> prints the per-config FP32 / INT8 / delta table + the deployed-point headline, and writes
#        results/tables/int8_compare.tex for the paper.
#
# Full procedure + how to read the delta: docs/training_kuma.md (section 6).
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/slurm/config.sh"
module purge; module load $PY_MODULES; source "$VENV/bin/activate"
cd "$CODE"; export PYTHONPATH="$CODE/src:$CODE/scripts" WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"   # match SLURM cores (onnxruntime/BLAS)

# onnxruntime must be the CPU wheel (the default is a CUDA-13 build that fails on the nodes:
# libcudart.so.13). It is provisioned by 00_env_setup.sh on the LOGIN node -- ASSERT here, never
# pip-install in-job (compute nodes may lack network, and an in-job install can pull the wrong build).
python -c "import onnx, onnxruntime as o; assert 'CPUExecutionProvider' in o.get_available_providers(); print('[int8val] onnxruntime', o.__version__, o.get_available_providers())" \
  || { echo '[int8val] FATAL: onnx/onnxruntime missing or not the CPU build. On the LOGIN node run: bash scripts/slurm/00_env_setup.sh  (or: pip install onnx onnxruntime==1.20.1)'; exit 1; }

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
