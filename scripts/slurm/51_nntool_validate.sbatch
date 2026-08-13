#!/usr/bin/env bash
#SBATCH --job-name=insect-nntoolval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%A_%a.out
# INT8 accuracy with the REAL deployed quantiser: GreenWaves NNTool (SQ8 scheme, NE16 mode)
# on the exported deployment graph -- the measured counterpart of 50_int8_validate.sbatch's
# ONNX-Runtime PROXY. Same CSV, same metric names, new provenance nntool-sq8-ne16(<ver>).
#
# NNTool's executer is a CPU numpy interpreter -- the GPU idles. --gres=gpu:1 is kept only
# because Kuma's QOS requires a GPU per job (QOSMinGRES, cf. 10_data_tile.sbatch); on a
# cluster with CPU partitions delete that line and submit to CPU. MIG slices work fine.
#
# PREREQUISITES (LOGIN node, once):
#   bash scripts/slurm/00b_nntool_env.sh                      # venv_nntool + GreenWaves nntool
#   # export the deployment graph next to the checkpoint (ladder rung 1 + parity reference):
#   source $WORK/venv_nntool/bin/activate
#   python $WORK/yolov5/export.py --weights $RUNS/sensei_arch/base_320_s0/weights/best.pt \
#          --include tflite --imgsz 320                       # -> weights/best-fp32.tflite
#   cp $RUNS/sensei_arch/base_320_s0/weights/best.pt /tmp/stage_best.pt   # protect dynamic best.onnx
#   python $WORK/yolov5/export.py --weights /tmp/stage_best.pt --include onnx --opset 13 --imgsz 320
#   mv /tmp/stage_best.onnx $RUNS/sensei_arch/base_320_s0/weights/best_static.onnx
#   # full ladder + rung-3 fallback: docs/training_kuma.md section 6b
#
# STAGES (MODE env var; RUN=<arm>_<size>_s<seed> like 50_*):
#   1) RUN=base_320_s0 MODE=prepare sbatch --account=$ACCOUNT --partition=mig12gb \
#        scripts/slurm/51_nntool_validate.sbatch              # stats+quantise+QSNR (~1 h)
#   2) RUN=base_320_s0 MODE=run sbatch --account=$ACCOUNT --partition=mig12gb --array=0-31 \
#        scripts/slurm/51_nntool_validate.sbatch              # 22,861 tiles / 32 shards
#   3) RUN=base_320_s0 MODE=merge sbatch --account=$ACCOUNT --partition=mig12gb \
#        scripts/slurm/51_nntool_validate.sbatch              # gates + rows appended (minutes)
#
# SMOKE (50 tiles, single shard; merge writes int8_validation_smoke.csv, never the paper CSV):
#   RUN=base_320_s0 MODE=run NNTOOL_LIMIT=50 sbatch --account=$ACCOUNT --partition=mig12gb \
#     scripts/slurm/51_nntool_validate.sbatch
#   RUN=base_320_s0 MODE=merge NNTOOL_ALLOW_PARTIAL=1 sbatch ... 51_nntool_validate.sbatch
#
# ENV: NNTOOL_ENV=venv (default; $WORK/venv_nntool from 00b) | sif (Apptainer GAP9 SDK:
#   NNTOOL_SIF=/path/to/gap9_sdk.sif, built per docs/gvsoc_deployment.md section 1).
#   NNTOOL_CONF=<float> overrides the frozen conf* at merge. NNTOOL_SKIP_FLOAT=1 is the
#   emergency int8-only lever (fp32 row reused + provenance-flagged).
#
# OUTPUT -> results/metrics/int8_validation.csv (3 rows: fp32, int8, delta(int8-fp32)),
#   results/metrics/quant_qsnr.csv (per-node QSNR), results/metrics/int8_nntool_detail.csv
#   (conf-grid sensitivity). Then: PYTHONPATH=src python scripts/int8_summary.py
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/slurm/config.sh"
module purge; module load $PY_MODULES
cd "$CODE"; export PYTHONPATH="$CODE/src:$CODE/scripts" WANDB_MODE=disabled
# This gap_sdk clone ships NNTool as a PYTHONPATH package (no setup.py -> not pip-installable),
# so put tools/nntool on the path. `from nntool.api import NNGraph` resolves from here.
export GAP_SDK_SRC="${GAP_SDK_SRC:-$WORK/gap_sdk_src}"
export PYTHONPATH="$GAP_SDK_SRC/tools/nntool:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# --- environment: venv (default) or the GAP9 SDK Apptainer image -------------
NNTOOL_ENV="${NNTOOL_ENV:-venv}"
if [ "$NNTOOL_ENV" = "sif" ]; then
  NNTOOL_SIF="${NNTOOL_SIF:?NNTOOL_ENV=sif needs NNTOOL_SIF=/path/to/gap9_sdk.sif}"
  PY() { apptainer exec --bind "$WORK","$HOME" "$NNTOOL_SIF" python "$@"; }
else
  source "${VENV_NNTOOL:-$WORK/venv_nntool}/bin/activate"
  PY() { python "$@"; }
fi

# Assert the REAL GreenWaves NNTool (PyPI 'nntool' is an unrelated impostor) -- provisioned
# by 00b_nntool_env.sh on the LOGIN node; never pip-install in-job (no network on compute).
PY -c "from nntool.api import NNGraph; print('[nntoolval] GreenWaves NNTool OK')" \
  || { echo '[nntoolval] FATAL: GreenWaves NNTool missing. LOGIN node: bash scripts/slurm/00b_nntool_env.sh'; exit 1; }
NNTOOL_VERSION="$(PY -c 'import importlib.metadata as im; print(im.version("nntool"))' 2>/dev/null || echo unknown)"
export NNTOOL_VERSION
echo "[nntoolval] env=$NNTOOL_ENV nntool=$NNTOOL_VERSION"

YOLOV5_REPO="${YOLOV5_REPO:-$WORK/yolov5}"
RUN="${RUN:?set RUN=<arm>_<size>_s<seed>, e.g. RUN=base_320_s0}"
ARM="$(echo "$RUN" | sed -E 's/_[0-9]+_s[0-9]+$//')"
SIZE="$(echo "$RUN" | sed -E 's/.*_([0-9]+)_s[0-9]+$/\1/')"
SEED="$(echo "$RUN" | sed -E 's/.*_s([0-9]+)$/\1/')"
# Overrides for RUN names that don't fit <arm>_<size>_s<seed> (e.g. an ablation suffix like _relu).
ARM="${NNTOOL_ARM:-$ARM}"; SIZE="${NNTOOL_SIZE:-$SIZE}"; SEED="${NNTOOL_SEED:-$SEED}"
WEIGHTS="$RUNS/sensei_arch/$RUN/weights/best.pt"
[ -f "$WEIGHTS" ] || { echo "[nntoolval] missing $WEIGHTS"; exit 1; }
CALIB="$DATA/val_tiled/images"; [ -d "$CALIB" ] || CALIB="$DATA/insects1201/val/images"

MODE="${MODE:-prepare}"
COMMON=( --weights "$WEIGHTS" --imgsz "$SIZE" --arm "$ARM" --seed "$SEED"
         --data "$DATA/tiled.yaml" --calib-dir "$CALIB" --yolov5-repo "$YOLOV5_REPO"
         --n-calib 300 )
# Optional: pin the exported graph (e.g. best_trunc.onnx = rung 3, decode in float Python,
# excluded from quantisation) and skip the strict float-parity gate (nntool-vs-ORT interpreter
# delta is ~5e-2 on this SDK; it cancels in the fp32->int8 delta, cross-check bounds the fp32 leg).
[ -n "${NNTOOL_GRAPH_PATH:-}" ] && COMMON+=( --graph-path "$NNTOOL_GRAPH_PATH" )
[ -n "${NNTOOL_SKIP_PARITY:-}" ] && COMMON+=( --skip-parity )
echo "[nntoolval] RUN=$RUN MODE=$MODE arm=$ARM size=$SIZE seed=$SEED calib=$CALIB"

case "$MODE" in
  prepare)
    PY "$CODE/scripts/validate_int8_nntool.py" --mode prepare "${COMMON[@]}"
    ;;
  run)
    SHARD="${SLURM_ARRAY_TASK_ID:-0}"
    NSHARDS="${SLURM_ARRAY_TASK_COUNT:-${NNTOOL_NUM_SHARDS:-1}}"
    EXTRA=()
    [ -n "${NNTOOL_LIMIT:-}" ] && EXTRA+=( --limit "$NNTOOL_LIMIT" )
    [ -n "${NNTOOL_SKIP_FLOAT:-}" ] && EXTRA+=( --skip-float )
    PY "$CODE/scripts/validate_int8_nntool.py" --mode run "${COMMON[@]}" \
       --shard "$SHARD" --num-shards "$NSHARDS" "${EXTRA[@]}"
    ;;
  merge)
    EXTRA=()
    [ -n "${NNTOOL_CONF:-}" ] && EXTRA+=( --conf "$NNTOOL_CONF" )
    [ -n "${NNTOOL_ALLOW_PARTIAL:-}" ] && EXTRA+=( --allow-partial )
    [ -n "${NNTOOL_SKIP_FLOAT:-}" ] && EXTRA+=( --skip-float )
    PY "$CODE/scripts/validate_int8_nntool.py" --mode merge "${COMMON[@]}" "${EXTRA[@]}"
    ;;
  *)
    echo "[nntoolval] unknown MODE=$MODE (prepare|run|merge)"; exit 1
    ;;
esac
echo "[nntoolval] $MODE done"
