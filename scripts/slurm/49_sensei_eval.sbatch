#!/usr/bin/env bash
#SBATCH --job-name=insect-senseieval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
# Full monitoring metrics (centre-matched F1, counting MAE/bias, per-species R2) AND the
# normalised confusion matrix for a SELECTED SENSEI-arch run (anchor-based, classic yolov5
# checkpoint -> torch.hub backend). Run this only on the finalist(s) chosen from the val-mAP
# sweep (collect_sensei_sweep.py), not on all 54 -- caching predictions over ~23k tiles needs
# the memory. Confidence is tuned on VAL and the confusion matrix reuses that exact value.
#
#   RUN=focal_256_s0 sbatch --account=$ACCOUNT --partition=l40s scripts/slurm/49_sensei_eval.sbatch
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/slurm/config.sh"
module purge; module load $PY_MODULES; source "$VENV/bin/activate"
cd "$CODE"; export PYTHONPATH="$CODE/src:$CODE/scripts" WANDB_MODE=disabled

YOLOV5_REPO="${YOLOV5_REPO:-$WORK/yolov5}"
RUN="${RUN:?set RUN=<arm>_<size>_s<seed>, e.g. RUN=focal_256_s0}"
SIZE="$(echo "$RUN" | sed -E 's/.*_([0-9]+)_s[0-9]+$/\1/')"
WEIGHTS="$RUNS/sensei_arch/$RUN/weights/best.pt"
[ -f "$WEIGHTS" ] || { echo "[senseieval] missing $WEIGHTS"; exit 1; }
echo "[senseieval] RUN=$RUN size=$SIZE weights=$WEIGHTS"

TAG="sensei_${RUN}"
# Counting metrics (conf tuned on val by F1, applied to test) -> per-run CSVs
python -m insect_gap9.monitor_metrics \
  --backend yolov5-classic --yolov5-repo "$YOLOV5_REPO" \
  --weights "$WEIGHTS" --data "$DATA/tiled.yaml" \
  --split test --tune-on val --tune-metric f1 --imgsz "$SIZE" --match-dist 0.05 \
  --out "results/metrics/${TAG}.csv" \
  --out-species "results/metrics/${TAG}_per_species.csv" \
  --out-sweep "results/metrics/${TAG}_conf_sweep.csv"

# Reuse the tuned conf for the confusion matrix (so figure == table)
CONF=$(python -c "import csv;print(next(csv.DictReader(open('results/metrics/${TAG}.csv')))['conf'])")
python -m insect_gap9.confusion \
  --backend yolov5-classic --yolov5-repo "$YOLOV5_REPO" \
  --weights "$WEIGHTS" --data "$DATA/tiled.yaml" \
  --split test --imgsz "$SIZE" --conf "$CONF" --match-dist 0.05 \
  --out "results/figures/confusion_${RUN}.pdf"

echo "[senseieval] done -> results/metrics/${TAG}*.csv ; results/figures/confusion_${RUN}.pdf"
