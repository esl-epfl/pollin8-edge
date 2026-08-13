#!/usr/bin/env bash
#SBATCH --job-name=insect-basefv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%A_%a.out
# Recover a stronger, CONVERGED FP32 base_320 baseline.
#
# WHY: the arch sweep (47_) validates each epoch on a 1,500-image FAST-VAL subset for speed, so
# early-stopping (patience) triggers on a noisy fitness and can stop the run under-converged
# (symptom: recall matches the reference but count-MAE is ~3x too high -> fuzzy, over-firing
# predictions -> depressed precision/F1). This script instead:
#   (1) validates each epoch on the FULL val split (tiled.yaml) -> clean early-stop fitness;
#   (2) trains 3 seeds (submit --array=0-2) so we can take the best;
#   (3) scores each with the EXACT reference metric (monitor_metrics, conf tuned on val by F1,
#       match_dist 0.05) so the number is directly comparable to results/metrics/sensei_base_320_s0.csv
#       (micro-F1 0.8171).
# Recipe is otherwise identical to 47_ (cfg yolov5p_sensei, hyp_sensei_base, yolov5n init, imgsz 320,
# batch 64), so this is a convergence/selection fix, not a recipe change. Longer budget + patience
# because full-val fitness is stable enough to keep improving past the fast-val stopping point.
#
#   sbatch --account=$ACCOUNT --partition=l40s --array=0-2 scripts/slurm/48_base_fullval_seeds.sbatch
#   # then pick the best:  python scripts/pick_best_seed.py base_320_fv
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/slurm/config.sh"
module purge; module load $PY_MODULES; source "$VENV/bin/activate"
cd "$CODE"; export PYTHONPATH="$CODE/src:$CODE/scripts" WANDB_MODE=disabled
python scripts/patch_yolov5_nwd.py "$WORK/yolov5" || true
YOLOV5_REPO="${YOLOV5_REPO:-$WORK/yolov5}"

seed="${SLURM_ARRAY_TASK_ID:?submit with --array=0-2 (seed = array task id)}"
size=320; BATCH=64; EPOCHS=300; PATIENCE=50
name="base_${size}_fv_s${seed}"
RUN_DIR="$RUNS/sensei_arch/$name"; BEST="$RUN_DIR/weights/best.pt"
echo "[basefv] seed=$seed name=$name FULL-val fitness epochs=$EPOCHS patience=$PATIENCE"

# --- train: FULL-val per-epoch fitness (tiled.yaml, not tiled_fastval.yaml) -----------------
python "$YOLOV5_REPO/train.py" \
  --cfg configs/yolov5p_sensei.yaml --weights yolov5n.pt \
  --data "$DATA/tiled.yaml" --hyp configs/hyp_sensei_base.yaml \
  --imgsz "$size" --batch-size "$BATCH" --epochs "$EPOCHS" --patience "$PATIENCE" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" --seed "$seed" \
  --project "$RUNS/sensei_arch" --name "$name" --exist-ok

# --- locked-test mAP (classic val.py) -------------------------------------------------------
python "$YOLOV5_REPO/val.py" --weights "$BEST" --data "$DATA/tiled.yaml" --task test --imgsz "$size" \
  --project "$RUNS/sensei_arch" --name "${name}_val" --exist-ok

# --- reference micro-F1 (conf tuned on val by F1) -> directly comparable to 0.8171 ----------
python -m insect_gap9.monitor_metrics \
  --backend yolov5-classic --yolov5-repo "$YOLOV5_REPO" \
  --weights "$BEST" --data "$DATA/tiled.yaml" \
  --split test --tune-on val --tune-metric f1 --imgsz "$size" --match-dist 0.05 \
  --out "results/metrics/sensei_${name}.csv" \
  --out-species "results/metrics/sensei_${name}_per_species.csv" \
  --out-sweep "results/metrics/sensei_${name}_conf_sweep.csv"
echo "[basefv] done seed=$seed -> $BEST ; results/metrics/sensei_${name}.csv"
