#!/usr/bin/env bash
#SBATCH --job-name=insect-train
#SBATCH --gres=gpu:1                 # GPU; QOS/partition passed by run_all.sh (verify: sacctmgr show qos ; sinfo)
#SBATCH --cpus-per-task=2
#SBATCH --mem=11G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# Submit with your account+resources:  sbatch --account=$ACCOUNT scripts/slurm/20_train.sbatch
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/slurm/config.sh"
module purge; module load $PY_MODULES; source "$VENV/bin/activate"
cd "$CODE"; export PYTHONPATH="$CODE/src:$CODE/scripts" WANDB_MODE=disabled
nvidia-smi

# from scratch, end-to-end, early stopping (config: epochs 300, patience 30, freeze 0)
python -m insect_gap9.train \
  --data "$DATA/tiled.yaml" --cfg configs/yolov5p_insect.yaml \
  --hyp configs/train.yaml --weights none.pt \
  --project "$RUNS" --name pico_tiled --cache disk

python -m insect_gap9.train_curve --run "$RUNS/pico_tiled"
echo "[train] best -> $RUNS/pico_tiled/weights/best.pt"
