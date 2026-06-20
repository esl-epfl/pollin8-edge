#!/usr/bin/env bash
#SBATCH --job-name=insect-senseiarch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=11G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%A_%a.out
# SENSEI-architecture sweep: train the anchor-based YOLOv5p that is OPERATOR- AND SIZE-
# IDENTICAL to the GAP9-silicon-characterised HeadCount model (Weise2025 Table I), so its
# latency/energy is a direct table lookup (no MAC interpolation). One array task =
# (arm, size, seed). Trained with the CLASSIC ultralytics/yolov5 repo ($YOLOV5_REPO) because
# the modern `ultralytics` package only builds the anchor-free head.
#
# Grid: ARMS x SIZES x SEEDS = 4 x 3 x 3 = 36 tasks. This single sweep delivers, at once:
#   - matched-architecture accuracy at small/medium/large input sizes 192/320/512
#     (energy is a DIRECT Wiese2025 Table I readoff at every size; all fit on-chip)
#   - 3-seed variance (criticism: single-run)
#   - loss ablation A (baseline) vs B (focal+image-weights) vs C (B + NWD tiny-object box loss),
#     plus D (B MINUS --image-weights) -- the CONTROL that isolates inverse-frequency resampling
#     from focal+copy-paste, so the honeybee/Batesian-mimic count collapse seen in B is cleanly
#     attributable (if D recovers the honeybee and B does not, --image-weights is the cause).
#
# Per-epoch validation uses a 1,500-image fast-val subset ($DATA/tiled_fastval.yaml, built on
# the login node) so an epoch is dominated by TRAINING, not val; the final test mAP below still
# scores the FULL locked test split. Build the subset before launching -- NOTE the val: path MUST
# be ABSOLUTE (tiled.yaml has no `path:` root, so a relative val path fails "Dataset not found"):
#   python - <<'PY'
#   import os,glob,random; d=os.environ['DATA']
#   im=sorted(glob.glob(os.path.join(d,'val_tiled','images','*'))); random.seed(0); random.shuffle(im)
#   txt=os.path.join(d,'val_fastval.txt'); open(txt,'w').write('\n'.join(im[:1500])+'\n')
#   lines=open(os.path.join(d,'tiled.yaml')).read().splitlines()
#   out=[(f'val: {txt}' if l.strip().startswith('val:') else l) for l in lines]
#   open(os.path.join(d,'tiled_fastval.yaml'),'w').write('\n'.join(out)+'\n')
#   PY
# Validate WITHOUT a SLURM job before launching (reproduces train.py's dataset check):
#   cd "$YOLOV5_REPO" && python -c "from utils.general import check_dataset; check_dataset('$DATA/tiled_fastval.yaml', autodownload=False)" && cd "$CODE"
#
# Each task is RESUMABLE (config-hash guard + per-task flock + last.pt-every-epoch): a re-run or
# SLURM requeue CONTINUES from the last completed epoch and never restarts at 0. ONE-TIME, before
# the first launch, wipe stale dirs so --resume can't continue an old-config run:  rm -rf "$RUNS/sensei_arch"
#
# Full grid (all 36):  arms 0-8=base, 9-17=focal, 18-26=nwd, 27-35=focal_noiw  (seed fastest-varying).
# The model is <1.5GB, so MIG slices are ideal and far more numerous than full GPUs -> high
# concurrency. Check the real ceiling first (the earlier '%3 / JobArrayTaskLimit' was self-imposed):
#   sacctmgr show qos format=Name,MaxTRESPU%30,MaxJobsPU,MaxSubmitPU ; sinfo -p mig12gb -o '%P %a %D %t'
# PREFERRED single array on MIG (512px@batch48 fits a 12GB slice, ~6.5GB measured-extrapolated):
#   sbatch --account=$ACCOUNT --partition=mig12gb --array=0-35%16 scripts/slurm/47_sensei_arch_sweep.sbatch
# HYBRID (only if MIG is congested) — 512px is the long pole, peel it onto a full GPU:
#   sbatch --account=$ACCOUNT --partition=mig12gb --array=0-5,9-14,18-23,27-32%16 scripts/slurm/47_sensei_arch_sweep.sbatch  # 192+320
#   sbatch --account=$ACCOUNT --partition=l40s    --array=6-8,15-17,24-26,33-35%12 scripts/slurm/47_sensei_arch_sweep.sbatch # 512
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/slurm/config.sh"
module purge; module load $PY_MODULES; source "$VENV/bin/activate"
cd "$CODE"; export PYTHONPATH="$CODE/src:$CODE/scripts" WANDB_MODE=disabled

YOLOV5_REPO="${YOLOV5_REPO:-$WORK/yolov5}"
if [ ! -d "$YOLOV5_REPO" ]; then
  echo "[senseiarch] classic yolov5 repo not found at $YOLOV5_REPO"
  echo "             git clone https://github.com/ultralytics/yolov5 $YOLOV5_REPO"
  exit 1
fi
# Enable the NWD box-loss blend (arm C). Idempotent + hyp-gated, so safe to call every task
# and harmless for arms A/B (their hyp sets nwd_ratio=0). Patch the repo ONCE; re-runs no-op.
python scripts/patch_yolov5_nwd.py "$YOLOV5_REPO"

ARMS=(base focal nwd focal_noiw) # A=base, B=focal+image-weights, C=B+NWD, D=B minus image-weights
SIZES=(192 320 512)              # small/medium/large MEASURED sizes (Wiese2025 Table I); all on-chip
SEEDS=(0 1 2)
nSize=${#SIZES[@]}; nSeed=${#SEEDS[@]}; perArm=$(( nSize * nSeed ))
i="${SLURM_ARRAY_TASK_ID:?submit with --array, e.g. --array=0-26%4}"
arm="${ARMS[$(( i / perArm ))]}"
rest=$(( i % perArm ))
size="${SIZES[$(( rest / nSeed ))]}"
seed="${SEEDS[$(( rest % nSeed ))]}"

case "$arm" in
  base)       HYP="configs/hyp_sensei_base.yaml";  EXTRA="" ;;
  focal)      HYP="configs/hyp_sensei_focal.yaml"; EXTRA="--image-weights" ;;   # B: class balance
  nwd)        HYP="configs/hyp_sensei_nwd.yaml";   EXTRA="--image-weights" ;;   # C: focal + NWD box loss
  focal_noiw) HYP="configs/hyp_sensei_focal.yaml"; EXTRA="" ;;                  # D: B MINUS --image-weights
                                                                               #    (isolates the resampling
                                                                               #     from focal+copy-paste so the
                                                                               #     honeybee/mimic collapse is
                                                                               #     cleanly attributable)
  *) echo "[senseiarch] unknown arm '$arm'"; exit 1 ;;
esac

DATA_YAML="$DATA/tiled.yaml"            # FULL split — used for the final locked-test mAP below
[ -f "$DATA_YAML" ] || { echo "[senseiarch] $DATA_YAML missing — run the tiling pipeline first"; exit 1; }
# TRAINING validates each epoch on a 1,500-image fast-val subset (see header) so an epoch is
# dominated by training, not val. Final test mAP (val.py below) still uses the FULL split.
TRAIN_DATA_YAML="$DATA/tiled_fastval.yaml"
[ -f "$TRAIN_DATA_YAML" ] || { echo "[senseiarch] $TRAIN_DATA_YAML missing — build val_fastval.txt + tiled_fastval.yaml on the login node (see header)"; exit 1; }

# Model is tiny (0.31M params, 1 GFLOP — 192px batch64 used only ~1.1GB of the 45GB L40S), so we
# keep large batches even at 512px to cut iteration count and fit the deadline.
if   [ "$size" -le 192 ]; then BATCH=64
elif [ "$size" -le 320 ]; then BATCH=64
else                           BATCH=48; fi
# Converged-submission budget: patience will usually stop earlier; resume (below) survives
# any wall-time requeue. WORKERS<=cpus-per-task: yolov5 defaults to 8, which thrashes the 2
# cores we own on a MIG slice and makes this host/dataloader-bound model SLOWER.
EPOCHS=150; PATIENCE=30; WORKERS="${SLURM_CPUS_PER_TASK:-2}"
name="${arm}_${size}_s${seed}"
echo "[senseiarch] task $i -> arm=$arm size=$size seed=$seed batch=$BATCH epochs=$EPOCHS workers=$WORKERS name=$name"

# --- resumable / idempotent / requeue-safe runner -----------------------------------------
# Verified yolov5 facts that drive this (github.com/ultralytics/yolov5, checked at master):
#  * --resume reloads opt.yaml WHOLESALE -> CLI --epochs/--data/etc are IGNORED on resume, so
#    a STALE run dir would silently re-train the OLD config. Guarded by a config-hash (.cfg_sig).
#  * last.pt is written EVERY epoch -> a requeue continues from the last COMPLETED epoch.
#  * --resume on a FULLY finished run raises AssertionError (non-zero). We treat "non-zero BUT
#    best.pt exists" as finished->go-to-eval; any other failure exits so SLURM/user can requeue.
#  * .sensei_done is touched ONLY after the locked-test mAP succeeds (no '|| true').
RUN_DIR="$RUNS/sensei_arch/$name"; LAST="$RUN_DIR/weights/last.pt"; BEST="$RUN_DIR/weights/best.pt"
DONE="$RUN_DIR/.sensei_done"; SIGFILE="$RUN_DIR/.cfg_sig"
mkdir -p "$RUN_DIR"

# per-task lock: a double-scheduled / requeued index becomes a clean no-op, never a corrupt write
exec 9>"$RUN_DIR/.lock"
flock -n 9 || { echo "[senseiarch] $name: another task holds the lock — exiting clean"; exit 0; }

[ -f "$DONE" ] && { echo "[senseiarch] $name: already complete — skip"; exit 0; }

# staleness guard: refuse to --resume a dir whose baked config differs from this submission
CFG_SIG="$(printf '%s' "cfg=yolov5p_sensei|hyp=$HYP|epochs=$EPOCHS|imgsz=$size|batch=$BATCH|data=$TRAIN_DATA_YAML|seed=$seed|extra=$EXTRA" | sha256sum | cut -d' ' -f1)"
if [ -f "$SIGFILE" ] && [ "$(cat "$SIGFILE")" != "$CFG_SIG" ]; then
  echo "[senseiarch] $name: STALE config dir (sig mismatch) — refusing to resume."
  echo "             wipe it intentionally:  rm -rf '$RUN_DIR'  then resubmit."; exit 1
fi

run_fresh() {
  echo "$CFG_SIG" > "$SIGFILE"
  python "$YOLOV5_REPO/train.py" \
    --cfg configs/yolov5p_sensei.yaml --weights yolov5n.pt \
    --data "$TRAIN_DATA_YAML" --hyp "$HYP" \
    --imgsz "$size" --batch-size "$BATCH" --epochs "$EPOCHS" --patience "$PATIENCE" \
    --workers "$WORKERS" --seed "$seed" $EXTRA \
    --project "$RUNS/sensei_arch" --name "$name" --exist-ok
}

set +e
if [ -f "$LAST" ] && [ -f "$RUN_DIR/opt.yaml" ] && python -c "import torch; torch.load('$LAST', map_location='cpu')" 2>/dev/null; then
  echo "[senseiarch] $name: resuming from $LAST (config from opt.yaml)"
  python "$YOLOV5_REPO/train.py" --resume "$LAST"; rc=$?
elif [ -f "$LAST" ]; then
  echo "[senseiarch] $name: last.pt present but unresumable (no opt.yaml / corrupt) — archive + fresh"
  mv "$RUN_DIR" "${RUN_DIR}.corrupt.$$"; mkdir -p "$RUN_DIR"; exec 9>"$RUN_DIR/.lock"; flock -n 9 || exit 0
  run_fresh; rc=$?
else
  echo "[senseiarch] $name: fresh training"; run_fresh; rc=$?
fi
set -e

# finished-run AssertionError is success-to-eval; a real failure (no best.pt) -> requeue, no .done
if [ "$rc" -ne 0 ] && [ ! -f "$BEST" ]; then
  echo "[senseiarch] $name: train exit=$rc and no best.pt — exiting for requeue (NOT marking done)"; exit "$rc"
fi

# --- mAP on the locked test split (classic val.py). .done gated on its SUCCESS. -------------
python "$YOLOV5_REPO/val.py" \
  --weights "$BEST" --data "$DATA_YAML" --task test --imgsz "$size" \
  --save-txt --save-conf \
  --project "$RUNS/sensei_arch" --name "${name}_val" --exist-ok

touch "$DONE"
# NOTE: centre-matched F1 / counting / confusion for these CLASSIC checkpoints use the torch.hub
# loader (insect_gap9.monitor_metrics --backend yolov5-classic; see 49_sensei_eval.sbatch).
echo "[senseiarch] done -> $RUN_DIR (best.pt + locked-test mAP)"
