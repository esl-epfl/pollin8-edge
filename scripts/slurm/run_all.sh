#!/usr/bin/env bash
# One-command automation: submit the whole pipeline as a Slurm dependency chain.
# Each job starts only if the previous one COMPLETED; a failure auto-cancels the rest.
#
# Usage:
#   bash scripts/slurm/run_all.sh             # full chain: tile -> train -> eval -> post
#   bash scripts/slurm/run_all.sh train       # start from training (data already tiled)
#   bash scripts/slurm/run_all.sh eval        # start from evaluation (best.pt already exists)
#   bash scripts/slurm/run_all.sh post        # just the figures/tables step
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
START="${1:-tile}"

sub() { sbatch --parsable --account="$ACCOUNT" "$@"; }
order="tile train eval post"
case " $order " in *" $START "*) ;; *) echo "bad start '$START' (use: $order)"; exit 1;; esac
echo "[run_all] ACCOUNT=$ACCOUNT  start=$START"

DEP=""; declare -A JID
submit() {  # name script [gpu]
  local name="$1" script="$2" gpu="${3:-}"
  local args=("$HERE/$script")
  [ -n "$DEP" ] && args=(--dependency=afterok:"$DEP" "${args[@]}")
  # partition (required on some clusters, e.g. Kuma) — applied to all steps if set
  [ -n "${PARTITION:-}" ] && args=(--partition="$PARTITION" "${args[@]}")
  # GPU steps get the QOS from config (only if set) — qos names vary per cluster
  [ -n "$gpu" ] && [ -n "${GPU_QOS:-}" ] && args=(--qos="$GPU_QOS" "${args[@]}")
  local id; id=$(sub "${args[@]}")
  JID[$name]=$id; DEP=$id
  echo "  + $name -> job $id"
}

started=0
for step in $order; do
  [ "$step" = "$START" ] && started=1
  [ "$started" = 1 ] || continue
  case "$step" in
    tile)  submit tile  10_data_tile.sbatch ;;
    train) submit train 20_train.sbatch gpu ;;
    eval)  submit eval  30_eval_sweep.sbatch gpu ;;
    post)  submit post  40_postprocess.sbatch ;;
  esac
done

ids=$(IFS=,; echo "${JID[*]}")
echo "[run_all] submitted: $ids"
echo "[run_all] watch :  squeue -u \$USER     (or: bash scripts/slurm/status.sh)"
echo "[run_all] result:  sacct -j $ids --format=JobID,JobName,State,Elapsed,ExitCode"
echo "[run_all] logs  :  tail -f insect-*-*.out"
