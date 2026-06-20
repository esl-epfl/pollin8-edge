#!/usr/bin/env bash
# ============================================================================
# SLURM configuration for the Kuma GPU cluster (or any Lmod + SLURM cluster).
# `source scripts/slurm/config.sh` before any training step. Scratch is
# per-cluster, so data staged elsewhere will not be visible to GPU jobs.
# Sanity check your allocation:  sacctmgr show assoc user=$USER ; sinfo -s
# ============================================================================
# Make `module` available in non-interactive shells (tmux / sbatch), not just login shells.
if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
           /usr/share/lmod/lmod/init/bash /etc/profile.d/z00_lmod.sh; do
    [ -f "$f" ] && . "$f" && break
  done
fi

# Your SLURM group/project account. REQUIRED on most clusters — set it before sourcing:
#   export ACCOUNT=<your_group_account>          (find it: sacctmgr show assoc user=$USER)
export ACCOUNT="${ACCOUNT:-}"
# GPU QOS: empty = cluster default. Set if your account needs an explicit one
# (e.g. normal | debug | long). Check: sacctmgr show assoc user=$USER format=QOS%-60
export GPU_QOS="${GPU_QOS:-}"
# Partition: Kuma has NO default and requires one explicitly. Find idle nodes with `sinfo -s`
# (NODES column A/I/O/T — want I>0). This model is tiny (<2 GB), so a MIG slice is ideal:
#   mig12gb / mig24gb (GPU SLICES — recommended) | l40s (full L40S) | h100 (full H100)
export PARTITION="${PARTITION:-mig12gb}"
export GPU_GRES="${GPU_GRES:-gpu:1}"
export PY_MODULES="${PY_MODULES:-gcc python}"      # adjust versions: module avail python

# --- Work dir: cluster scratch, NEVER inside the git repo -------------------
# Pick the first WRITABLE scratch/work base. Override anytime: export IGAP_WORK=/full/path
if [ -n "${IGAP_WORK:-}" ]; then
  _IGW="${IGAP_WORK}"
else
  _base=""
  for _b in "${SCRATCH:-}" "/scratch/kuma/${USER}" "/scratch/${USER}" "${WORK:-}"; do
    if [ -n "${_b}" ] && [ -w "${_b}" ]; then _base="${_b%/}"; break; fi
  done
  if [ -n "${_base}" ]; then
    _IGW="${_base}/pollin8-edge"
  else
    _IGW="${HOME}/pollin8-edge-work"
    echo "[config] WARNING: no writable scratch/work found -> ${_IGW} (mind home quota)"
  fi
fi
WORK="${_IGW}"; export WORK

export CODE="${HOME}/pollin8-edge"                 # this repo, checked out in /home
export VENV="${WORK}/venv"
export DATA="${WORK}/data"
export RUNS="${WORK}/runs"
# so `python -m insect_gap9.<x>` works after sourcing config + activating the venv
export PYTHONPATH="${CODE}/src:${CODE}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=disabled
mkdir -p "${WORK}" "${DATA}" "${RUNS}"
[ -z "${ACCOUNT}" ] && echo "[config] NOTE: ACCOUNT is empty — set 'export ACCOUNT=<group>' if your cluster requires it."
echo "[config] PARTITION=${PARTITION}  WORK=${WORK}"
