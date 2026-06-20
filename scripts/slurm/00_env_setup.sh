#!/usr/bin/env bash
# Run ON THE LOGIN NODE (it has internet; compute nodes usually do not).
# Builds the venv on /scratch and downloads the 14 GB benchmark once.
set -euo pipefail
source "$(dirname "$0")/config.sh"

module purge
module load $PY_MODULES

# --- venv (reuse if already built) ----------------------------------------
if [ -x "$VENV/bin/python" ]; then
  echo "[setup] reusing existing venv at $VENV"
  source "$VENV/bin/activate"
else
  python -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --upgrade pip
fi
# Install a CUDA-MATCHED PyTorch FIRST so the default pip resolution doesn't pull a
# mismatched build (e.g. cu130 on a CUDA-12 driver -> silent CPU fallback on the GPU node).
# Match Kuma's driver CUDA (see `nvidia-smi` top-right). Override: TORCH_CUDA=cu121|cu126
TORCH_CUDA="${TORCH_CUDA:-cu124}"
# (re)install only if the installed torch isn't already the target CUDA build —
# checked by the build string, since cuda.is_available() is always False on a login node.
if python -c "import torch,sys; sys.exit(0 if '+${TORCH_CUDA}' in torch.__version__ else 1)" 2>/dev/null; then
  echo "[setup] torch already built for ${TORCH_CUDA}"
else
  pip install --force-reinstall torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi
# idempotent: already-satisfied packages are a no-op
pip install ultralytics==8.4.67 pandas matplotlib pyyaml requests tqdm pytest
python -c "import torch;print('[setup] torch',torch.__version__,'(verify cuda on a GPU node)')"

# --- data (skip entirely if already prepared) -----------------------------
cd "$CODE"
export PYTHONPATH="$CODE/src:$CODE/scripts"
if [ -f "$DATA/insects1201/insects1201.auto.yaml" ] && \
   [ -d "$DATA/insects1201/train" ] && [ -d "$DATA/insects1201/test" ]; then
  echo "[setup] data already prepared at $DATA/insects1201 -- skipping download + prepare"
else
  # download_data is idempotent + resumable: verified files are skipped instantly,
  # partial files resume (no full re-download), so reruns after a drop are cheap.
  python -m insect_gap9.download_data --dest "$DATA/raw"          # train+val+test+models (~14 GB)
  python -m insect_gap9.prepare_data  --raw "$DATA/raw" --out "$DATA/insects1201"
fi
echo "[setup] done. venv=$VENV  data=$DATA"
echo "Next: sbatch --account=\$ACCOUNT scripts/slurm/10_data_tile.sbatch"
