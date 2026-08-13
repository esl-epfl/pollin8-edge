#!/usr/bin/env bash
# Run ON THE LOGIN NODE (compute nodes have no internet). One-off setup for the NNTool
# (deployed-quantiser) INT8 validation -- scripts/slurm/51_nntool_validate.sbatch.
#
# Builds a SEPARATE venv ($WORK/venv_nntool) so nntool's dependency pins (protobuf/onnx/
# numpy) can never poison the training venv, and installs GreenWaves NNTool from a clone
# of the public gap_sdk. NOTE: `pip install nntool` from PyPI is an UNRELATED package --
# the final import check below guards against that impostor.
#
# Alternative env: the GAP9 SDK Apptainer image (exact silicon toolchain) -- build it per
# docs/gvsoc_deployment.md section 1 and submit 51_* with NNTOOL_ENV=sif NNTOOL_SIF=<path>.
set -euo pipefail
source "$(dirname "$0")/config.sh"

module purge
module load $PY_MODULES

VENV_NNTOOL="${VENV_NNTOOL:-$WORK/venv_nntool}"
GAP_SDK_SRC="${GAP_SDK_SRC:-$WORK/gap_sdk_src}"
GAP_SDK_URL="${GAP_SDK_URL:-https://github.com/GreenWaves-Technologies/gap_sdk}"

# --- venv (reuse if already built) ----------------------------------------
if [ -x "$VENV_NNTOOL/bin/python" ]; then
  echo "[nntool-env] reusing venv at $VENV_NNTOOL"
  source "$VENV_NNTOOL/bin/activate"
else
  python -m venv "$VENV_NNTOOL"
  source "$VENV_NNTOOL/bin/activate"
  pip install --upgrade pip
fi

# CPU-only torch (classic yolov5 NMS runs on CPU here; keeps the venv small).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# tflite export of the trained checkpoint (classic yolov5 models/tf.py); FP32 export only,
# NNTool quantises from its own statistics (never chain --int8 in front of it).
pip install "tensorflow-cpu==2.15.1"
# eval + parity deps; onnxruntime pinned CPU-only for the same reason as 00_env_setup.sh.
pip install ultralytics==8.4.67 pandas scipy pyyaml pillow opencv-python-headless tqdm \
            onnx "onnxruntime==1.20.1"

# --- GreenWaves NNTool from the public gap_sdk (reuse clone if present) ---
if [ -d "$GAP_SDK_SRC/.git" ]; then
  echo "[nntool-env] reusing gap_sdk clone at $GAP_SDK_SRC"
else
  git clone --depth 1 "$GAP_SDK_URL" "$GAP_SDK_SRC"
fi
pip install "$GAP_SDK_SRC/tools/nntool"

# --- impostor guard + version record --------------------------------------
# Only `from nntool.api import NNGraph` proves the REAL GreenWaves NNTool; the unrelated
# PyPI 'nntool' also satisfies a bare `import nntool`.
python - <<'EOF'
from nntool.api import NNGraph  # noqa: F401
import importlib.metadata as im
print("[nntool-env] GreenWaves NNTool OK, version", im.version("nntool"))
EOF
echo "[nntool-env] gap_sdk commit: $(git -C "$GAP_SDK_SRC" rev-parse --short HEAD)"
echo "[nntool-env] done. venv=$VENV_NNTOOL"
echo "Next (login node): export the deployment graph, then submit 51_nntool_validate.sbatch"
echo "  see docs/training_kuma.md section 6b for the export ladder + submissions"
