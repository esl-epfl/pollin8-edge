# Exact environment locks (full reproducibility)

The pipeline uses **three separate venvs** (Python **3.11.7**) so their pins never conflict.
Full `pip freeze` locks are committed here; the summary is in the top-level README.

| lock file | built by | purpose | key pins |
|---|---|---|---|
| `requirements-train.lock` | `scripts/slurm/00_env_setup.sh` | training + FP32 eval (GPU) | torch 2.3.1+cu121, torchvision 0.18.1+cu121, ultralytics 8.4.118, numpy 2.4.4, opencv-python 5.0.0.93, onnx 1.22.0, onnxruntime 1.20.1 |
| `requirements-nntool-eval.lock` | `scripts/slurm/00b_nntool_env.sh` (+ pins below) | INT8 NNTool SQ8 eval (CPU) | numpy **1.26.4**, torch 2.13.0+cpu, onnx **1.14.1**, onnxruntime 1.20.1, opencv-python-headless **4.10.0.84**, cmd2 **1.0.2**, bfloat16 1.2.0, texttable 1.7.0 |
| `requirements-tflite-export.lock` | tflite export only (rung 1) | FP32 tflite export | tensorflow-cpu **2.15.1**, torch 2.13.0+cpu, numpy 1.26.4 |

**External repos (pinned commits, cloned on the login node):**
- classic `ultralytics/yolov5` @ `20d1d78` (training/eval/export; the modern `ultralytics` pkg only builds an anchor-free head)
- GreenWaves `gap_sdk` @ `a230265` (NNTool via `PYTHONPATH=$GAP_SDK_SRC/tools/nntool`; **not** the PyPI `nntool`)

**Cluster:** RHEL 9.4, SLURM 24.11.5, CUDA driver 12.x (cu121 wheels run via minor-version compat).
**Dataset:** Bjerge et al. 2022, Zenodo `10.5281/zenodo.7395752` (not redistributed).
