# pollin8-edge — milliwatt-class, species-level pollinator & pest counting on a ULP SoC

Reproducibility artifacts for the paper *"Pollin8: Species-Level Pollinator and Pest Counting on a
Milliwatt System-on-Chip."* A compact, anchor-based
**YOLOv5p** detector (~0.31 M params, INT8) runs **wholly on the GreenWaves GAP9** SoC of the
SENSEI node and counts **nine** insect species on-node at **sub-100 mW** — within roughly a year of
battery life on a 3 Ah cell.

## Headline results
- **Monotonic accuracy vs. resolution** (base mAP@0.5): 192 px = 0.77, 320 px = 0.86, **512 px = 0.91**.
- **Deployed point = 320 px** (a deliberate *energy* choice; 512 px is the accuracy ceiling but
  costs ~2× the per-inference energy): centre-matched **F₁ 0.82**, recall **0.83**, per-species
  count-weighted **mean absolute count error 8.5 %** (non-cancelling), total count near-unbiased (+4 %).
- **Negative result**: off-the-shelf class re-balancing **hurts** counting — image-weighting
  collapses the abundant honeybee (recall 0.87 → 0.58) onto its rare Batesian-mimic drone fly. This
  is **invisible to standard mAP** and only exposed by counting-aligned metrics.
- **Measured cost on GAP9**: 2.6–13.0 mJ and 51–149 ms across 192–512 px at 40–74 mW.

All numbers above are regenerated from `results/metrics/*.csv` by the scripts in `scripts/`
(one-number-one-script: nothing is hand-typed).

## Repository layout
```
pollin8-edge/
├── configs/            network + training-recipe + dataset YAMLs
│   ├── yolov5p_sensei.yaml          the deployed detector (~0.31 M params)
│   ├── hyp_sensei_{base,focal,nwd}.yaml   the four loss-ablation recipes
│   └── insects1201.yaml             dataset / class config
├── src/insect_gap9/    pipeline: prepare/tile data, train, evaluate, confusion,
│                       quantize, simulate_gap9, measure_board, sensei_energy, ...
├── scripts/            analysis (make_figures/tables/values, sig_test, collect_*)
│   └── slurm/          training protocol for the Kuma cluster (see docs/training_kuma.md)
├── results/
│   ├── metrics/        FINAL CSVs: arch sweep, per-species, confidence sweeps,
│   │                   significance (Welch t-test), training curve
│   ├── tables/         LaTeX tables (monitoring, convergence, significance, ablation)
│   └── figures/        paper figures + ALL 9 confusion matrices + training curve
└── docs/               training, deployment, measurement, metrics, rationale, reproducibility
```

## Quick start (analysis only — no GPU needed)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# regenerate every paper figure/table/value from the committed metric CSVs:
PYTHONPATH=src python scripts/make_values.py
PYTHONPATH=src python scripts/make_tables.py
PYTHONPATH=src python scripts/make_figures.py
PYTHONPATH=src python scripts/sig_test.py        # significance.csv
```

## Dataset
We use the public **Bjerge *et al.* (2022)** benchmark (Zenodo `10.5281/zenodo.7395752`): 21,841
labelled training images over nine pollinator/pest species — among them the **drone fly**, a
hoverfly that mimics the honeybee (the source of the mimic-collapse result). The dataset is **not
redistributed here** (respect the upstream licence); fetch and tile it with:
```bash
PYTHONPATH=src python -m insect_gap9.download_data --dest data/raw          # fetch + md5-verify (~14 GB)
PYTHONPATH=src python -m insect_gap9.prepare_data  --raw data/raw --out data/insects1201
PYTHONPATH=src python -m insect_gap9.tile --src data/insects1201/train --out data/train_tiled --mode object --tile 320
PYTHONPATH=src python -m insect_gap9.tile --src data/insects1201/val   --out data/val_tiled   --mode grid --tile 320 --stride 320
PYTHONPATH=src python -m insect_gap9.tile --src data/insects1201/test  --out data/test_tiled  --mode grid --tile 320 --stride 320
# (scripts/slurm/10_data_tile.sbatch runs exactly this on the cluster)
```
Camera-trap insects occupy few pixels, so each image is cut into **320 px tiles** (object-centred
for training; a non-overlapping grid for val/test so counts are never double-scored).

## Training  →  [`docs/training_kuma.md`](docs/training_kuma.md)
Trained **from scratch** on the **Kuma** GPU cluster with the *classic* `ultralytics/yolov5` repo
(the modern `ultralytics` package only builds an anchor-free head). The core experiment is a
**4 arms × 3 sizes × 3 seeds = 36-task** SLURM array sweep (resumable, MIG-parallel). The four arms
isolate the loss/imbalance recipe; `base` (standard BCE, no re-balancing) is the deployed one.
The deployed baseline is trained to convergence with **full-validation fitness + best-of-3 seeds**
(`scripts/slurm/48_base_fullval_seeds.sbatch`, `scripts/pick_best_seed.py`) — recovering centre-matched
F₁ 0.82 on the locked test split.

## Pretrained weights  →  [`models/README.md`](models/README.md)
The **deployed checkpoint** (`sensei_base_320_fv_s1.pt` — the converged FP32 baseline, recovered
by full-validation fitness + best-of-3 seeds) and its **ReLU6** variant are committed under
[`models/checkpoints/`](models/checkpoints/) with truncated export graphs + `SHA256SUMS.txt`. The
full **4 arms × 3 sizes × 3 seeds = 36-checkpoint sweep** is published as a **GitHub Release** and
fetched on demand:
```bash
bash scripts/fetch_models.sh   # full 36-ckpt sweep from the GitHub Release (to be published);
                               # the deployed checkpoint above is already committed — no fetch needed
PYTHONPATH=src python -m insect_gap9.evaluate \
    --weights models/checkpoints/sensei_base_320_fv_s1.pt --split test  # reproduce the deployed result
```
Naming is `sensei_<arm>_<imgsz>_s<seed>.pt`; the deployed point is the `base`/320 px model.
Maintainers publish a release with `bash scripts/publish_models.sh <dir_of_renamed_ckpts>`.

## Deployment  →  [`docs/gvsoc_deployment.md`](docs/gvsoc_deployment.md)
The trained network is quantised to **8-bit integer** (accelerator requirement) and mapped onto
GAP9's NE16 convolution accelerator so weights and the largest activation fit the chip's 1.5 MB of
on-chip memory (only the input frame streams from on-package memory at the largest resolution).
GVSOC (cycle-accurate) and on-board flows are documented.

**INT8 accuracy (measured).** The deployed point is validated end-to-end with the **on-device quantiser** — GreenWaves NNTool, symmetric **SQ8** (the scheme NE16 runs) — on the locked 22,861-tile test split (`scripts/validate_int8_nntool.py`, [`51_nntool_validate.sbatch`](scripts/slurm/51_nntool_validate.sbatch)). It scores the INT8 convolutional backbone with the YOLO anchor decode in **software** (as on GAP9), and emits per-node quantisation SNR (`results/metrics/quant_qsnr.csv`) and a per-precision conf-grid (`results/metrics/int8_nntool_detail.csv`); FP32/INT8/Δ rows land in `results/metrics/int8_validation.csv`. On the SiLU detector, FP32→INT8 costs a few points (≈ΔF₁ −3.5 pp, ΔmAP −5 pp), dominated by the unbounded SiLU activation; a bounded **ReLU6** head makes the conversion lossless (forfeiting the operator-identical silicon energy lookup). Full analysis, failure modes, and the FP32-baseline recovery: [`docs/int8_quantization_analysis.md`](docs/int8_quantization_analysis.md).

> **NOTE.** The accuracy is measured in plain SQ8 because this `gap_sdk` clone's NE16 numpy emulation zeroes the graph output (NE16 is the accuracy-equivalent on-chip execution mode; set `NNTOOL_NE16=1` with the exact silicon SDK / Apptainer image). GreenWaves NNTool is **not** the PyPI `nntool` package — provision it via [`00b_nntool_env.sh`](scripts/slurm/00b_nntool_env.sh) (public `gap_sdk`) or the GAP9 SDK container. A runnable ONNX-Runtime PTQ proxy (`scripts/validate_int8.py`, [`50_int8_validate.sbatch`](scripts/slurm/50_int8_validate.sbatch)) remains for the full sweep.

## Measurement
Per-inference **energy and latency are measured on GAP9 silicon** for the operator-identical
reference detector and read off directly (`insect_gap9.sensei_energy`); the deployed nine-class
network differs **only** in the final detection convolution (single- → nine-class head, 18 → 42
channels, <2 % of the MACs, matching within 3.5 % at every size). `results/metrics/sensei_arch_sweep.csv`
carries the measured M-MAC / latency / energy / power per resolution. See
[`docs/training_kuma.md`](docs/training_kuma.md#measured-energy) and the architecture graph `results/figures/arch_graph.pdf`.

## Supporting evidence
- **All nine confusion matrices** (`results/figures/confusion_*.pdf`) — base/focal/NWD/focal-noiw
  across 192/256/320/512 px — show the deployed baseline keeping the honeybee on the diagonal while
  image-weighting leaks it onto the drone-fly column.
- **Per-arm / per-size metrics** (`results/metrics/sensei_*.csv`) and the **build-up ablation**
  (`results/tables/ablation.tex`).
- **Statistical significance** of the baseline's mAP lead (`results/metrics/significance.csv`,
  `results/tables/significance.tex`) — Welch's t-test over the three seeds.
- **Per-class convergence** with resolution (`results/figures/convergence.pdf`,
  `results/tables/convergence.tex`) and the **training curve** (`results/figures/train_curve.pdf`,
  `results/metrics/train_curve.csv`).

## Documentation
| Doc | Contents |
|-----|----------|
| [`docs/training_kuma.md`](docs/training_kuma.md) | Kuma training protocol (SLURM sweep, tiling, eval, measured energy) |
| [`docs/gvsoc_deployment.md`](docs/gvsoc_deployment.md) | Quantisation + GAP9/GVSOC deployment & on-board flow |
| [`docs/int8_quantization_analysis.md`](docs/int8_quantization_analysis.md) | INT8 (NNTool SQ8) accuracy: measured FP32→INT8 deltas, failure modes, calibration/conf/ReLU6 strategies, FP32-baseline recovery |
| [`docs/reproducibility.md`](docs/reproducibility.md) | End-to-end reproduction recipe |
| [`docs/rationale.md`](docs/rationale.md) | Why these design choices (architecture, metrics, loss) |
| [`docs/metrics.html`](docs/metrics.html) | Plain-language guide to the counting-aligned metrics |
| [`docs/extending.md`](docs/extending.md) | Extending the pipeline (new species, stronger detectors) |

## Reproducibility notes
- **One number, one script, one CSV**: every figure/table/value derives from `results/metrics/*.csv`.
- **Determinism**: seeds are pinned. The deployed baseline is the **best of three seeds** (`48_base_fullval_seeds.sbatch`); the committed per-arm sweep CSVs are seed-0 (the full 3-seed sweep + significance are reproduced via `47_sensei_arch_sweep.sbatch`).
- The **classic `yolov5`** repo and the **GreenWaves GAP SDK** (GAPflow / NNTool / GVSOC) are
  external; the quantise/simulate scripts degrade gracefully to a documented analytical model when
  the SDK is absent.

## Environment (exact versions)
Python **3.11.7** on RHEL 9.4 / SLURM 24.11.5. The pipeline uses **three separate venvs** so their pins never conflict; full `pip freeze` locks are in [`env/`](env/):
- **Training + FP32 eval** (`scripts/slurm/00_env_setup.sh`): `torch 2.3.1+cu121`, `torchvision 0.18.1+cu121`, `ultralytics 8.4.118`, `numpy 2.4.4`, `opencv-python 5.0.0.93`, `onnx 1.22.0`, `onnxruntime 1.20.1`.
- **INT8 NNTool SQ8 eval** (`scripts/slurm/00b_nntool_env.sh` + pins): `numpy 1.26.4`, `torch 2.13.0+cpu`, `onnx 1.14.1`, `onnxruntime 1.20.1`, `opencv-python-headless 4.10.0.84`, `cmd2 1.0.2`, `bfloat16 1.2.0`, `texttable 1.7.0`; NNTool via `PYTHONPATH=$GAP_SDK_SRC/tools/nntool`.
- **FP32 tflite export** (rung 1 only): `tensorflow-cpu 2.15.1`, `torch 2.13.0+cpu`, `numpy 1.26.4`.

External repos (pinned commits): classic `ultralytics/yolov5` @ `20d1d78`, GreenWaves `gap_sdk` @ `a230265`. Dataset: Zenodo `10.5281/zenodo.7395752`. Full table + rationale: [`env/README.md`](env/README.md).

## Citation
Please cite the paper. A BibTeX entry will be added on publication.

## License
See [`LICENSE`](LICENSE).
