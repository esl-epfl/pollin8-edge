# Training protocol — Kuma GPU cluster

All networks in the paper were **trained from scratch** on the **Kuma** GPU cluster (a SLURM +
Lmod cluster with NVIDIA MIG slices and full L40S/H100 nodes). The pipeline trains with the
**classic [`ultralytics/yolov5`](https://github.com/ultralytics/yolov5) repo** — *not* the modern
`ultralytics` package, whose head is anchor-free — and reads back **measured** per-inference energy
via a direct lookup against the silicon-characterised reference network (see
[`measurement`](#measured-energy)). Porting to another SLURM cluster is just *environment + jobs*;
ready-made scripts are in [`scripts/slurm/`](../scripts/slurm/).

> **Verify before running** — these are allocation/cluster-specific and change over time:
> your **account** (`sacctmgr show assoc user=$USER`), GPU **QOS/partition** (`sinfo -s`), Python
> **module** names (`module avail python`), and your **scratch path** (`echo $SCRATCH`). Set them in
> [`scripts/slurm/config.sh`](../scripts/slurm/config.sh) (or override via env vars) before sourcing.

## 0. Setup
```bash
source scripts/slurm/config.sh          # sets ACCOUNT/WORK/DATA/RUNS/PYTHONPATH, inits Lmod
bash   scripts/slurm/00_env_setup.sh    # one-time: venv + classic-yolov5 clone + pinned deps
source "$VENV/bin/activate"
```
The model is **tiny** (~0.31 M params, ~1 GFLOP; 192 px @ batch 64 used ~1.1 GB on an L40S), so
**`mig12gb` GPU slices are ideal** — far more numerous than full GPUs, giving high job concurrency.
512 px @ batch 48 still fits a 12 GB slice; use a **full GPU** (`--partition=l40s`) only for the
long-pole 512 px arm if MIG is congested.

## 1. Tile the dataset
Object-centred 320 px crops for training and a **non-overlapping** 320 px grid for val/test, so
counts are never double-scored:
```bash
sbatch --account=$ACCOUNT scripts/slurm/10_data_tile.sbatch   # -> $DATA/tiled.yaml
```

## 2. Architecture sweep (the core experiment)

> **Deployed baseline — train to convergence.** The sweep validates each epoch on a 1,500-image fast-val subset for speed, which can early-stop a run under-converged (symptom: recall matches the reference but count-MAE is ~3× too high). For the *deployed* point use full-validation fitness + best-of-3 seeds instead:
>
> ```bash
> sbatch --account=$ACCOUNT --partition=l40s --array=0-2 scripts/slurm/48_base_fullval_seeds.sbatch
> python scripts/pick_best_seed.py base_320_fv   # picks the best of 3
> ```
`scripts/slurm/47_sensei_arch_sweep.sbatch` trains the anchor-based YOLOv5p
([`configs/yolov5p_sensei.yaml`](../configs/yolov5p_sensei.yaml), ~0.31 M params) that is
**operator- and size-identical** to the GAP9-silicon-characterised reference detector — which is
what makes the energy a direct table read-off rather than an interpolation.

| Axis  | Values |
|-------|--------|
| arm   | `base` (standard BCE — **deployed**), `focal` (+focal+image-weights), `nwd` (+focal+NWD box loss), `focal_noiw` (focal **without** image-weights — the control) |
| size  | 192, 320, 512 px |
| seed  | 0, 1, 2 |

→ 4 × 3 × 3 = **36 array tasks**, **resumable** (per-config hash guard + done-manifest) and
**MIG-parallel**:
```bash
sbatch --account=$ACCOUNT --partition=mig12gb --array=0-35%16 scripts/slurm/47_sensei_arch_sweep.sbatch
```
Hyperparameters per arm live in [`configs/hyp_sensei_{base,focal,nwd}.yaml`](../configs/). Training
uses SGD, a cosine LR schedule, and standard augmentation, stopping when the validation score
plateaus (so training length is set by the data, not a fixed epoch count).

## 3. Monitoring evaluation on the finalist(s)
Run the counting-aligned metrics + the normalised confusion matrix on the deployed checkpoint
(`base_320_s0`) — and any others you want as supporting evidence — not on all 36:
```bash
RUN=base_320_s0 sbatch --account=$ACCOUNT --partition=l40s scripts/slurm/49_sensei_eval.sbatch
```
This produces, per run, `results/metrics/sensei_<run>.csv` (overall F₁/recall/count-bias),
`..._per_species.csv`, `..._conf_sweep.csv`, and `results/figures/confusion_<run>.pdf`.

## <a name="measured-energy"></a>4. Measured energy
`scripts/collect_sensei_sweep.py` reads each run's best validation mAP and attaches the **direct**
per-size latency/energy of the operator-identical reference network (`insect_gap9.sensei_energy`).
Because the two networks share every operator and size — differing **only** in the final detection
convolution (single- → nine-class head, 18 → 42 channels, <2 % of the MACs) — this is a lookup,
not a projection. The on-board/cycle-accurate cross-check of the exact nine-class network is
described in [`gvsoc_deployment.md`](gvsoc_deployment.md).

## 5. Collect + significance + figures (login node)
```bash
python scripts/collect_sensei_sweep.py --runs "$RUNS/sensei_arch"   # -> results/metrics/sensei_arch_sweep.csv
python scripts/sig_test.py                                          # Welch t-test -> significance.csv
python scripts/make_values.py && python scripts/make_tables.py && python scripts/make_figures.py
```

## 6. INT8 accuracy validation (FP32 vs INT8)
The headline accuracy numbers are measured on the trained **FP32** model; the deployed network is
INT8 (NE16). This step quantifies what INT8 *costs in accuracy*. We use **ONNX-Runtime static
PTQ** as a runnable **proxy** for the on-device NE16/nntool quantiser (same INT8/PTQ family, only the
calibration/rounding differ) — so treat the delta as representative, not bit-identical to silicon.
The eval uses the **classic `yolov5` repo** (the same path as step 3, because the modern
`ultralytics` package mis-parses the anchor-based head and crashes), so the comparison is
apples-to-apples: export the trained `.pt` to ONNX, statically quantise it (calibrated on val
tiles), then evaluate **both** ONNX graphs through the classic repo — `val.py` for mAP and
`monitor_metrics --backend yolov5-classic` (loads ONNX via `DetectMultiBackend`) for the
centre-matched counting metrics.

> **onnxruntime must be the CPU wheel.** The default `pip install onnxruntime` now pulls a **CUDA-13**
> build that fails to import on CPU nodes (`libcudart.so.13`). The sbatch pins `onnxruntime==1.20.1`
> (CPU-only, numpy-2.x compatible); if you install it by hand, use that pin. The eval runs
> onnxruntime on **CPU** (the model is tiny), so the GPU on the l40s node is idle — `--partition=l40s`
> is requested only for its **6 cores** (MIG slices cap at 2).

Submit with `--partition=l40s`. Prerequisite: the arch sweep (step 2) so `$RUNS/sensei_arch/<run>/weights/best.pt` exists.
```bash
# deployed point:
RUN=base_320_s0 sbatch --account=$ACCOUNT --partition=l40s scripts/slurm/50_int8_validate.sbatch
# full sweep (4 arms x 3 sizes x 3 seeds = 36 jobs; all APPEND to one CSV):
for a in base focal focal_noiw nwd; do for s in 192 320 512; do for d in 0 1 2; do
  RUN=${a}_${s}_s${d} sbatch --account=$ACCOUNT --partition=l40s scripts/slurm/50_int8_validate.sbatch
done; done; done
```
Each job appends three rows to `results/metrics/int8_validation.csv` — `precision` ∈
`{fp32, int8, delta(int8-fp32)}` with `map50, map5095, micro_f1` (centre-matched F1), `recall,
count_mae_pct, net_bias_pct, model_mb`. Obtain the **final FP32-vs-INT8 comparison** on the login
node once the sweep finishes:
```bash
PYTHONPATH=src python scripts/int8_summary.py    # prints FP32/INT8/Δ per config + deployed headline
                                                 # writes results/tables/int8_compare.tex (paper-ready)
```
Read the `Δ(int8-fp32)` column as the INT8 cost: PTQ is typically within a few points of FP32 on
mAP/F1 while shrinking the model ~4× (`model_mb`). Quote the deployed point (base @ 320 px) in the
paper; the on-device NE16 result is expected comparable (the proxy caveat above).

## 6b. NNTool (deployed quantiser) validation — the measured INT8 delta

> **⚠️ Measured-run corrections** (full account: [`int8_quantization_analysis.md`](int8_quantization_analysis.md)). Running this end-to-end diverged from the plan below in three ways:
> 1. **Measure in plain SQ8, not NE16.** This `gap_sdk` clone's NE16 numpy emulation zeroes the graph output; `SQ8_OPTIONS={"scheme":"SQ8"}` is the default, `NNTOOL_NE16=1` re-enables NE16 for the exact silicon SDK (NE16 is the accuracy-equivalent on-chip execution mode).
> 2. **Use rung 3, not rung 1.** Quantising the in-graph anchor decode (rung-1 tflite) destroys box coordinates; export `best_trunc.onnx` (three raw conv heads) + `anchors.json` and submit with `NNTOOL_GRAPH_PATH=$W/best_trunc.onnx NNTOOL_SKIP_PARITY=1` so the decode runs in software (as on GAP9).
> 3. **NNTool is PYTHONPATH-based here** (no `pip install`): pin `numpy<2`, `cmd2==1.0.2`, `onnx==1.14.1`, `bfloat16`; the eval sbatch adds `PYTHONPATH=$GAP_SDK_SRC/tools/nntool`. The eval venv also needs `torchvision`, `opencv-python-headless<5`, `requests`, `matplotlib` for the real yolov5 NMS.

Section 6 measures the delta with a **proxy**. This step measures it with **GreenWaves NNTool
itself** (SQ8 scheme, NE16 mode) — the quantiser whose output runs on GAP9 — via
`scripts/validate_int8_nntool.py` + `scripts/slurm/51_nntool_validate.sbatch`. Rows land in the
same `results/metrics/int8_validation.csv` with provenance `nntool-sq8-ne16(<version>)`; proxy
rows are never touched or relabelled. Two deliberate differences from section 6, so the delta
isolates *quantisation* error: (i) the FP32 reference is NNTool **float** execution of the same
adjusted/fused graph (cross-checked against the recorded classic row, expect micro-F1 within
~0.02); (ii) the confidence threshold is **frozen** at the FP32 deployed operating point instead
of re-tuned per precision — the full conf-grid sensitivity is emitted anyway
(`results/metrics/int8_nntool_detail.csv`).

**One-off env (LOGIN node).** GreenWaves NNTool is **not on PyPI** (the PyPI `nntool` is an
unrelated package). `00b_nntool_env.sh` builds a separate venv (`$WORK/venv_nntool`) and installs
NNTool from a public `gap_sdk` clone; the alternative is the GAP9 SDK Apptainer image
(`NNTOOL_ENV=sif NNTOOL_SIF=...`, built per `docs/gvsoc_deployment.md` §1) for the exact silicon
toolchain — the NNTool version is recorded in the provenance either way.
```bash
bash scripts/slurm/00b_nntool_env.sh
```

**Export ladder (LOGIN node, venv_nntool).** Artifacts are cached next to the checkpoint; eval
jobs never export. FP32 export only — never `--int8`, which would chain TF's quantiser in front
of NNTool's and contaminate the measurement.
```bash
source $WORK/venv_nntool/bin/activate
W=$RUNS/sensei_arch/base_320_s0/weights
# rung 1 (primary; SENSEI-proven, anchor decode in-graph):
python $WORK/yolov5/export.py --weights $W/best.pt --include tflite --imgsz 320   # -> best-fp32.tflite
# parity reference (also rung 2 if tflite trips the importer). Stage a copy so the ORT
# pipeline's DYNAMIC best.onnx is never clobbered:
cp $W/best.pt /tmp/stage_best.pt
python $WORK/yolov5/export.py --weights /tmp/stage_best.pt --include onnx --opset 13 --imgsz 320
mv /tmp/stage_best.onnx $W/best_static.onnx
# rung 3 (last resort): truncated ONNX + float Python decode -- the prepare/run error
# messages print the anchors.json recipe; provenance is flagged `;decode=fp32-python`
# because decode quantisation is then excluded.
```
The `--mode prepare` **parity gate** (NNTool float vs onnxruntime on `best_static.onnx`,
normalised coords, tol 2e-3) decides whether a rung is usable; on failure fall to the next.

**Stages** (`MODE` env var, `RUN=<arm>_<size>_s<seed>` as in section 6). NNTool's executer is a
CPU numpy interpreter (~seconds/tile), hence the array sharding; MIG slices are fine (the GPU
idles — `--gres` is kept only for the QOS).
```bash
RUN=base_320_s0 MODE=prepare sbatch --account=$ACCOUNT --partition=mig12gb scripts/slurm/51_nntool_validate.sbatch
RUN=base_320_s0 MODE=run     sbatch --account=$ACCOUNT --partition=mig12gb --array=0-31 scripts/slurm/51_nntool_validate.sbatch
RUN=base_320_s0 MODE=merge   sbatch --account=$ACCOUNT --partition=mig12gb scripts/slurm/51_nntool_validate.sbatch
```
Rough wallclock at 320 px: prepare ≈ 1 h (300-tile statistics + quantise + 32-tile QSNR); run =
22,861 tiles / 32 shards ≈ 715 tiles × ~6.5 s (float+quant) ≈ 1.3 h/shard (4 h limit = 3×
margin); merge = minutes. Smoke first:
```bash
RUN=base_320_s0 MODE=run NNTOOL_LIMIT=50 sbatch --account=$ACCOUNT --partition=mig12gb scripts/slurm/51_nntool_validate.sbatch
RUN=base_320_s0 MODE=merge NNTOOL_ALLOW_PARTIAL=1 sbatch ...   # -> int8_validation_smoke.csv, never the paper CSV
```

**Outputs.** `int8_validation.csv` gains the three `nntool-sq8-ne16(<ver>)` rows (the headline);
`quant_qsnr.csv` holds the per-node QSNR — read it worst-first: a low-dB node names *where* the
quantisation error concentrates (the prepare log prints the worst-10); `int8_nntool_detail.csv`
shows the delta at every conf grid point (is it stable, or threshold luck?). Then
`PYTHONPATH=src python scripts/int8_summary.py` prints both quantiser families side by side and
writes `results/tables/int8_compare.tex` (proxy rows per resolution + the NNTool deployed-point
row); `make_values.py` fills `\valIntEightDelta{Map,FOne,Mae,Bias}` (NNTool row preferred).

## Useful job commands
```bash
squeue -u $USER                 # queue (PD pending, R running; gone = done)
tail -f *-*.out                 # live logs
sacct -j <id> --format=JobID,JobName,State,Elapsed,ExitCode
scancel -u $USER                # cancel all my jobs
```
