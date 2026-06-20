# GVSOC cycle-accurate deployment & cross-check runbook (OPTIONAL)

**Goal.** The paper's primary latency/energy is already a **direct measured look-up** from
Wiese2025 Table I (`src/insect_gap9/sensei_energy.py`), valid because our network is
operator- and size-identical to the silicon-characterised HeadCount model
(`configs/yolov5p_sensei.yaml`). This runbook is an **optional cycle-accurate cross-check**:
run the *actual* nc=9 insect graph under **GVSOC** to confirm that the look-up cycle count
matches the real model. GVSOC gives trustworthy cycles → latency; energy is then `latency ×
measured GAP9 active power` from the same Wiese2025 silicon. It is a **camera-ready / appendix
item**, not the headline number — a `gvsoc-cycle-accurate (this model)` row sits *beside* the
`measured (Wiese2025 Table I)` row, it never replaces it.

This reuses the SENSEI demo's GAP9 flow (`sensei-demo-yolov5/src_GAP`), which already
deploys this YOLOv5p family: per-size `tflite` → autotiler `generated_NxN/` → CMake build →
run. We swap their **nc=1 head-count** model for our **nc=9 insect** model.

---

## 0. Reality check (read first)

- **This is a multi-day bring-up, and it is OPTIONAL.** The risky steps are nntool ingestion
  of our model and autotiler generation, not the simulation itself. The paper does **not**
  depend on it: the shipped energy/latency is the measured Wiese2025 Table I look-up
  (`sensei_energy.measured_at_size`), already honest because the network is operator-identical.
  GVSOC is a *cross-check* — if it has not produced a clean per-layer cycle report, keep the
  measured look-up as the headline and land GVSOC for the camera-ready. Never ship a
  half-verified GVSOC number as the primary figure.
- **GVSOC is single-threaded CPU simulation — it does not need an HPC GPU.** The fastest
  path to the *first* number is to finish the Docker/SDK setup **on your laptop** (you
  already started `gap9_docker`), not the cluster. Use the cluster (Apptainer) only to *automate the
  sweep* (all sizes × many input images) once the flow works once by hand.
- **Sanity gate.** Before touching our model, build and GVSOC-run the **stock
  `YOLOv5_HeadCount_128x128`** model end-to-end. If that doesn't produce a cycle count, the
  problem is the toolchain, not our model — fix that first.

---

## 1. Toolchain

The demo's CMake needs two trees via environment variables:

- `GAP_SDK_HOME` → the GreenWaves GAP9 SDK (autotiler, nntool, gap riscv toolchain, GVSOC).
- `SENSEI_SDK_ROOT` → the ETH SENSEI BSP (`$SENSEI_SDK_ROOT/GAP9`), from
  `https://github.com/pulp-bio/sensei-sdk`, placed at `DIR/sensei-sdk`, entered via
  `./run.sh -d DIR` (your `gap9_docker` flow).

### 1a. Laptop (Docker) — finish what you started
1. Clone `sensei-sdk` to `DIR/sensei-sdk`; finish its SDK config so `./run.sh -d DIR` drops
   you into the container with `GAP_SDK_HOME`/`SENSEI_SDK_ROOT` exported.
2. Inside: `cd src_GAP && cmake -B build && cmake --build build` for the stock model
   (sanity gate above).

### 1b. the cluster (Apptainer) — for automation only
the cluster compute nodes **cannot run Docker** (no root). Convert the GAP9 Docker image to an
Apptainer image once, on a node where you can pull it (or on the laptop, then `scp` the
`.sif`):
```bash
# from the docker image used by gap9_docker (tag it locally first):
apptainer build gap9_sdk.sif docker-daemon://gap9_docker:latest
# or from a registry: apptainer build gap9_sdk.sif docker://<registry>/gap9:<tag>
```
Then everything below runs as `apptainer exec --bind $SCRATCH gap9_sdk.sif <cmd>`. Bind the
`sensei-sdk` and repo dirs so the env vars resolve inside the container.

---

## 2. Flip the platform to GVSOC

`src_GAP/sdk.config` currently has:
```
# CONFIG_PLATFORM_GVSOC is not set
CONFIG_PLATFORM_BOARD=y
```
Change to:
```
CONFIG_PLATFORM_GVSOC=y
# CONFIG_PLATFORM_BOARD is not set
```
Keep `io = host` (CMakeLists) so semihosting prints reach stdout under GVSOC.

---

## 3. Export our model → INT8 tflite (per size)

The autotiler/nntool flow wants the **network forward pass** (backbone + neck + the three
conv detection outputs), **not** the YOLOv5 anchor decode/NMS (anchor-grid/sigmoid/concat
ops that nntool may not support and that we don't need for a latency number).

In `this repo`, export each trained arch-sweep checkpoint
(`runs/sensei_arch/<arm>_<size>_s<seed>/weights/best.pt`, e.g. the deployed `base_320_s0`)
truncated at the three raw conv feature maps. These are **classic anchor-based** YOLOv5
checkpoints, so export with the classic `ultralytics/yolov5` `export.py`:
```bash
# pseudocode for scripts/export_tflite.py (to be added):
#  python "$YOLOV5_REPO/export.py" --weights runs/sensei_arch/base_320_s0/weights/best.pt \
#      --include tflite --imgsz 320 --int8 --data "$DATA/tiled.yaml"
#  # int8 needs a representative calibration set (val tiles) -> yolov5 export reads it via --data
```
Produce `model_320.tflite` (deployed) and the other measured sizes `model_192.tflite`,
`model_512.tflite`. If the tflite export trips nntool, fall back to **ONNX**
(`--include onnx --opset 13`) and use the GAP **ONNX** front-end in nntool — GreenWaves
supports both.

**Channels.** Our model is 3-channel (RGB tiles); the demo CMake sets
`YOLO_INPUT_CHANNEL 3`, so they match. (The monochrome-sensor question is a separate paper
caveat; for the cycle count it's irrelevant — keep 3ch to match the trained weights.)

---

## 4. Autotiler generation → `generated_NxN/`

The committed `src/networks/YOLOv5/generated_128x128/` (model.c, weights_tensors/,
Expression_Kernels.c, modelInfos.h, YOLOv5.h) is the **autotiler output for the nc=1
model**. Regenerate it for ours. The demo's `networks/YOLOv5/YOLOv5_trained_deploy.ipynb`
contains the nntool+autotiler recipe — adapt it:
```
nntool model_320.tflite
  > adjust_model / fusions --scale8        # NE16-friendly INT8 graph
  > set graph_produce_node_names true
  > save_state
  > gen <autotiler model.c>               # emits generated_320x320/
```
Then point CMake at it: `generated_320x320/` selected by `YOLO_MODEL_DIR`.

**Risks & mitigations**
- *Unsupported op (anchor decode/NMS, SiLU/Hardswish, concat):* export truncated at conv
  outputs (Step 3); if a backbone activation isn't fused, swap SiLU→ReLU and **retrain**
  (cheap on Kuma) — ReLU is NE16-native and is what TinyissimoYOLO uses anyway.
- *Tiling/L1 pressure:* our deployed sizes are the Wiese2025-measured 192/320/512 px, all
  characterised on-chip in Table I and within L2; autotiler tiles L1 automatically. If a
  layer won't tile, reduce the autotiler L1 budget knob or the tile cluster count.

---

## 5. Build firmware for our model + sizes

The CMake hardcodes per-size config (`YOLO_INPUT_SIZE` ∈ {64,128,…,512}). Our deployed
sizes **192/320/512 are multiples of 64**, so 512 is already in the list; 192 and 320 just
need a branch each (square tiles):
```cmake
elseif (YOLO_INPUT_SIZE EQUAL 320)
    set(YOLO_INPUT_WIDTH 320)
    set(YOLO_INPUT_HEIGHT 320)   # we trained square 320px tiles
    set(INPUT_IN_L3 0)
    set(DOWNSAMPLE_FACTOR 4)
    math(EXPR ROWS "<grid_rows_for_our_head>")   # see note
```
`ROWS` in the demo = total detection-grid rows; with nc=9 the per-row width is `5 + nc`
(anchor-based) or the Ultralytics layout — but **ROWS/decode only feed `yolo_utils.c`
post-processing, not the NN cycle count.** For the latency/energy number we can leave decode
stubbed and measure the network graph alone. Wire real decode only if we also want
functional on-device boxes (not needed for the paper's hardware number).

Build: `cmake -B build -DYOLO_INPUT_SIZE=320 && cmake --build build`.

---

## 6. Add cycle measurement + run on GVSOC

`main.c` has no perf hooks. Two options (use whichever the autotiler graph exposes):

**A — autotiler graph perf (preferred):** build the model with per-node perf enabled
(`AT_GRAPH_DUMP_TENSOR`/`GRAPH_DUMP` off, but the autotiler `--perf`/`GraphTrace` on). The
generated `RunNetwork` then prints per-layer cycles and a total at the end.

**B — explicit counter around inference:**
```c
pi_perf_conf(1 << PI_PERF_CYCLES);
pi_perf_reset(); pi_perf_start();
RunNetwork();                       // the generated entrypoint (check YOLOv5.h)
pi_perf_stop();
uint32_t cyc = pi_perf_read(PI_PERF_CYCLES);
printf("INFER_CYCLES %u\n", cyc);
```
Run under GVSOC via the SDK run target (with `CONFIG_PLATFORM_GVSOC=y`), capturing stdout:
```bash
cmake --build build --target run 2>&1 | tee gvsoc_320.log
```

---

## 7. Parse → CSV → paper

`latency_ms = cycles / (FREQ_CL_MHz * 1e3)` with `FREQ_CL = 370`. Energy:
`energy_mj = latency_ms * P_active_mW / 1000`, with `P_active` the **measured** GAP9 power
at 370 MHz / 0.8 V from Wiese2025 (so latency is GVSOC-measured, power is silicon-measured —
both honest, no MAC extrapolation).

Add `scripts/gvsoc_parse.py` → `results/metrics/gvsoc.csv` with columns
`imgsz,cycles,latency_ms,energy_mj,freq_mhz,power_mw,provenance` and
`provenance=gvsoc-cycle-accurate`. This is a **cross-check that sits alongside** the measured
Wiese2025 look-up (`insect_gap9.sensei_energy`), not a replacement:
- `make_tables.py`: keep the `measured (Wiese2025 Table I)` energy as the headline; report
  the GVSOC cycle count beside it as an independent confirmation (camera-ready / appendix).
- `make_figures.py`: optionally overlay the `gvsoc.csv` point on the accuracy–energy figure
  when available, to show look-up and cycle-accurate sim agree.
- `main.tex`: add (do not replace) a sentence noting the measured look-up was cross-checked
  by a cycle-accurate GVSOC simulation of the deployed model.

**Validate the parser against the first real `gvsoc_320.log`** — the autotiler perf print
format is version-dependent; do not trust a regex until you've seen one run.

---

## 8. Automation (the cluster)

Once Steps 3–6 work once by hand, automate the sweep. Skeleton sbatch (mirrors the existing
`scripts/slurm/*.sbatch` conventions; GVSOC is CPU-only so request **CPU**, no `--gres`):
```bash
#!/usr/bin/env bash
#SBATCH --job-name=insect-gvsoc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
SIF=$SCRATCH/gap9_sdk.sif
for SZ in 192 320 512; do
  apptainer exec --bind "$SCRATCH" "$SIF" bash -lc "
     cd src_GAP &&
     sed -i 's/^# CONFIG_PLATFORM_GVSOC.*/CONFIG_PLATFORM_GVSOC=y/' sdk.config &&
     cmake -B build_$SZ -DYOLO_INPUT_SIZE=$SZ &&
     cmake --build build_$SZ --target run 2>&1 | tee gvsoc_$SZ.log"
done
python scripts/gvsoc_parse.py gvsoc_*.log   # -> results/metrics/gvsoc.csv
```
(Autotiler generation, Step 4, is also containerized — run it as a prior step or commit the
`generated_NxN/` for our model once and reuse.)

---

## 9. Where this fits in the current pipeline

The headline results already stand on their own without GVSOC:
- **Energy/latency** is the measured Wiese2025 Table I look-up
  (`insect_gap9.sensei_energy.measured_at_size`, op-identical architecture).
- **Variance** is covered by the 3-seed arch sweep (`scripts/slurm/47_sensei_arch_sweep.sbatch`
  → `scripts/collect_sensei_sweep.py`, mean±std).
- **The loss ablation** (base / focal / nwd / focal_noiw) and its significance
  (`scripts/sig_test.py` → `significance.csv`) are the headline finding.
- **Counting robustness** is reported per-species (`49_sensei_eval.sbatch` →
  `insect_gap9.monitor_metrics` / `insect_gap9.confusion`, count-error figure).

So GVSOC is purely an **independent confirmation** of the measured energy number for the
camera-ready/appendix, and a stepping stone toward a fully on-board power trace (INA219,
`provenance="measured"`) — the only remaining upgrade beyond the silicon-grounded look-up.
