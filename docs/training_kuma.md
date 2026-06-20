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

## Useful job commands
```bash
squeue -u $USER                 # queue (PD pending, R running; gone = done)
tail -f *-*.out                 # live logs
sacct -j <id> --format=JobID,JobName,State,Elapsed,ExitCode
scancel -u $USER                # cancel all my jobs
```
