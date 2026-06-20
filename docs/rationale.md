# Design rationale, options & trade-offs

## The governing constraint
GAP9 has **1.5 MB L2 SRAM + 64 KB L1/core** and no DRAM. That single fact prunes
the design space more than any accuracy target: any model + its activation
high-water mark must fit on-chip after INT8, or pay prohibitive tiling/latency.

## Model choice — operator-identical to a silicon-characterised net
| Option | Params | Fits GAP9? | Verdict |
|---|---|---|---|
| **YOLOv5p** (anchor-based, `depth/width=0.10`) | ~0.31 M | yes | **chosen** — see below |
| YOLOv5n @640 | 1.9 M | no (activations) | rejected: tiling latency |
| MobileNetV3-SSD | 3–5 M | no | rejected: memory |
| Two-stage detect+classify | >10 M | no | rejected: memory + latency |

The decisive reason for **YOLOv5p** is not just that it fits: it is **operator- and
size-identical** to the HeadCount people-counter that Wiese et al. (cite
`Weise2025SENSEI`) characterised on GAP9 *silicon*. `configs/yolov5p_sensei.yaml` is
reconstructed from the committed HeadCount tflite (backbone widths 8/16/32/56/104 =
`width_multiple=0.10`; all C3 collapse to `n=1`; SPPF; PANet neck; three anchor-based
`Detect` heads). The only change is the final 1×1 head, `nc: 1 -> 9` (18 -> 42
channels, <1 % of MACs). Because the network is byte-for-byte the same compute graph
on the same engine, **the measured Wiese2025 Table I latency/energy transfers directly**
to our detector (see "Energy provenance" below). That is what makes the deployment
figures defensible without re-running silicon.

We accept a small mAP gap versus heavier reference detectors because biodiversity
*abundance* estimation aggregates over many frames, so per-frame error averages out.

## Starting weights & training schedule — why from scratch
We train **from scratch** with the **classic `ultralytics/yolov5` repo** (v7.0
anchor-based `train.py`), `cfg=configs/yolov5p_sensei.yaml`, on the public Bjerge et al.
(2022) 9-species benchmark (Zenodo `10.5281/zenodo.7395752`), tiled to 320 px crops.

There is no usable pretrained initialisation for this net: at `width_multiple=0.10`
the channel counts (8/16/32/56/104) do not match any released YOLOv5 checkpoint, so
`yolov5n.pt` weights transfer almost nothing into the backbone (the `--weights yolov5n.pt`
arg in `47_sensei_arch_sweep.sbatch` only seeds whatever shapes happen to align; the
backbone is effectively random). Freezing such a backbone would freeze random weights
and the model cannot learn. We therefore train end-to-end. From-scratch tiny detectors
need a generous epoch budget, so we set a high upper bound (`--epochs 150`) and use
**early stopping** (`--patience 30`) on validation mAP@0.5, reporting the best-val
checkpoint rather than a fixed epoch count.

## Input resolution & tiling
Bjerge frames are large and insects occupy few pixels; resizing a whole frame to a
small input makes them sub-pixel and the detector learns nothing. Two fixes, both
implemented:
- **Tiling** (`src/insect_gap9/tile.py`, driven by `scripts/slurm/10_data_tile.sbatch`):
  crop **object-centric** tiles (one per insect, jittered) for training and overlapping
  **grid** tiles for val/test, all at **320 px**, so each insect is a sizeable fraction
  of a small tile. This is the DSORT-MCU strategy and is what lets a small-input model
  actually see insects.
- **Resolution sweep**: we do not assert one input size. The arch sweep trains the
  **measured** input sizes 192 / 320 / 512 (multiples of 64, all in Wiese2025 Table I),
  so every point on the accuracy–energy curve has a *direct* measured energy and all
  sizes fit on-chip and stay milliwatt-class.

## Energy provenance — measured look-up, not simulation
Energy/latency/power are a **direct measured look-up** from Wiese2025 Table I via
`src/insect_gap9/sensei_energy.py`: `measured_at_size(px)` returns the silicon-measured
per-inference cost at a measured input size with no interpolation, valid precisely
because the network is operator-identical (`estimate(macs)` is only the interpolation
fallback for off-grid sizes). `collect_sensei_sweep.py` attaches these measured numbers
to the accuracy table, and every results CSV carries a `provenance` column
(`measured (Wiese2025 Table I; op-identical architecture)`).

GVSOC (`docs/gvsoc_deployment.md`) is an **optional cycle-accurate cross-check** of this
measured number, not the primary energy source — a camera-ready nicety, never relabelled
as the headline figure.

## Loss-ablation: why the deployed recipe is the plain baseline
The deployed recipe is the **baseline** arm: standard BCE objectness/classification loss,
**no class re-balancing**. This is a *result*, not a default. The arch sweep
(`scripts/slurm/47_sensei_arch_sweep.sbatch`) runs a controlled A/B/C/D loss ablation,
4 arms × 3 sizes × 3 seeds:
- **A `base`** — standard loss, no re-balancing (deployed).
- **B `focal`** — focal loss + `--image-weights` inverse-frequency resampling.
- **C `nwd`** — B plus a Normalized Wasserstein Distance tiny-object box loss
  (`scripts/patch_yolov5_nwd.py`, hyp-gated so it no-ops for A/B).
- **D `focal_noiw`** — B *minus* `--image-weights`, the control that isolates
  resampling from focal+copy-paste.

**Headline finding: class-balancing HURTS.** Up-weighting the rare classes collapses
the common honeybee onto its rare Batesian mimic (the drone fly), wrecking both the
honeybee F1 and the aggregate count. The baseline wins decisively:
**mAP@.5 ≈ 0.83 @320 px** (mean of 3 seeds), honeybee **F1 ≈ 0.87**, near-unbiased
aggregate count. `scripts/sig_test.py` (Welch's t-test over the 3 seeds in
`sensei_arch_sweep.csv`) confirms the lead is real: base vs the focal recipes **p<0.01**,
base vs NWD **p<0.03**.

## Test-split label handling
The Zenodo **test** split adds coarse classes 9–14. For a like-for-like 9-class
evaluation we keep train/val/test on the same 9 species; the locked test mAP is scored
on the full tiled test split via the classic `val.py` (`--task test`).

## Quantisation
Post-training INT8 (NE16-friendly, symmetric calibration on val tiles). We use PTQ not
QAT to keep the pipeline a thin wrapper; QAT is a documented future improvement.

## Why a CSV-in-the-middle architecture
Decoupling "produce numbers" (scripts → `results/metrics/*.csv`) from "render numbers"
(`make_values.py` / `make_tables.py` / `make_figures.py` → `results/tables`,
`results/figures`) means the LaTeX never contains a hand-typed value, figures regenerate
deterministically, and a reviewer can diff CSVs across runs. Every row is tagged
`provenance ∈ {measured, simulated}`; a measured row is never relabelled as simulated.
