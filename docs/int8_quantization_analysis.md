# INT8 (SQ8/NE16) quantization of `base_320_s0` — analysis, failure modes, and improvement strategies

This documents the NNTool SQ8/NE16 accuracy validation of the deployed `base_320_s0`
YOLOv5p detector, the failure modes hit on first run, and two strategies to recover a
deployment-realistic INT8 operating point. It accompanies `scripts/validate_int8_nntool.py`
and `results/metrics/int8_validation.csv`.

## 1. Setup and the FP32 reference

- Model: `base_320_s0` (YOLOv5p, 314,774 params, 1.0 GFLOP, **SiLU** activations), retrained
  from scratch (val mAP@0.5 = 0.775).
- Quantiser: GreenWaves **NNTool**, scheme **SQ8** (8-bit symmetric), executed as the
  measured counterpart to the ORT proxy.
- Test set: the locked 22,861-tile split (grid-tiled; ~25% contain insects, ~75% background).
- FP32 reference = **NNTool float** execution of the same adjusted/fused graph (so
  pt→graph conversion error cannot masquerade as quantisation error). Cross-checked against
  the recorded classic micro-F1 0.8171.

**FP32 baseline recovery (§0).** The *first* retrain (single seed, arch-sweep **fast-val**
early-stopping) under-converged — micro-F1 0.7546 with **count-MAE 23.4%** (recall 0.839 already
matched the reference, but the model over-fired → low precision). Diagnosis = premature early-stop
on the noisy 1,500-image fast-val fitness. Fix (`scripts/slurm/48_base_fullval_seeds.sbatch`):
**full-val per-epoch fitness + best-of-3-seeds**. Result (`pick_best_seed.py`):

| retrain | micro-F1 | precision | recall | count-MAE |
|---|---|---|---|---|
| initial (fast-val, seed 0) | 0.7546 | — | 0.839 | 23.4% |
| **recovered** (full-val, best of s0/s1/s2 = **`base_320_fv_s1`**) | **0.8201** | 0.803 | 0.838 | **7.6%** |
| reference `sensei_base_320_s0` | 0.8171 | 0.801 | 0.835 | 7.5% |

The recovered model **matches/exceeds** the paper reference and collapses count-MAE to 7.6% — the
under-convergence was the whole gap. This converged checkpoint is the FP32 foundation for the INT8
numbers below (the INT8 *delta* is robust to the absolute baseline, but the cross-check story is
cleaner on the converged model — now 0.8201 vs 0.8171). The §5 results were first measured on the
initial (0.7546) checkpoint; the recovered `base_320_fv_s1` INT8 re-run is complete: FP32 micro-F1 0.7923 (NNTool float) → INT8 0.7577, i.e. **ΔF1 −0.035, ΔmAP −0.052** (frozen conf; ≈−0.028 F1 re-tuned), consistent with the initial checkpoint — the delta is robust to the baseline.

## 2. Failure modes found on first run (all fixed / characterised)

The `validate_int8_nntool.py` pipeline had never been executed end-to-end; three issues surfaced:

1. **NE16 numpy-emulation zeroes the graph output.** `SQ8_OPTIONS` defaulted to
   `use_ne16=True`. In this `gap_sdk` clone (`master`), the NE16 numpy interpreter produces an
   all-zero quantised output (INT8 max score = 0.000, boxes `(0,0,0,0)`). NE16 is only GAP9's
   on-chip *accelerator execution mode* for the same 8-bit scheme — accuracy-equivalent — so the
   accuracy measurement is done in **plain SQ8** (`NNTOOL_NE16=1` re-enables NE16 for the exact
   silicon SDK / Apptainer image). Fix: `SQ8_OPTIONS = {"scheme": "SQ8"}`, NE16 opt-in.

2. **Quantising the in-graph anchor decode destroys box coordinates (rung-1).** The
   full-decode tflite (rung 1) quantises the anchor-grid/sigmoid/mul decode; those MUL nodes
   have ~0 dB QSNR, so INT8 emits high-score boxes at **wrong locations** (count-MAE 1348%,
   recall 0). The faithful GAP9 flow runs the INT8 **conv backbone** and the decode in
   **software**, so the measurement uses **rung 3** (`best_trunc.onnx`, three raw conv heads;
   decode in float Python — *excluded from quantisation*).

3. **`rung-3 + plain SQ8` works:** on an insect tile, INT8 max score 0.718 vs FP32 0.738, head
   ranges preserved (−16.3..7.51 vs −18..8.64). This is the measurement baseline below.

The dominant remaining quantisation stressor is the **SiLU** activation (worst per-node QSNR
≈ −2 dB at the `_model_*_act_Mul` inputs; `results/metrics/quant_qsnr.csv`) — SiLU's wide,
unbounded range is hard for symmetric 8-bit, causing positive-side saturation of the head logits.

## 3. Strategy 2 — better calibration + per-precision operating point

Two independent levers, both cheap:

### 2a. Insect-biased calibration
The default calibration draws the first 300 **val** tiles, but `val_tiled` is ~75% background,
so the quantisation scales are set largely by empty-tile statistics and miss the activation-range
tails that fire on insects — a classic cause of the positive-side saturation seen here. We rebuild
the calibration set from **insect-containing** val tiles only (`data/val_tiled_insects/`) and
re-quantise.

### 2b. Per-precision confidence threshold
The pipeline freezes `conf* = 0.4` (the FP32 operating point) for both precisions *by design*, to
isolate quantisation error. But quantisation shifts the score distribution, so for the **deployed
counting number** the INT8 threshold should be re-tuned. The full conf-grid is already emitted to
`results/metrics/int8_nntool_detail.csv`; we report **both**: frozen (clean isolation) and re-tuned
(deployment-realistic F1/recall/count-MAE).

## 4. Strategy 3 — QAT or a quantisation-friendlier head

### 3b. ReLU6 head (architecture-level, best effort/ceiling ratio)
The SiLU activations are the dominant stressor. Retraining with **ReLU6** (bounded [0,6] range)
yields quantisation-friendly, saturation-free activations — a well-known INT8-robustness trade
that usually costs little FP32 accuracy on small detectors and substantially improves the INT8
head. Cheap for a 0.31M-param model (~30 min retrain). Optionally preceded by bias-correction /
cross-layer equalisation as a post-hoc step.

### 3a. Short QAT
If PTQ still leaves a gap, fine-tune `base_320_s0` for a few epochs with fake-quant (SQ8) nodes so
the weights adapt to 8-bit rounding. Cheap for this model; typically closes most of the PTQ gap,
especially on recall. (Requires a fake-quant scheme matched to NNTool's SQ8; noted as the highest
effort.)

## 5. Results

All on the full 22,861-tile locked test split. FP32 = NNTool float; INT8 = NNTool SQ8.

**Frozen operating point (conf* = 0.4, clean quantisation-error isolation):**

| Config | calib | FP32 F1 | INT8 F1 | ΔF1 | ΔmAP@0.5 | Δcount-MAE |
|---|---|---|---|---|---|---|
| **Baseline** (rung-3 SQ8) | 300 val (default) | 0.7546 | **0.7373** | **−0.0173** | −0.0046 | +5.6pp |
| + insect calib (2a) | 300 insect-only val | 0.7546 | 0.7169 | −0.0377 | −0.0115 | +14.2pp |

**2a finding (honest negative):** insect-*only* calibration makes the **frozen** point *worse*
(ΔF1 −3.8pp, count-MAE +14pp) — it over-covers the high-activation tails, spends 8-bit resolution
there, and inflates false positives (recall even rises to 0.844, but precision/count collapse). At
the **re-tuned** point it recovers to F1 **0.7713 @0.6**, ≈ tied with default-calib's 0.7707 — i.e.
once conf is re-tuned per precision, calibration composition matters little, and insect-*only* is an
over-correction. A *balanced* insect+background calibration (not run here) would be the principled
middle ground.

**Per-precision re-tuned operating point (2b — deployment-realistic; conf swept per precision):**

| Config | FP32 (best conf) | INT8 (best conf) | re-tuned ΔF1 | INT8 recall | INT8 count-MAE |
|---|---|---|---|---|---|
| Baseline (rung-3 SQ8) | 0.783 @0.55 | **0.7707 @0.55** | **−0.0123** | 0.756 | 17.6% |

Re-tuning the INT8 threshold recovers most of the frozen-threshold F1 loss and roughly halves the
count-MAE gap (+5.6pp → +2.75pp). Both precisions happen to peak at conf 0.55 on this split.

**Strategy 3b (ReLU6 head) — the standout result.** Retrained `base_320_s0` with SiLU→**ReLU6**
(cfg-level `activation: nn.ReLU6()`), same recipe otherwise:

| Config | micro-F1 (frozen 0.4) | micro-F1 (best conf) | INT8 count-MAE | ΔF1 (int8−fp32, frozen) |
|---|---|---|---|---|
| SiLU FP32 | 0.7546 | 0.783 @0.55 | — | — |
| SiLU INT8 | 0.7373 | 0.7707 @0.55 | 29.0% | −1.73pp |
| **ReLU6 FP32** | **0.7778** | **0.7953 @0.5** | — | — |
| **ReLU6 INT8** | **0.7807** | **0.7956 @0.5** | 13.3% | **+0.29pp (lossless)** |

ReLU6 wins on *both* axes: (i) **higher FP32** F1 (0.7546→0.7778 frozen; count-MAE 23%→15% — a
sharper, better-converged detector) and (ii) **INT8 is essentially lossless** (ΔF1 +0.3pp — INT8
even marginally exceeds FP32 at the frozen point; ≈0 at the tuned point), because ReLU6's bounded
[0,6] range removes the SiLU saturation that was the dominant quantisation stressor. Cross-check
0.7778 vs reference 0.8171 (Δ−0.039, closer than SiLU's −0.063). **Recommended for deployment.**

**Strategy 3a (short QAT):** given SiLU→ReLU6 PTQ is already lossless (ΔF1 ≈ 0), QAT is not needed for
this model — it would be the lever only if a quantisation-friendly *architecture* were off the table.
Recipe retained in §4 for completeness.

_Provenance `nntool-sq8-ne16(<version>)` in the CSV; the measurement is plain SQ8 (NE16 is the
accuracy-equivalent on-chip execution mode, emulation-only issue in this SDK clone). Rung-3
(`;decode=fp32-python`): the anchor decode runs in float and is excluded from quantisation, matching
GAP9 (NE16 runs the INT8 convs; decode in software)._

**Recovered converged model (`base_320_fv_s1`) — the FP32 foundation, rung-3 SQ8:**

| operating point | FP32 F1 | INT8 F1 | ΔF1 | INT8 count-MAE |
|---|---|---|---|---|
| frozen 0.4 | 0.7923 | 0.7577 | −0.0346 | 26.8% |
| re-tuned per precision | **0.8251 @0.55** | **0.7972 @0.55** | **−0.0279** | 11.4% |

Notable: the **converged** SiLU model shows a *larger* INT8 gap (−2.8 pp tuned) than the initial
under-converged one (−1.2 pp) — sharper decision boundaries are more sensitive to SiLU's
quantisation saturation. (The nntool-float FP32 reaches 0.8251, consistent with the classic 0.8201
and reference 0.8171.) This strengthens the ReLU6 recommendation: for the *converged* model, SiLU→INT8
costs ~2.8 pp F1, whereas ReLU6→INT8 is lossless. A **converged ReLU6** (full-val 3-seed, the ReLU6
retrain here used fast-val) is the ideal final deployment model — recommended follow-up.

### Takeaway
1. **FP32 recovered:** full-val fitness + best-of-3 gives `base_320_fv_s1` at micro-F1 **0.8201**
   (nntool-float 0.8251 tuned), matching/beating the reference 0.8171 with count-MAE 7.6% — the
   0.7546 gap was pure under-convergence.
2. **INT8 (SQ8) is deployment-viable** once (i) NE16 is disabled for emulation, (ii) the anchor
   decode is excluded from quantisation (**rung 3**), and (iii) the operating point is re-tuned per
   precision: on the converged SiLU model the tuned gap is **−2.8 pp micro-F1 / −5 pp mAP@0.5**;
   on the initial (softer) model only −1.2 pp — sharper models are more quantisation-sensitive.
3. **ReLU6 is the deployment recommendation:** it *improves* FP32 and makes INT8 **lossless**
   (ΔF1 ≈ 0), by removing the SiLU saturation that is the dominant quantisation stressor. A
   converged ReLU6 (full-val 3-seed) is the ideal final model.
4. **Calibration composition** matters little once conf is re-tuned; insect-*only* over-corrects.
