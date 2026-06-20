# Reproducibility (FAIR)

This artefact reproduces the paper's numbers from code: one script per number, one CSV per
result, generators that turn CSVs into the manuscript's tables/figures. Every metric row carries
a `provenance ∈ {measured, simulated}` column and we never relabel a measured number as simulated.

**What the paper deploys.** An anchor-based **YOLOv5p** (`configs/yolov5p_sensei.yaml`, ~0.31 M
params) that is **operator- and size-identical** to the GAP9-silicon-characterised HeadCount
people-counter (cite `Weise2025SENSEI`). It is trained **from scratch** with the **classic
`ultralytics/yolov5`** repo on the public Bjerge et al. (2022) 9-species benchmark
(Zenodo `10.5281/zenodo.7395752`), **tiled** to 320 px crops, quantised **INT8**, and run on the
SENSEI/GAP9 ULP node. Because the network is operator-identical to the characterised model,
**latency/energy/power are a direct measured look-up** from Wiese et al. 2025 Table I
(`src/insect_gap9/sensei_energy.py`) — not GVSOC-simulated. GVSOC (`docs/gvsoc_deployment.md`) is
an optional cycle-accurate cross-check, not the primary energy source.

**Headline finding.** A controlled A/B/C/D loss ablation shows class re-balancing *hurts*: focal
loss + `--image-weights` + NWD box loss collapse the common honeybee onto its rare Batesian mimic
(drone fly). The **deployed recipe is the plain baseline** (standard BCE, no re-balancing): mAP@.5
≈ 0.83 @320 px (mean of 3 seeds), honeybee F1 ≈ 0.87, near-unbiased aggregate count. The baseline
lead is significant (base vs the focal recipes p<0.01, vs NWD p<0.03; `scripts/sig_test.py`).

## Reproduce the paper end-to-end

The heavy steps (tiling, the sweep, the monitoring eval) run on a SLURM cluster — see
`docs/training_kuma.md` for the Kuma cluster setup, partitions, and the classic-`yolov5` clone. The paper
artefacts (tables/figures/PDF) build wherever TeX is installed. After
`source scripts/slurm/config.sh && source "$VENV/bin/activate"`:

```bash
# 0. environment (login node) — venv + INT8/quant deps; classic yolov5 clone:
pip install -r requirements.txt
git clone https://github.com/ultralytics/yolov5 "$WORK/yolov5"   # classic repo (anchor-based head)

# 1. data: download+verify (md5) the Zenodo benchmark, unzip into a 9-class split, then TILE to 320 px
python -m insect_gap9.download_data --dest "$DATA/raw"               # Bjerge 2022, md5-verified
python -m insect_gap9.prepare_data  --raw "$DATA/raw" --out "$DATA/insects1201"  # -> train/val/test + auto.yaml
sbatch --account=$ACCOUNT scripts/slurm/10_data_tile.sbatch         # -> $DATA/{train,val,test}_tiled + tiled.yaml

# 2. the SENSEI-architecture sweep: 4 arms (base/focal/nwd/focal_noiw) x 3 sizes (192/320/512)
#    x 3 seeds = 36 tasks; classic yolov5; resumable + MIG-parallel.
#    Build the 1,500-image fast-val subset first (tiled_fastval.yaml — see the sbatch header):
sbatch --account=$ACCOUNT --partition=mig12gb --array=0-35%16 scripts/slurm/47_sensei_arch_sweep.sbatch

# 3. collect the sweep -> mAP±std + MEASURED energy (Wiese2025 Table I) per (arm,size):
python scripts/collect_sensei_sweep.py --runs "$RUNS/sensei_arch"   # -> results/metrics/sensei_arch_sweep.csv

# 4. full monitoring eval of the DEPLOYED base@320 (centre-matched F1, counting, confusion):
RUN=base_320_s0 sbatch --account=$ACCOUNT --partition=l40s scripts/slurm/49_sensei_eval.sbatch
#    -> results/metrics/sensei_base_320_s0{,_per_species,_conf_sweep}.csv
#       results/figures/confusion_base_320_s0.pdf

# 5. significance of the ablation (base vs each re-balancing arm, n=3 seeds):
python scripts/sig_test.py                                       # -> results/metrics/significance.csv

# 6. paper artefacts (where TeX is installed): regenerate -> stage -> compile -> page/ref report
python scripts/make_values.py     # results/tables/values.tex   (inline \val macros)
python scripts/make_tables.py     # ablation + monitoring + complexity tables
python scripts/make_figures.py    # accuracy_energy, ablation, count_error
scripts/build_paper.sh            # all of the above + latexmk + checks (--submission for the 4+1 budget)
```

`scripts/build_paper.sh` regenerates values/tables/figures from `results/metrics/*.csv`, stages
them into the (private, gitignored) LaTeX tree, runs `latexmk`, and reports page count, undefined
refs/cites, and remaining DRAFT placeholders. Pass `--submission` for an annex-free PDF to check
the 4+1-page budget, `--compile-only` to skip regeneration.

## What each number traces to

The "deployed size" used by the generators is data-driven: the base row with the highest
`map50_mean` in `sensei_arch_sweep.csv` (320 px in the paper), so `make_values`/`make_tables`/
`make_figures` all key off the same row.

| Paper element | Command | Artefact |
|---|---|---|
| `tab:ablation` (loss A/B/C/D + honeybee collapse) | `make_tables.py` | `results/tables/ablation.tex` |
| `tab:monitoring` (per-species F1 / count / R²) | `49_sensei_eval` → `make_tables.py` | `results/tables/monitoring.tex` |
| `tab:complexity` (params / MACs / fits-L2) | `make_tables.py` | `results/tables/model_complexity.tex` |
| Inline `\val` macros (F1, recall, count err.) | `make_values.py` | `results/tables/values.tex` |
| Fig. accuracy–energy frontier (base, measured) | `make_figures.py` | `results/figures/accuracy_energy.pdf` |
| Fig. build-up ablation (F1 / honeybee / bias) | `make_figures.py` | `results/figures/ablation.pdf` |
| Fig. per-species count error | `make_figures.py` | `results/figures/count_error.pdf` |
| Confusion matrix | `49_sensei_eval` | `results/figures/confusion_<run>.pdf` |
| mAP±std per (arm,size) | `collect_sensei_sweep.py` | `results/metrics/sensei_arch_sweep.csv` |
| measured latency/energy/power | `insect_gap9.sensei_energy` (in the above) | the `energy_mj` / `latency_ms` / `gap9_power_mw` columns of `sensei_arch_sweep.csv` |
| per-species F1 / counting / R² | `49_sensei_eval` | `results/metrics/sensei_base_320_s0{,_per_species}.csv` |
| ablation p-values | `sig_test.py` | `results/metrics/significance.csv` |

The energy columns of `sensei_arch_sweep.csv` carry
`provenance = "measured (Wiese2025 Table I; op-identical architecture)"`. These are a **direct
table read-off**, valid because `yolov5p_sensei.yaml` is operator- and size-identical to the
characterised HeadCount model — not a MAC interpolation. Never relabel them as simulated.

## Determinism

- **Seeds:** each sweep task is trained at `--seed ∈ {0,1,2}` (3 seeds per arm/size for the
  variance answer); `WANDB_MODE=disabled`, fixed train/val/test tiling.
- **Resumability:** `47_sensei_arch_sweep` is requeue-safe (config-hash guard + per-task flock +
  `last.pt` every epoch); a SLURM requeue continues from the last completed epoch, never restarts.
- **Versions pinned** in `requirements.txt`; record `pip freeze > results/metrics/env.lock` and the
  git SHA alongside results so a run is fully described.
- **Data integrity:** md5 sums verified on download (`insect_gap9.download_data`).
- **Energy is closed-form:** `sensei_energy.measured_at_size` is a deterministic table look-up.

## Licence & provenance

Code is **Apache-2.0**; documentation, figures and tables are **CC-BY-4.0**. The Bjerge et al.
(2022) benchmark (Zenodo `10.5281/zenodo.7395752`) is **CC-BY-4.0** and is **not** redistributed
here — `insect_gap9.download_data` fetches it from the original source. See `LICENSE`. Every results
CSV carries a `provenance ∈ {measured, simulated}` column; we never overwrite a measured CSV with a
simulated one.
