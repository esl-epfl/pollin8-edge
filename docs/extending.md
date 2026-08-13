# Extending & improving

## New species / custom taxonomy
1. Assemble a YOLO-format dataset for the new species (co-located `images/` + `labels/`,
   one box file per image).
2. Tile it so small insects are well-sized: `python -m insect_gap9.tile --src <split>
   --out <split>_tiled --mode object --tile 320` for train, `--mode grid --stride 320`
   for val/test (see `scripts/slurm/10_data_tile.sbatch`), then write a `tiled.yaml`
   pointing `train`/`val`/`test`/`nc`/`names` at the new tiles.
3. Update `nc` in `configs/yolov5p_sensei.yaml` (only the final 1×1 head changes; the
   backbone/neck and the Wiese2025 energy look-up are unaffected while `nc` stays small).
4. Retrain with the classic-yolov5 arm of `scripts/slurm/47_sensei_arch_sweep.sbatch`
   (or its single-task `train.py --cfg configs/yolov5p_sensei.yaml --weights yolov5n.pt`
   invocation). Effort: hours, not field seasons.

## Push accuracy up
- Quantisation-aware training (QAT) instead of PTQ → recover INT8 mAP.
- Re-tune tiling (`--tile`, `--stride`, `--vis`, `--bg-frac` in `tile.py`) — denser
  object tiles and more background coverage trade dataset size for recall on tiny insects.
- Larger input within the measured set: the sweep already trains 192/320/512 px; if a
  size fits L1 after neck pruning you can add it to `SIZES` in `47_sensei_arch_sweep.sbatch`
  (keep it a Wiese2025-measured multiple of 64 so energy stays a direct look-up).

## Push energy down
- NE16 mapping coverage: ensure all depthwise/pointwise convs dispatch to NE16.
- Lower cluster frequency with voltage scaling; trade latency for energy.
- Longer duty-cycle period or motion-gated wake to cut average power.

## Loss-recipe ablations
The deployed recipe is the plain `base` arm; `47_sensei_arch_sweep.sbatch` also runs
`focal` (B), `nwd` (C) and `focal_noiw` (D). To extend the ablation, add an arm in the
`ARMS`/`case` block (new `configs/hyp_sensei_*.yaml` + any `--image-weights`/patch flag),
rerun the array, then `python scripts/collect_sensei_sweep.py` and
`python scripts/sig_test.py` to refresh `sensei_arch_sweep.csv` and `significance.csv`.

## Add the real-board measurement track
Energy today is a **measured** Wiese2025 Table I look-up (`insect_gap9.sensei_energy`,
provenance `measured (Wiese2025 Table I; op-identical architecture)`). To add a
*this-model* board trace, emit rows with `provenance="measured"` against an INA219; the
generators already consume the same CSV schema, so no other change is needed. For a
cycle-accurate cross-check between look-up and silicon, see `docs/gvsoc_deployment.md`
(an optional `results/metrics/gvsoc.csv` with `provenance=gvsoc-cycle-accurate`).

## Regenerate paper artefacts
After any rerun: `python scripts/make_values.py`, `python scripts/make_tables.py`,
`python scripts/make_figures.py` (or `scripts/build_paper.sh` end-to-end). Each reads
`results/metrics/*.csv` and degrades gracefully when a CSV is absent.

