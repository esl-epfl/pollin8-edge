#!/usr/bin/env python3
"""Pick the best-micro-F1 seed from monitor_metrics CSVs written by 48_base_fullval_seeds.sbatch.

    python scripts/pick_best_seed.py base_320_fv
reads results/metrics/sensei_<prefix>_s*.csv (each a monitor_metrics row) and prints the ranking
plus the winner, comparable to the reference results/metrics/sensei_base_320_s0.csv (0.8171).
"""
import sys, csv, glob, os

prefix = sys.argv[1] if len(sys.argv) > 1 else "base_320_fv"
rows = []
import re
for f in sorted(glob.glob(f"results/metrics/sensei_{prefix}_s*.csv")):
    if not re.search(r"_s\d+\.csv$", f):   # skip _conf_sweep.csv / _per_species.csv
        continue
    try:
        r = next(csv.DictReader(open(f)))
        rows.append((f, float(r["micro_f1"]), r))
    except Exception as e:  # noqa: BLE001
        print("skip", f, e)
if not rows:
    print("no results yet for prefix", prefix)
    sys.exit(1)
rows.sort(key=lambda x: -x[1])
print(f"{'seed csv':44s} micro_f1  prec   recall  count_mae  conf")
for f, f1, r in rows:
    print(f"{os.path.basename(f):44s} {f1:.4f}   {r.get('precision','?'):>5}  "
          f"{r.get('recall','?'):>5}  {r.get('count_mae_img','?'):>6}   {r.get('conf','?')}")
best = rows[0]
print(f"\nBEST: {os.path.basename(best[0]).replace('sensei_','').replace('.csv','')}  "
      f"micro_f1={best[1]:.4f}   (reference sensei_base_320_s0 = 0.8171)")
