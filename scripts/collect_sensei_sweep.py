"""Aggregate the SENSEI-architecture sweep (47_sensei_arch_sweep.sbatch) into one table.

Reads each run's yolov5 `results.csv` (written by the classic train.py) for its best validation
mAP, groups by (arm, size) across seeds to get mean +/- std (the variance answer), and attaches
the DIRECT measured per-inference latency/energy at that input size from Wiese2025 Table I via
insect_gap9.sensei_energy.measured_at_size (valid because the arch is op-identical). Output:
results/metrics/sensei_arch_sweep.csv -- the accuracy(+/-std) vs measured-energy Pareto, with an
A/B/C ablation column. Pure parsing + stats; no torch, fast, unit-tested.
"""
from __future__ import annotations
import argparse, csv, math, re
from pathlib import Path

from insect_gap9 import sensei_energy as SE

# focal_noiw listed before focal so the longer arm name wins the alternation.
RUN_RE = re.compile(r"(?P<arm>base|focal_noiw|focal|nwd)_(?P<size>\d+)_s(?P<seed>\d+)$")


def _best_map(results_csv: Path) -> dict | None:
    """Best (max) validation mAP@0.5 and mAP@0.5:0.95 over epochs from a yolov5 results.csv.
    yolov5 pads column names with spaces, so headers are stripped."""
    rows = list(csv.DictReader(results_csv.open()))
    if not rows:
        return None
    def col(row, key):
        for k, v in row.items():
            if k.strip() == key:
                return float(v)
        return float("nan")
    m50 = [col(r, "metrics/mAP_0.5") for r in rows]
    m5095 = [col(r, "metrics/mAP_0.5:0.95") for r in rows]
    m50 = [x for x in m50 if not math.isnan(x)]
    m5095 = [x for x in m5095 if not math.isnan(x)]
    if not m50:
        return None
    return {"map50": max(m50), "map5095": max(m5095) if m5095 else float("nan")}


def _mean_std(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / n) if n > 1 else 0.0
    return mu, sd


def collect(runs_dir: Path) -> list[dict]:
    by_group: dict[tuple, list[dict]] = {}
    for results in sorted(runs_dir.glob("*/results.csv")):
        m = RUN_RE.search(results.parent.name)
        if not m:
            continue
        best = _best_map(results)
        if best is None:
            print(f"[warn] no mAP rows in {results}; skipping")
            continue
        by_group.setdefault((m["arm"], int(m["size"])), []).append(best)

    rows = []
    for (arm, size), runs in sorted(by_group.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        mu50, sd50 = _mean_std([r["map50"] for r in runs])
        mu5095, _ = _mean_std([r["map5095"] for r in runs if not math.isnan(r["map5095"])])
        e = SE.measured_at_size(size)            # direct measured readoff (op-identical arch)
        rows.append(dict(
            arm=arm, imgsz=size, n_seeds=len(runs),
            map50_mean=round(mu50, 4), map50_std=round(sd50, 4),
            map5095_mean=round(mu5095, 4),
            macs_M=e["macs_M"], latency_ms=e["latency_ms"], energy_mj=e["energy_mj"],
            gap9_power_mw=e["gap9_power_mw"], energy_provenance=e["provenance"]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs/sensei_arch",
                    help="dir holding <arm>_<size>_s<seed>/results.csv")
    ap.add_argument("--out", default="results/metrics/sensei_arch_sweep.csv")
    a = ap.parse_args(argv)
    rows = collect(Path(a.runs))
    if not rows:
        raise SystemExit(f"[collect] no runs parsed under {a.runs}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"[done] {len(rows)} (arm,size) groups -> {a.out}")
    for r in rows:
        print(f"  {r['arm']:>5} @ {r['imgsz']:>3}px  mAP50={r['map50_mean']:.3f}"
              f"+/-{r['map50_std']:.3f} (n={r['n_seeds']})  {r['energy_mj']}mJ measured")


if __name__ == "__main__":
    main()
