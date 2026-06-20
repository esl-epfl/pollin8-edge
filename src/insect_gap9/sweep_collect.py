"""Collect the TRAINED resolution sweep into one accuracy-vs-deployability Pareto.

Unlike `sweep.py` (which evaluates ONE 160-trained model at several sizes), this reads
students that were actually TRAINED at each resolution, for one or more regimes
(scratch, distilled). For each (regime, imgsz) it pairs:
  - measured accuracy  : from results/metrics/sweep/<regime>_<imgsz>.csv (evaluate.py output)
  - measured-static deploy: peak activation / MACs / fits-L2 from model_stats.profile_torch
                            at that imgsz (real forward hooks; analytic fallback if no torch)
  - latency / energy   : from the shared GAP9 performance model (sweep.py helpers)

Emits results/metrics/sweep_resolution_trained.csv with a `regime` column and, per regime,
a `recommended` flag = the SMALLEST imgsz within `--tol` of that regime's best mAP that still
fits L2 (the cheapest deployable input without an accuracy cliff).
"""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path

from . import sweep as S
from . import model_stats as MS

IN = Path("results/metrics/sweep")          # per-run accuracy CSVs: <regime>_<imgsz>.csv
OUT = Path("results/metrics/sweep_resolution_trained.csv")
ACTIVE_MW = S.ACTIVE_MW

# runs/<...>/sweep/sweep_<regime>_<imgsz>/weights/best.pt  (matches 45_res_sweep.sbatch)
WEIGHTS_TMPL = "sweep_{regime}_{imgsz}/weights/best.pt"


def _read_accuracy(p: Path) -> dict:
    row = next(csv.DictReader(p.open()))
    return {k: row[k] for k in ("map50", "map5095", "precision", "recall")}


def _deploy_at(weights: Path, imgsz: int, precision: str) -> dict:
    """Measured-static deployability at one resolution (reuses model_stats + sweep models)."""
    try:
        if not weights.is_file():
            raise FileNotFoundError(weights)
        _layers, params, macs, peak_act, prov = MS.profile_torch(str(weights), imgsz, precision)
    except Exception as e:
        print(f"[warn] static profiling unavailable for {weights} ({type(e).__name__}: {e})"
              f" -> analytic deployability")
        _layers, params, macs, peak_act, prov = MS.fallback(imgsz, precision)
    model_bytes = params * (1 if precision == "int8" else 4)
    fits = (model_bytes + peak_act + MS.CODE_BYTES) < MS.L2_BUDGET_BYTES
    # Latency/energy: measured-interpolated from on-board YOLOv5p measurements (Wiese2025
    # Table I), by operation count. energy = GAP9 compute energy = power x latency.
    from . import sensei_energy as SE
    e = SE.estimate(macs)
    energy_gap9 = round(e["gap9_power_mw"] * e["latency_ms"] / 1000.0, 3)
    return dict(macs=macs, peak_activation_bytes=peak_act, fits_l2=int(fits),
                latency_ms=e["latency_ms"], energy_mj=energy_gap9,
                gap9_power_mw=e["gap9_power_mw"],
                deploy_provenance=f"footprint:{prov}; energy:measured-interp(Wiese2025)")


def collect(runs_dir: Path, precision: str, tol: float) -> list[dict]:
    rows = []
    for p in sorted(IN.glob("*.csv")):
        m = re.fullmatch(r"(?P<regime>[a-z0-9]+)_(?P<imgsz>\d+)", p.stem)
        if not m:
            continue
        regime, imgsz = m["regime"], int(m["imgsz"])
        weights = runs_dir / WEIGHTS_TMPL.format(regime=regime, imgsz=imgsz)
        acc = _read_accuracy(p)
        dep = _deploy_at(weights, imgsz, precision)
        rows.append(dict(regime=regime, imgsz=imgsz, acc_provenance="measured",
                         **{k: acc[k] for k in ("map50", "map5095", "precision", "recall")},
                         **dep))
    # recommend the cheapest deployable resolution per regime
    for regime in {r["regime"] for r in rows}:
        grp = [r for r in rows if r["regime"] == regime]
        best = max(float(r["map50"]) for r in grp)
        eligible = [r for r in grp if r["fits_l2"] and float(r["map50"]) >= best - tol]
        rec = min((r["imgsz"] for r in eligible), default=None)
        for r in grp:
            r["recommended"] = int(rec is not None and r["imgsz"] == rec)
    rows.sort(key=lambda r: (r["regime"], r["imgsz"]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs/sweep",
                    help="dir holding sweep_<regime>_<imgsz>/weights/best.pt")
    ap.add_argument("--precision", default="int8", choices=["int8", "fp32"])
    ap.add_argument("--tol", type=float, default=0.02,
                    help="accuracy tolerance for the per-regime 'recommended' resolution")
    a = ap.parse_args(argv)

    if not IN.exists() or not any(IN.glob("*.csv")):
        raise SystemExit(f"[sweep_collect] no per-run CSVs in {IN} — run 45_res_sweep.sbatch first")
    rows = collect(Path(a.runs), a.precision, a.tol)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    regimes = sorted({r["regime"] for r in rows})
    print(f"[done] {len(rows)} points across regimes {regimes} -> {OUT}")
    for r in rows:
        star = " *recommended" if r["recommended"] else ""
        print(f"  {r['regime']:>9} @ {r['imgsz']:>3}px  mAP50={r['map50']}  "
              f"fits_L2={r['fits_l2']}  lat={r['latency_ms']}ms{star}")


if __name__ == "__main__":
    main()
