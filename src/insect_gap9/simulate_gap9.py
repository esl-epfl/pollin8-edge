"""GVSOC simulation -> cycles, latency, energy CSVs.

With the GAP SDK present we invoke gvsoc on the compiled model and parse the
cycle counters. Without it we fall back to an analytical performance model
derived from published GAP9 YOLO numbers (TinyissimoYOLO: 17 ms, 1.59 mJ at a
comparable op count) scaled to our ~42 MOp budget. Output is tagged accordingly.
"""
from __future__ import annotations
import argparse, csv, os, shutil
from pathlib import Path
from . import metrics as M

OUT = Path("results/metrics")
DEPLOY_SIZE = 320   # deployed operating point (matches scripts/make_figures.py DEPLOY_SIZE)


def _have_gvsoc() -> bool:
    return bool(os.environ.get("SENSEI_SDK_ROOT")) and shutil.which("gvsoc") is not None


def measured_perf(imgsz: int = DEPLOY_SIZE) -> dict | None:
    """Deployed operating point from the silicon sweep (Wiese-measured energy/power/latency).

    Preferred over the analytical model so the deployment projection reproduces the paper's
    Table II. Returns None if results/metrics/sensei_arch_sweep.csv is absent.
    """
    p = OUT / "sensei_arch_sweep.csv"
    if not p.exists():
        return None
    with p.open() as f:
        for r in csv.DictReader(f):
            if (r.get("arm") or "").strip() == "base" and int(r["imgsz"]) == imgsz:
                lat = float(r["latency_ms"])
                return dict(cycles=int(lat / 1e3 * M.GAP9_CLUSTER_FREQ_HZ),
                            latency_ms=lat, active_power_mw=float(r["gap9_power_mw"]),
                            energy_mj=float(r["energy_mj"]))
    return None


def analytic_perf(mops: float = 42.0) -> dict:
    """Scale a published GAP9 operating point to our op count.
    Reference: ~42 MOp -> NE16-accelerated INT8. Conservative active power 70 mW."""
    cycles = int(mops * 1e6 / 8 / 2.5)          # 8 cores, ~2.5 INT8 MACs/cycle/core
    latency_ms = M.latency_from_cycles_ms(cycles)
    active_mw = 70.0
    energy_mj = M.energy_per_inference_mj(latency_ms, active_mw)
    return dict(cycles=cycles, latency_ms=round(latency_ms, 3),
                active_power_mw=active_mw, energy_mj=round(energy_mj, 4))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="build/insect_int8")
    ap.add_argument("--period-s", type=float, default=30.0, help="duty-cycle period")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if _have_gvsoc():
        from .gapflow_real import run_gvsoc       # provided on SDK hosts
        perf = run_gvsoc(a.model); provenance = "simulated"  # GVSOC = cycle-accurate sim
    elif (m := measured_perf()) is not None:
        perf = m; provenance = "measured"         # deployed op-point from the silicon sweep
        print(f"[info] using measured deployed operating point ({DEPLOY_SIZE}px) from sensei_arch_sweep.csv")
    else:
        print("[warn] gvsoc not found and no sweep CSV -> analytical perf model")
        perf = analytic_perf(); provenance = "simulated"

    # latency + cycles
    with (OUT / "latency.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["platform", "cycles", "latency_ms", "provenance"])
        w.writerow(["GAP9", perf["cycles"], perf["latency_ms"], provenance])

    # energy + deployment projection
    avg_mw = M.duty_cycle_avg_power_mw(perf["active_power_mw"], perf["latency_ms"], a.period_s)
    with (OUT / "energy.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["platform", "energy_mj", "active_power_mw", "avg_power_mw_dutycycle",
                    "monthly_wh_upperbound", "provenance"])
        w.writerow(["GAP9", perf["energy_mj"], perf["active_power_mw"], round(avg_mw, 4),
                    round(M.monthly_energy_wh(perf["active_power_mw"]), 2), provenance])
    with (OUT / "deployment.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["platform", "battery_days_3Ah", "period_s", "provenance"])
        w.writerow(["GAP9", round(M.battery_life_days(avg_mw), 1), a.period_s, provenance])

    print("[done] wrote latency.csv, energy.csv, deployment.csv to", OUT)


if __name__ == "__main__":
    main()
