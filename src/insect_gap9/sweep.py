"""Accuracy vs deployability sweeps -> Pareto CSVs.

resolution sweep: evaluate the trained model at several input sizes; for each, record
mAP (real if Ultralytics present, else analytic curve) plus MACs/latency/energy scaled
from the model-complexity and GAP9 performance models.

Emits results/metrics/sweep_resolution.csv and sweep_precision.csv.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from . import metrics as M

OUT = Path("results/metrics")
ACTIVE_MW = 70.0


def _macs_at(imgsz: int) -> int:
    return int(1.15e9 * (imgsz / 640.0) ** 2)


def _latency_ms_from_macs(macs: int) -> float:
    cycles = int(macs / 8 / 2.5)               # 8 cores, ~2.5 INT8 MAC/cycle/core
    return M.latency_from_cycles_ms(cycles)


def _analytic_map(imgsz: int) -> float:
    """Saturating accuracy-vs-resolution curve (placeholder until real eval)."""
    import math
    return round(0.92 * (1 - math.exp(-(imgsz) / 110.0)), 3)


def resolution_sweep(weights, data, sizes, real: bool):
    rows = []
    for s in sizes:
        macs = _macs_at(s)
        lat = _latency_ms_from_macs(macs)
        en = M.energy_per_inference_mj(lat, ACTIVE_MW)
        if real:
            from ultralytics import YOLO
            r = YOLO(weights).val(data=data, imgsz=s, split="test", verbose=False)
            map50 = round(float(r.box.map50), 3); prov = "measured"
        else:
            map50 = _analytic_map(s); prov = "analytic"
        rows.append(dict(imgsz=s, map50=map50, latency_ms=round(lat, 3),
                        energy_mj=round(en, 4), macs=macs, provenance=prov))
    return rows


def precision_sweep(map50_int8: float):
    """INT8 vs FP32: FP32 is 4x bytes; INT8 typically within ~3 pp on this task."""
    params = 625_251
    return [
        dict(precision="fp32", map50=round(map50_int8 + 0.03, 3),
             model_bytes=params * 4, energy_mj=None, provenance="analytic"),
        dict(precision="int8", map50=round(map50_int8, 3),
             model_bytes=params * 1, energy_mj=None, provenance="analytic"),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/pico_insect/weights/best.pt")
    ap.add_argument("--data", default="configs/insects1201.yaml")
    ap.add_argument("--sizes", type=int, nargs="*", default=[96, 128, 160, 224, 320])
    ap.add_argument("--real", action="store_true", help="run real Ultralytics eval per size")
    ap.add_argument("--map50", type=float, default=0.88, help="INT8 mAP for precision sweep")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="accuracy tolerance: recommend the SMALLEST size within tol of the best mAP")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    real = a.real and Path(a.weights).exists()
    if a.real and not real:
        print("[warn] --real requested but weights/ultralytics missing -> analytic curve")
    res = resolution_sweep(a.weights, a.data, a.sizes, real)
    # Recommend the smallest resolution whose mAP is within `tol` of the best —
    # i.e. the cheapest input that still fits GAP9 without an accuracy cliff.
    best = max(r["map50"] for r in res)
    rec = min((r["imgsz"] for r in res if r["map50"] >= best - a.tol), default=res[-1]["imgsz"])
    for r in res:
        r["recommended"] = int(r["imgsz"] == rec)
    with (OUT / "sweep_resolution.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    prec = precision_sweep(a.map50)
    with (OUT / "sweep_precision.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prec[0])); w.writeheader(); w.writerows(prec)
    print(f"[done] resolution sweep ({len(res)} pts; recommended {rec}px), "
          f"precision sweep -> {OUT}")


if __name__ == "__main__":
    main()
