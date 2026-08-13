"""Export the validation-mAP learning curve + best epoch from an Ultralytics run.

Reads <run>/results.csv (written every epoch) and emits results/metrics/train_curve.csv
with epoch, mAP@0.5, mAP@0.5:0.95 and a `best` flag at the early-stopping best epoch.
This is what justifies the training length: the curve shows convergence and the chosen
checkpoint, instead of a fixed epoch count.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

OUT = Path("results/metrics")


def _col(header, *needles):
    for i, h in enumerate(header):
        hl = h.strip().lower()
        if all(n in hl for n in needles):
            return i
    return None


def parse_results_csv(path: Path):
    rows = list(csv.reader(path.open()))
    header, data = rows[0], rows[1:]
    ie = _col(header, "epoch")
    i50 = _col(header, "map50") or _col(header, "map50(b)")
    i5095 = _col(header, "map50-95") or _col(header, "map50-95(b)")
    out = []
    for r in data:
        if not r:
            continue
        out.append({
            "epoch": int(float(r[ie])) if ie is not None else len(out) + 1,
            "map50": round(float(r[i50]), 4) if i50 is not None else 0.0,
            "map5095": round(float(r[i5095]), 4) if i5095 is not None else 0.0,
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="Ultralytics run dir (contains results.csv)")
    a = ap.parse_args(argv)
    rc = Path(a.run) / "results.csv"
    if not rc.exists():
        raise SystemExit(f"no results.csv in {a.run}; run training first")
    curve = parse_results_csv(rc)
    best = max(range(len(curve)), key=lambda i: curve[i]["map50"]) if curve else -1
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "train_curve.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "map50", "map5095", "best"])
        for i, c in enumerate(curve):
            w.writerow([c["epoch"], c["map50"], c["map5095"], int(i == best)])
    if curve:
        print(f"[done] {len(curve)} epochs; best epoch {curve[best]['epoch']} "
              f"mAP@0.5={curve[best]['map50']} -> {OUT}/train_curve.csv")


if __name__ == "__main__":
    main()
