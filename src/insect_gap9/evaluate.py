"""Evaluate detection accuracy on a split and write results/metrics/accuracy.csv.

Always evaluate the headline number on the LOCKED test split.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

OUT = Path("results/metrics/accuracy.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/pico_insect/weights/best.pt")
    ap.add_argument("--data", default="configs/insects1201.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--provenance", default="measured",
                    help="measured (real run) | simulated (placeholder)")
    ap.add_argument("--out", default=str(OUT),
                    help="output CSV (use distinct names to compare runs, e.g. accuracy_scratch.csv)")
    a = ap.parse_args(argv)

    from ultralytics import YOLO  # lazy
    m = YOLO(a.weights)
    r = m.val(data=a.data, split=a.split, imgsz=a.imgsz, workers=2)  # low workers: avoid OOM on MIG
    rows = [{
        "split": a.split, "imgsz": a.imgsz,
        "map50": round(float(r.box.map50), 4),
        "map5095": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "provenance": a.provenance,
    }]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("[done] wrote", out, "-> mAP@0.5", rows[0]["map50"])


if __name__ == "__main__":
    main()
