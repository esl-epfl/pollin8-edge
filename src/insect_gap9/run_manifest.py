"""Provenance manifest: tie every result to the exact conditions that produced it.

Captures code (git SHA + dirty flag), environment (python/torch/ultralytics, host,
GPU), data (Zenodo DOI + md5s), the training config, and the headline outcomes
(best mAP + epoch, params/MACs/fits-L2, latency/energy/battery). One JSON + one flat
CSV so a reviewer can answer "what was run, on what, with which code/data, and what came out".
"""
from __future__ import annotations
import argparse, csv, json, subprocess, socket, sys
from datetime import datetime, timezone
from pathlib import Path

METRICS = Path("results/metrics")


def _sh(*cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _ver(pkg):
    try:
        import importlib.metadata as m
        return m.version(pkg)
    except Exception:
        return "?"


def _row(name, key, default=""):
    p = METRICS / name
    if not p.exists():
        return default
    r = list(csv.DictReader(p.open()))
    return r[0].get(key, default) if r else default


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="", help="Ultralytics run dir (for args.yaml + results.csv)")
    a = ap.parse_args(argv)
    METRICS.mkdir(parents=True, exist_ok=True)

    # best epoch from the learning curve
    best_ep, best_map = "", ""
    tc = METRICS / "train_curve.csv"
    if tc.exists():
        rows = list(csv.DictReader(tc.open()))
        for r in rows:
            if r.get("best") == "1":
                best_ep, best_map = r["epoch"], r["map50"]

    from . import download_data as dd  # Zenodo md5s
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": {
            "git_sha": _sh("git", "rev-parse", "HEAD") or "unknown",
            "git_dirty": bool(_sh("git", "status", "--porcelain")),
            "git_branch": _sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
        },
        "env": {
            "host": socket.gethostname(),
            "python": sys.version.split()[0],
            "torch": _ver("torch"),
            "ultralytics": _ver("ultralytics"),
            "numpy": _ver("numpy"),
            "gpu": _sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").splitlines()[0]
                   if _sh("nvidia-smi", "-L") else "cpu",
        },
        "data": {
            "source": "Bjerge et al. 2022, Zenodo 10.5281/zenodo.7395752",
            "md5": dd.FILES,
        },
        "results": {
            "best_epoch": best_ep,
            "val_map50_best": best_map,
            "test_map50": _row("accuracy.csv", "map50"),
            "test_map5095": _row("accuracy.csv", "map5095"),
            "params": _row("model_stats.csv", "params"),
            "macs": _row("model_stats.csv", "macs"),
            "fits_l2": _row("model_stats.csv", "fits_l2"),
            "latency_ms": _row("latency.csv", "latency_ms"),
            "energy_mj": _row("energy.csv", "energy_mj"),
            "battery_days_3Ah": _row("deployment.csv", "battery_days_3Ah"),
            "provenance": _row("energy.csv", "provenance", "simulated"),
        },
    }
    # training hyperparameters from the run's args.yaml (if available)
    if a.run:
        args_yaml = Path(a.run) / "args.yaml"
        if args_yaml.exists():
            try:
                import yaml
                hp = yaml.safe_load(args_yaml.read_text())
                manifest["train_args"] = {k: hp.get(k) for k in
                    ("epochs", "patience", "imgsz", "batch", "lr0", "optimizer", "freeze", "seed")}
            except Exception:
                pass

    (METRICS / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    # flat one-row CSV for quick diffing across runs
    flat = {"timestamp": manifest["timestamp_utc"], **manifest["code"],
            **{f"env_{k}": v for k, v in manifest["env"].items()},
            **manifest["results"]}
    with (METRICS / "run_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat)); w.writeheader(); w.writerow(flat)
    print("[done] run_manifest.json + run_manifest.csv ->", METRICS)
    print(f"       git={manifest['code']['git_sha'][:8]} best_ep={best_ep} "
          f"test_map50={manifest['results']['test_map50']} params={manifest['results']['params']}")


if __name__ == "__main__":
    main()
