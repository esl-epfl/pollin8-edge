"""Static model-complexity exporter for the deployability study.

Emits:
  results/metrics/model_stats.csv  — params, INT8 bytes, MACs @ deployment input,
                                      peak activation, L2 budget, fits_l2
  results/metrics/layers.csv       — per top-level layer: type, out shape, params, MACs, act bytes

Real path: profiles the trained Torch model with forward hooks (Conv2d MACs +
activation sizes), aggregated per top-level YOLO block. Fallback (no Torch): uses the
known pico-YOLOv5 parameter table and an analytic MAC estimate so the pipeline still
runs in CI. Every row is tagged `provenance ∈ {measured-static, analytic}`.
"""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path

OUT = Path("results/metrics")
L2_BUDGET_BYTES = 1_500_000          # GAP9 L2 SRAM
CODE_BYTES = 200_000                 # approx C runtime + weights-independent code

# Known pico-YOLOv5 (nc=9) top-level layers — exact params from the model summary.
FALLBACK_TYPES = ["Conv","Conv","C3","Conv","C3","Conv","C3","Conv","C3","SPPF","Conv",
                  "Upsample","Concat","C3","Conv","Upsample","Concat","C3","Conv","Concat",
                  "C3","Conv","Concat","C3","Detect"]
FALLBACK_PARAMS = [880,1184,1248,4672,4800,16240,14448,52624,49296,27352,5936,0,0,17584,
                   1856,0,0,5824,9280,0,14896,28336,0,50128,318667]


def _macs_total_estimate(imgsz: int) -> int:
    """~1.15 GMAC at 640px (from 2.3 GFLOPs) scaled to the deployment resolution."""
    return int(1.15e9 * (imgsz / 640.0) ** 2)


def profile_torch(weights: str, imgsz: int, precision: str):
    import torch
    from ultralytics import YOLO
    net = YOLO(weights).model.float().eval()
    rows = {}                                   # top_idx -> dict
    hooks = []

    def top_index(name: str) -> int:
        m = re.match(r"model\.(\d+)", name)
        return int(m.group(1)) if m else -1

    def make_hook(name, module):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, (list, tuple)) else out
            if not hasattr(o, "shape") or o.dim() < 4:
                return
            oc, oh, ow = int(o.shape[1]), int(o.shape[2]), int(o.shape[3])
            kh, kw = module.kernel_size
            macs = (module.in_channels // module.groups) * module.out_channels * kh * kw * oh * ow
            ti = top_index(name)
            r = rows.setdefault(ti, dict(macs=0, act=0, out=""))
            r["macs"] += macs
            r["act"] = max(r["act"], oc * oh * ow)   # peak activation within the block
            r["out"] = f"{oc}x{oh}x{ow}"
        return hook

    for name, m in net.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(make_hook(name, m)))
    with torch.no_grad():
        net(torch.zeros(1, 3, imgsz, imgsz))
    for h in hooks:
        h.remove()

    # params per top-level block
    pcount = {}
    for name, p in net.named_parameters():
        ti = top_index(name)
        pcount[ti] = pcount.get(ti, 0) + p.numel()
    types = {}
    for name, m in net.named_modules():
        ti = top_index(name)
        if "." not in name.replace("model.", "", 1) and ti >= 0:
            types[ti] = type(m).__name__

    bytes_per = 1 if precision.lower() == "int8" else 4
    layers = []
    for ti in sorted(k for k in rows if k >= 0):
        r = rows[ti]
        layers.append(dict(idx=ti, type=types.get(ti, "?"), out=r["out"],
                           params=pcount.get(ti, 0), macs=r["macs"],
                           act_bytes=r["act"] * bytes_per))
    total_params = sum(p.numel() for p in net.parameters())
    total_macs = sum(r["macs"] for r in rows.values())
    peak_act = max((l["act_bytes"] for l in layers), default=0)
    return layers, total_params, total_macs, peak_act, "measured-static"


def fallback(imgsz: int, precision: str):
    bytes_per = 1 if precision.lower() == "int8" else 4
    total_params = sum(FALLBACK_PARAMS)
    total_macs = _macs_total_estimate(imgsz)
    layers = []
    for i, (t, p) in enumerate(zip(FALLBACK_TYPES, FALLBACK_PARAMS)):
        share = (p / total_params) if total_params else 0
        layers.append(dict(idx=i, type=t, out="-", params=p,
                           macs=int(total_macs * share), act_bytes=0))
    peak_act = int(0.12 * L2_BUDGET_BYTES) * bytes_per // 1  # ~180 KB working set (INT8)
    return layers, total_params, total_macs, peak_act, "analytic"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/pico_insect/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=160, help="deployment input resolution")
    ap.add_argument("--precision", default="int8", choices=["int8", "fp32"])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    try:
        if not Path(a.weights).exists():
            raise FileNotFoundError(a.weights)
        layers, params, macs, peak_act, prov = profile_torch(a.weights, a.imgsz, a.precision)
    except Exception as e:
        print(f"[warn] torch profiling unavailable ({type(e).__name__}: {e}) -> analytic fallback")
        layers, params, macs, peak_act, prov = fallback(a.imgsz, a.precision)

    model_bytes = params * (1 if a.precision == "int8" else 4)
    fits = (model_bytes + peak_act + CODE_BYTES) < L2_BUDGET_BYTES
    with (OUT / "model_stats.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input_h", "input_w", "precision", "params", "model_bytes", "macs",
                    "peak_activation_bytes", "code_bytes", "l2_budget_bytes", "fits_l2", "provenance"])
        w.writerow([a.imgsz, a.imgsz, a.precision, params, model_bytes, macs,
                    peak_act, CODE_BYTES, L2_BUDGET_BYTES, fits, prov])
    with (OUT / "layers.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "type", "out", "params", "macs", "act_bytes"])
        w.writeheader()
        for l in layers:
            w.writerow(l)
    print(f"[done] params={params:,} macs@{a.imgsz}={macs:,} peak_act={peak_act:,}B "
          f"fits_L2={fits} ({prov}) -> {OUT}/model_stats.csv, layers.csv")


if __name__ == "__main__":
    main()
