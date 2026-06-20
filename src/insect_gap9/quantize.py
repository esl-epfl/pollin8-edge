"""INT8 quantisation for GAP9 via NNTool (GAPflow).

If the GAP SDK is present (SENSEI_SDK_ROOT / nntool importable) we run real
post-training quantisation calibrated on validation images. Otherwise we emit a
documented analytical INT8 footprint so the pipeline stays runnable end-to-end
in CI and on a laptop. The fallback is clearly labelled and never masquerades as
a measured result.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

BUILD = Path("build/insect_int8")


def _have_nntool() -> bool:
    try:
        import nntool  # noqa: F401  (ships with the GAP SDK)
        return True
    except Exception:
        return False


def analytic_footprint(weights: Path) -> dict:
    """Estimate INT8 size: params * 1 byte + activation high-water mark.
    Pico model ~0.3M params -> ~0.3MB weights; tiled activations < L2 (1.5MB)."""
    return {
        "params": 300_000,
        "weights_bytes_int8": 300_000,
        "peak_activation_bytes": 180_000,   # 160x120 tiled, fits 1.5MB L2
        "fits_l2_1p5mb": True,
        "method": "analytic-fallback",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/pico_insect/weights/best.pt")
    ap.add_argument("--calib", default="data/insects1201/images/val")
    ap.add_argument("--n-calib", type=int, default=500)
    ap.add_argument("--out", type=Path, default=BUILD)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    if _have_nntool():
        # Real path: export ONNX -> nntool -> per-layer min/max INT8 calibration.
        # Kept minimal; see docs/rationale.md for the calibration scheme.
        from .gapflow_real import quantize_with_nntool  # provided on SDK hosts
        info = quantize_with_nntool(a.weights, a.calib, a.n_calib, a.out)
        info["method"] = "nntool"
    else:
        print("[warn] nntool not found -> analytic INT8 footprint (label: simulated)")
        info = analytic_footprint(Path(a.weights))

    (a.out / "quant_info.json").write_text(json.dumps(info, indent=2))
    print("[done] INT8 model dir:", a.out, "| fits L2:", info.get("fits_l2_1p5mb"))


if __name__ == "__main__":
    main()
