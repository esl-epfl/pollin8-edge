"""Measured-grounded latency/energy model for YOLOv5p on the SENSEI/GAP9 node.

Source: Wiese et al., "A Multi-Modal IoT Node for Energy-Efficient Environmental Monitoring
with Edge AI Processing," IEEE COINS 2025, Table I --- on-board (not simulated) measurements
of the same quantised YOLOv5p family on the same GAP9 SoC (NE16, INT8, AutoTiler). Because
energy on this engine is compute-bound, we model it as a function of operation count (MACs,
in millions) and interpolate to our detector's per-resolution MACs. This grounds our
deployment figures in measured silicon rather than a standalone cycle estimate.
"""
from __future__ import annotations
import bisect

# Wiese 2025, Table I. Op = MACs [M]; energy/frame [mJ]; FPS [1/s]; GAP9-only power [mW].
# Each entry was MEASURED on GAP9 for the YOLOv5p HeadCount model at the matching square
# input size in _PX. Our insect detector uses the IDENTICAL architecture (configs/
# yolov5p_sensei.yaml, op-verified against the committed HeadCount tflite) differing only in
# the final 1x1 head (nc 1->9, <1% of MACs), so at a measured size these rows are a DIRECT
# readoff for our model -- not an interpolation across a model boundary.
_PX   = [64, 128, 192, 256, 320, 384, 448, 512]            # measured input resolutions
_OP   = [5, 19, 42, 74, 116, 167, 227, 297]
_EMJ  = [1.50, 1.98, 2.63, 3.87, 5.46, 7.29, 9.57, 12.99]   # total (GAP9+mem+camera)
_FPS  = [25.2, 22.2, 19.5, 15.1, 12.2, 9.7, 7.9, 6.7]
_GAP9 = [23.5, 31.2, 40.1, 49.4, 58.7, 64.2, 65.6, 73.9]    # GAP9-only power [mW]


def measured_at_size(px: int) -> dict:
    """Direct measured per-inference cost at a Wiese2025-measured input size (no interpolation).

    Valid only when our deployed network is operator-/size-identical to the characterised
    HeadCount model (it is, by yolov5p_sensei.yaml). Raises if `px` was not measured -- use
    `estimate(macs)` only as the interpolation fallback for off-grid sizes.
    """
    if px not in _PX:
        raise ValueError(f"{px}px not in measured set {_PX}; use estimate(macs) to interpolate")
    i = _PX.index(px)
    return dict(imgsz=px, macs_M=_OP[i],
                latency_ms=round(1000.0 / _FPS[i], 2),
                energy_mj=_EMJ[i], gap9_power_mw=_GAP9[i],
                provenance="measured (Wiese2025 Table I; op-identical architecture)")


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:                       # extrapolate on the last linear segment
        x0, x1, y0, y1 = xs[-2], xs[-1], ys[-2], ys[-1]
    else:
        i = bisect.bisect_right(xs, x)
        x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)


def estimate(macs: float) -> dict:
    """Measured-interpolated per-inference cost for a model of `macs` MACs at GAP9."""
    op = macs / 1e6
    emj = _interp(op, _OP, _EMJ)
    fps = _interp(op, _OP, _FPS)
    gap9_mw = _interp(op, _OP, _GAP9)
    return dict(macs_M=round(op, 1),
                latency_ms=round(1000.0 / fps, 2),
                energy_mj=round(emj, 3),
                gap9_power_mw=round(gap9_mw, 1),
                provenance="measured-interpolated (Wiese2025 Table I)")


if __name__ == "__main__":
    # our detector's MACs per swept resolution (from sweep_resolution_trained.csv)
    for px, macs in [(128, 46e6), (160, 72.4e6), (224, 141e6), (320, 287.5e6)]:
        e = estimate(macs)
        print(f"{px:>3}px  {e['macs_M']:>6} MMAC -> {e['latency_ms']:>6} ms  "
              f"{e['energy_mj']:>6} mJ  ({e['gap9_power_mw']} mW GAP9)")
