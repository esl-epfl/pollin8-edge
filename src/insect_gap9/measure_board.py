"""Hook for REAL on-board measurement (INA219) — fills the 'measured' rows later.

Deferred track: when the physical SENSEI node is available, this reads the
on-board INA219 current sensor during inference and overwrites the 'simulated'
rows with 'measured' ones. Left as a thin, documented stub so the contract is
fixed now and the swap is a one-file change.
"""
from __future__ import annotations
import argparse


def measure(duration_s: float = 10.0) -> dict:
    raise NotImplementedError(
        "Connect INA219 over I2C and sample current at >=100 Hz during inference. "
        "Return {cycles, latency_ms, active_power_mw, energy_mj, provenance='measured'}."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration-s", type=float, default=10.0)
    ap.parse_args()
    print(__doc__)
