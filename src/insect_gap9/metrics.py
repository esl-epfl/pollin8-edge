"""Energy / deployment model for the SENSEI GAP9 node.

All constants are documented with their source so reviewers can audit them.
Functions are pure and unit-tested (tests/test_metrics.py). Units are explicit
in every name. This module has NO heavy dependencies so it imports anywhere.
"""
from __future__ import annotations
from dataclasses import dataclass

# --- Platform constants (source: paper Sec. III + GAP9 datasheet) -------------
GAP9_CLUSTER_FREQ_HZ = 370e6        # 8-core cluster max frequency
ACTIVE_POWER_MW_RANGE = (30.0, 80.0)  # camera capture + cluster inference
DEEP_SLEEP_POWER_MW = 0.1           # < 100 µW PMU/RTC/MRAM domain
CAMERA_POWER_MW = 0.07              # HiMax HM01B0 always-on (70 µW)


def energy_per_inference_mj(latency_ms: float, active_power_mw: float) -> float:
    """Active energy of one inference. mW * ms = µJ -> /1000 = mJ."""
    return active_power_mw * latency_ms / 1000.0


def latency_from_cycles_ms(cycles: int, freq_hz: float = GAP9_CLUSTER_FREQ_HZ) -> float:
    """Wall-clock latency from a GVSOC/HW cycle count."""
    return cycles / freq_hz * 1e3


def duty_cycle_avg_power_mw(
    active_power_mw: float,
    latency_ms: float,
    period_s: float,
    detect_fraction: float = 0.05,
    deep_sleep_power_mw: float = DEEP_SLEEP_POWER_MW,
) -> float:
    """Average system power under duty-cycling.

    Each `period_s` we run one inference (`latency_ms`); on `detect_fraction`
    of wakes we pay a small radio/record cost (folded into active for an upper
    bound). The rest of the period is deep sleep.
    """
    active_s = latency_ms / 1000.0
    sleep_s = max(period_s - active_s, 0.0)
    energy_j = (
        active_power_mw * active_s
        + deep_sleep_power_mw * sleep_s
    ) / 1000.0
    # detect_fraction reserved for future radio term; kept explicit for clarity
    _ = detect_fraction
    return energy_j / period_s * 1000.0  # -> mW


def battery_life_days(
    avg_power_mw: float, capacity_mah: float = 3000.0, voltage_v: float = 3.7
) -> float:
    """Projected lifetime on a Li-Po cell from average power."""
    energy_wh = capacity_mah / 1000.0 * voltage_v
    avg_power_w = avg_power_mw / 1000.0
    return energy_wh / avg_power_w / 24.0


def monthly_energy_wh(active_power_mw: float) -> float:
    """Architectural upper bound: continuous active operation for 30 days."""
    return active_power_mw / 1000.0 * 24 * 30


@dataclass
class InferenceResult:
    """One row of the latency/energy CSVs."""
    platform: str
    cycles: int | None
    latency_ms: float
    active_power_mw: float
    energy_mj: float
    provenance: str  # "simulated" | "measured"
