"""IoT deployment profile: per-phase energy breakdown + a duty-cycle power trace.

Synthesises (from the GAP9 power model) the two canonical ULP figures:
  results/metrics/energy_breakdown.csv — energy per phase (capture/compute/radio/sleep)
  results/metrics/power_trace.csv      — power vs time over one wake->sleep cycle
Both tagged `simulated` until replaced by INA219 board measurements (measure_board.py).
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from . import metrics as M

OUT = Path("results/metrics")

# Phase model (durations ms, power mW). Sources: paper Sec. III + GAP9 datasheet.
CAPTURE_MS, CAPTURE_MW = 8.0, 5.0       # HM01B0 QVGA frame
COMPUTE_MW = 70.0                       # cluster + NE16 active
RADIO_MS, RADIO_MW = 6.0, 25.0          # BLE event TX (only on detection)
SLEEP_MW = M.DEEP_SLEEP_POWER_MW


def build(latency_ms: float, period_s: float, detect_fraction: float):
    cap_e = M.energy_per_inference_mj(CAPTURE_MS, CAPTURE_MW)
    cmp_e = M.energy_per_inference_mj(latency_ms, COMPUTE_MW)
    rad_e = M.energy_per_inference_mj(RADIO_MS, RADIO_MW) * detect_fraction
    sleep_ms = max(period_s * 1000 - CAPTURE_MS - latency_ms - RADIO_MS * detect_fraction, 0)
    slp_e = M.energy_per_inference_mj(sleep_ms, SLEEP_MW)
    breakdown = [("capture", cap_e), ("compute", cmp_e), ("radio", rad_e), ("sleep", slp_e)]

    # power trace timeline
    t, trace = 0.0, []
    def seg(dur, pw, phase, step=1.0):
        nonlocal t
        n = max(int(dur / step), 1)
        for _ in range(n):
            trace.append((round(t, 2), pw, phase)); t += dur / n
    seg(CAPTURE_MS, CAPTURE_MW, "capture")
    seg(latency_ms, COMPUTE_MW, "compute")
    if detect_fraction >= 0.5:
        seg(RADIO_MS, RADIO_MW, "radio")
    seg(min(sleep_ms, 40.0), SLEEP_MW, "sleep")   # cap drawn sleep tail for readability
    return breakdown, trace


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latency-ms", type=float, default=5.68)
    ap.add_argument("--period-s", type=float, default=30.0)
    ap.add_argument("--detect-fraction", type=float, default=1.0,
                    help="fraction of wakes that transmit (1.0 shows the radio phase)")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    breakdown, trace = build(a.latency_ms, a.period_s, a.detect_fraction)
    with (OUT / "energy_breakdown.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["phase", "energy_mj", "provenance"])
        for p, e in breakdown:
            w.writerow([p, round(e, 6), "simulated"])
    with (OUT / "power_trace.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_ms", "power_mw", "phase"])
        w.writerows(trace)
    print(f"[done] energy_breakdown.csv ({len(breakdown)} phases), power_trace.csv -> {OUT}")


if __name__ == "__main__":
    main()
