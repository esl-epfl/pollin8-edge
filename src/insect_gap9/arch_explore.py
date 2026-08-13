"""Explore alternative TinyML architectures for insect monitoring on GAP9/NE16.

This is an ANALYTIC, LITERATURE-GROUNDED design-space study — NOT a training run. For
three candidate architectures (beyond our pico-YOLOv5 baseline) it estimates the GAP9
deployability (INT8 footprint vs 1.5 MB L2, MACs, NE16 operator coverage, ideal latency /
energy), ranks them by suitability for detection+counting, and — if the best candidate is
not *efficiently* deployable — emits a concrete gap report: which operators the NE16 engine
and the SENSEI/GAPflow SDK would need to support, and which hardware budgets (L1/L2) to grow.

Every number is an ESTIMATE from the cited papers (different datasets) or an analytic model;
all rows carry a provenance tag. Insect-specific accuracy requires actually training these
models — these figures are indicative, to guide what is worth training. Edit the ARCHS table
to refine. Outputs results/metrics/arch_explore.csv, arch_gaps.csv, results/arch_exploration.md.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from . import metrics as M

OUT = Path("results/metrics")
REPORT = Path("results/arch_exploration.md")

# --- GAP9 platform budgets (source: paper Sec. III + GAP9 datasheet) ----------
L2_BYTES = 1_500_000
L1_BYTES = 128_000                 # shared cluster L1
CODE_BYTES = 200_000               # runtime + weight-independent code
ACTIVE_MW = 70.0
# Throughput model (analytic): NE16 INT8 engine vs the 8-core RISC-V cluster fallback.
NE16_MACS_PER_CYCLE = 256          # representative NE16 INT8 peak (order of magnitude)
CLUSTER_MACS_PER_CYCLE = 16        # 8 cores x ~2 INT8 MAC/cycle (unsupported-op fallback)

# --- NE16 / GAPflow operator support ------------------------------------------
# (supported?, note). "Supported" = runs efficiently on NE16 or is a cheap memory/cluster op.
NE16_OPS = {
    "conv1x1":        (True,  "1x1 convolution — native NE16"),
    "conv3x3":        (True,  "3x3 convolution — native NE16"),
    "dwconv3x3":      (True,  "3x3 depthwise — supported (channel-aligned)"),
    "linear":         (True,  "fully-connected — native"),
    "maxpool":        (True,  "max pooling — cluster, cheap"),
    "avgpool":        (True,  "global/avg pooling — cluster, cheap"),
    "add":            (True,  "residual add — memory op"),
    "concat":         (True,  "concatenation — DMA memory op"),
    "relu":           (True,  "ReLU/ReLU6 — fused into NE16 conv"),
    "sigmoid":        (True,  "sigmoid on a small head — cluster LUT, cheap"),
    "silu":           (False, "SiLU/Swish — not fused on NE16; poly/LUT on the cluster"),
    "hardswish":      (False, "Hardswish — not in the fused-activation set; cluster fallback"),
    "conv5x5":        (False, "5x5+ kernels — NE16 is tuned for 1x1/3x3; inefficient"),
    "upsample":       (False, "nearest/transposed upsample — not in NE16 datapath; cluster"),
    "transpose_conv": (False, "deconvolution — no native primitive; cluster"),
    "squeeze_excite": (False, "SE block (global-pool x channel-scale) — not fused; cluster"),
    "patch_inference":(False, "depth-first/patch-based inference scheduling — NOT in GAPflow/AutoTiler"),
    "attention":      (False, "self-attention (matmul+softmax+LayerNorm) — not NE16-accelerated"),
}

GAP_FIX = {
    "silu":           "Retrain with ReLU6 (NE16-fused), or add a fused Swish LUT to the NE16 activation unit.",
    "hardswish":      "Add Hardswish to the NE16 fused-activation set, or substitute ReLU6 at training time.",
    "conv5x5":        "Provide an efficient 5x5 NE16 path, or factorise into stacked 3x3.",
    "upsample":       "Add nearest-neighbour upsample / transposed-conv to the NE16 datapath (today: cluster fallback).",
    "transpose_conv": "Provide a deconvolution primitive in GAPflow + NE16.",
    "squeeze_excite": "Fuse SE (global-avg-pool -> 1x1 -> scale) in GAPflow; add efficient channel-broadcast multiply on NE16.",
    "patch_inference":"Add depth-first / patch-based inference scheduling (a la MCUNetV2 TinyEngine) to GAPflow/AutoTiler "
                      "so peak activation is bounded by a patch, not the full feature map — this is what lets a stronger "
                      "backbone fit L2 at higher input resolution.",
    "attention":      "Add INT8 matmul/softmax/LayerNorm kernels, or avoid transformer blocks on this node.",
}
HW_FIX = ("Enlarge L1 (128->256 KB) and/or L2 (1.5->3 MB) to hold the peak-activation tile, "
          "or adopt patch-based inference to shrink the working set instead of growing SRAM.")

# --- Candidate architectures (estimates; edit to refine) ----------------------
# peak_act_kb is the binding L2 quantity (largest activation tensor, INT8).
# peak_act_kb_native (if set) is the peak WITHOUT patch-based inference, to show the gap.
ARCHS = [
    dict(name="pico-YOLOv5 (baseline)", family="anchor-free YOLO", source="this work / Ultralytics",
         input_px=160, params=625_251, macs=72_400_000, peak_act_kb=180,
         ops=["conv1x1", "conv3x3", "maxpool", "upsample", "concat", "silu", "sigmoid"],
         est_map50=0.50, acc_provenance="measured (insects, this work)",
         task_note="incumbent; SiLU + upsample already need cluster fallback / LUT."),
    dict(name="FOMO (centroid)", family="MobileNetV2-0.35 trunk + heatmap head",
         source="Edge Impulse; DSORT-MCU on GAP9 [DSORT2024]",
         input_px=160, params=110_000, macs=22_000_000, peak_act_kb=120,
         ops=["conv1x1", "dwconv3x3", "relu", "sigmoid"],
         est_map50=0.40, acc_provenance="literature-estimate (counting tasks)",
         task_note="predicts centres not boxes — ideal for counting/tracking; weak on overlapping insects."),
    dict(name="TinyissimoYOLO / XiNet", family="NE16-co-designed single-shot detector",
         source="[TinyissimoYOLO2023, Moosmann2023Flexible, Ancilotto2023XiNet]",
         input_px=256, params=450_000, macs=170_000_000, peak_act_kb=250,
         ops=["conv1x1", "conv3x3", "maxpool", "concat", "relu", "sigmoid"],
         est_map50=0.52, acc_provenance="literature-estimate (COCO/Pascal, GAP9)",
         task_note="built for GAP9 NE16 (ReLU, 1x1/3x3) — the most apples-to-apples ULP detector."),
    dict(name="MCUNetV2 (patch-based)", family="TinyNAS backbone + patch inference",
         source="[Lin2020MCUNet, Lin2021MCUNetV2]",
         input_px=224, params=730_000, macs=120_000_000, peak_act_kb=80, peak_act_kb_native=520,
         ops=["conv1x1", "dwconv3x3", "squeeze_excite", "hardswish", "patch_inference", "sigmoid"],
         est_map50=0.57, acc_provenance="literature-estimate (VWW/COCO det)",
         task_note="strongest accuracy/memory via patch inference + NAS; but SE+Hardswish+patch need SDK/NE16 support."),
]


# --- analysis (pure) ----------------------------------------------------------
def op_coverage(arch):
    sup = [o for o in arch["ops"] if NE16_OPS.get(o, (False, ""))[0]]
    uns = [o for o in arch["ops"] if not NE16_OPS.get(o, (False, ""))[0]]
    return sup, uns


def deployability(arch):
    model_bytes = arch["params"] * 1                       # INT8 weights
    peak = arch["peak_act_kb"] * 1024
    fits = (model_bytes + peak + CODE_BYTES) < L2_BYTES
    # would it fit WITHOUT patch-based inference (if applicable)?
    fits_native = None
    if arch.get("peak_act_kb_native"):
        fits_native = (model_bytes + arch["peak_act_kb_native"] * 1024 + CODE_BYTES) < L2_BYTES
    cycles = arch["macs"] / NE16_MACS_PER_CYCLE             # NE16-ideal (supported ops)
    lat = M.latency_from_cycles_ms(cycles)
    energy = M.energy_per_inference_mj(lat, ACTIVE_MW)
    sup, uns = op_coverage(arch)
    return dict(model_kb=round(model_bytes / 1024, 1), peak_act_kb=arch["peak_act_kb"],
                fits_l2=int(fits), fits_l2_native=("" if fits_native is None else int(fits_native)),
                latency_ms=round(lat, 3), energy_mj=round(energy, 4),
                ne16_ops_supported=len(sup), ne16_ops_total=len(arch["ops"]),
                unsupported=";".join(uns))


def efficiently_deployable(arch, dep):
    return dep["fits_l2"] and not dep["unsupported"]


def gap_report(arch, dep):
    lines = []
    _sup, uns = op_coverage(arch)
    for o in uns:
        lines.append((o, "operator", NE16_OPS[o][1], GAP_FIX.get(o, "add native support in NE16/GAPflow.")))
    if not dep["fits_l2"]:
        lines.append(("L2 budget", "memory",
                      f"footprint {dep['model_kb']:.0f} KB weights + {dep['peak_act_kb']} KB activation "
                      f"exceeds {L2_BYTES//1024} KB L2", HW_FIX))
    if dep["fits_l2_native"] == 0:   # only fits BECAUSE of patch inference
        lines.append(("L2 budget (native)", "memory",
                      f"without patch inference the peak activation ({arch['peak_act_kb_native']} KB) blows L2; "
                      f"patch-based scheduling is what keeps it on-chip", HW_FIX))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT / "arch_explore.csv"))
    ap.add_argument("--out-gaps", default=str(OUT / "arch_gaps.csv"))
    ap.add_argument("--report", default=str(REPORT))
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for arch in ARCHS:
        d = deployability(arch)
        rows.append(dict(name=arch["name"], family=arch["family"], input_px=arch["input_px"],
                         params=arch["params"], macs=arch["macs"], est_map50=arch["est_map50"],
                         acc_provenance=arch["acc_provenance"], **d, source=arch["source"]))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # best ALTERNATIVE (exclude the incumbent baseline) by task accuracy
    alts = [arch for arch in ARCHS if "baseline" not in arch["name"]]
    best = max(alts, key=lambda x: x["est_map50"])
    bdep = deployability(best)
    gaps = gap_report(best, bdep) if not efficiently_deployable(best, bdep) else []

    with open(a.out_gaps, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["architecture", "item", "kind", "why", "recommended_fix"])
        for item, kind, why, fix in gaps:
            w.writerow([best["name"], item, kind, why, fix])

    # human-readable report
    md = ["# Alternative-architecture exploration for insect monitoring on GAP9\n",
          "_Analytic / literature-grounded estimates — not trained on insects. Provenance tagged; "
          "edit `arch_explore.py:ARCHS` to refine._\n",
          "| Architecture | input | params | MACs (M) | peak act (KB) | fits L2 | lat (ms)¹ | energy (mJ)¹ | NE16 ops | est mAP@.5² |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | {r['input_px']} | {r['params']:,} | {r['macs']/1e6:.0f} | "
                  f"{r['peak_act_kb']} | {'yes' if r['fits_l2'] else 'NO'} | {r['latency_ms']} | "
                  f"{r['energy_mj']} | {r['ne16_ops_supported']}/{r['ne16_ops_total']} | {r['est_map50']} |")
    md += ["\n¹ NE16-ideal (all supported ops on the engine); unsupported ops add cluster-fallback "
           "overhead not included here. ² cross-dataset literature estimates — confirm by training.\n",
           f"## Best task fit: **{best['name']}**  (est mAP@0.5 {best['est_map50']})",
           f"{best['task_note']}\n"]
    if gaps:
        md.append("**Not efficiently deployable as-is.** To deploy it on the SENSEI/GAP9 node, the "
                  "SDK and NE16 engine would need:\n")
        md.append("| Missing item | Kind | Why it matters | Recommended fix |")
        md.append("|---|---|---|---|")
        for item, kind, why, fix in gaps:
            md.append(f"| `{item}` | {kind} | {why} | {fix} |")
        md.append("\nUntil then, the **deployable** picks are the NE16-native ones "
                  "(TinyissimoYOLO/XiNet for detection, FOMO for pure counting); the patch-based "
                  "backbone is the highest-value SDK investment for a future accuracy lift.")
    else:
        md.append("**Efficiently deployable** on GAP9 as-is (fits L2, all ops NE16-native).")
    md += ["\n## Sources", "FOMO/centroid + GAP9 tiling: DSORT2024. GAP9 NE16 detectors: "
           "TinyissimoYOLO2023, Moosmann2023Flexible, Ancilotto2023XiNet (XiNet). NAS + patch "
           "inference: Lin2020MCUNet, Lin2021MCUNetV2. Platform budgets: paper Sec. III + GAP9 datasheet."]
    Path(a.report).write_text("\n".join(md) + "\n")

    print(f"[done] {len(rows)} architectures -> {a.out}")
    print(f"[best alternative] {best['name']} (est mAP@0.5 {best['est_map50']}) -> "
          f"{'efficiently deployable' if not gaps else str(len(gaps)) + ' deployability gap(s)'}")
    for item, kind, why, _fix in gaps:
        print(f"   gap: {item} ({kind}) — {why}")
    print(f"[report] {a.report} ; gaps -> {a.out_gaps}")


if __name__ == "__main__":
    main()
