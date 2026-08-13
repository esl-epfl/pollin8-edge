"""Aggregate results/metrics/int8_validation.csv into the FINAL FP32-vs-INT8 comparison.

`validate_int8.py` (ONNX-Runtime PTQ, a PROXY for NE16) and `validate_int8_nntool.py`
(GreenWaves NNTool SQ8/NE16, the DEPLOYED quantiser) both append, per run, three rows
(precision in {fp32, int8, delta(int8-fp32)}) to the same CSV, distinguished by the
`provenance` column. This script groups by QUANTISER FAMILY first -- proxy and deployed
rows are never deduped or averaged across each other -- then aggregates fp32/int8 over
seeds per (arm, imgsz), prints the INT8 degradation per family, highlights the deployed
point (base @ 320 px), and writes the LaTeX table `results/tables/int8_compare.tex`
(proxy rows per resolution + the NNTool deployed-point row).

    PYTHONPATH=src python scripts/int8_summary.py [results/metrics/int8_validation.csv]
"""
import csv, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "results/metrics/int8_validation.csv")
OUT = Path("results/tables/int8_compare.tex")
METRICS = [("map50", "mAP@0.5"), ("micro_f1", "F1(count)"), ("count_mae_pct", "count-MAE%"),
           ("net_bias_pct", "net-bias%"), ("model_mb", "size MB")]
FAM_LABEL = {"ort-proxy": "ONNX-Runtime PTQ (proxy for NE16)",
             "nntool": "NNTool SQ8/NE16 (deployed quantiser)"}


def fam(provenance: str) -> str:
    """Quantiser family of a row. `classic-onnx-fp32-ref(reused)` is the --skip-float
    fp32 leg of an NNTool comparison, so it pairs with the nntool family."""
    p = (provenance or "").lower()
    return "nntool" if p.startswith(("nntool", "classic-onnx-fp32-ref")) else "ort-proxy"


if not SRC.exists():
    sys.exit(f"[int8_summary] {SRC} not found -- run scripts/slurm/50_int8_validate.sbatch "
             "(proxy) and/or 51_nntool_validate.sbatch (deployed quantiser) first")

# Dedupe reruns WITHIN a family: a resubmitted (fam,arm,size,seed,precision) must not be
# averaged twice -> keep the LAST row. Families never clobber each other.
rows = {}
for r in csv.DictReader(SRC.open()):
    if r["precision"] not in ("fp32", "int8"):
        continue                                   # ignore the per-run delta rows; we recompute means
    f = fam(r.get("provenance", ""))
    rows[(f, r["arm"], r["size"], r["seed"], r["precision"])] = r

# group: (fam, arm, size, precision) -> {metric: [values over seeds]}
g = defaultdict(lambda: defaultdict(list))
for (f, arm, size, _seed, prec), r in rows.items():
    for m, _ in METRICS:
        try:
            x = float(r[m])
        except (KeyError, ValueError, TypeError):
            continue
        if not math.isnan(x):                      # drop a failed-parse nan instead of poisoning st.mean
            g[(f, arm, int(size), prec)][m].append(x)

def mean(f, arm, size, prec, m):
    v = g.get((f, arm, size, prec), {}).get(m, [])
    return st.mean(v) if v else float("nan")

def nseeds(f, arm, size):
    return len(g.get((f, arm, size, "fp32"), {}).get("map50", []))

fams = [f for f in ("ort-proxy", "nntool") if any(k[0] == f for k in g)]
for f in fams:
    configs = sorted({(a, s) for (ff, a, s, _) in g if ff == f},
                     key=lambda t: (t[0] != "base", t[1]))
    print(f"\nINT8 vs FP32 -- {FAM_LABEL[f]} -- mean over seeds -- {SRC}\n")
    print(f"{'arm':11}{'px':>5}{'seeds':>6}  {'metric':11}{'FP32':>9}{'INT8':>9}{'D(int8-fp32)':>14}")
    for (arm, size) in configs:
        n = nseeds(f, arm, size)
        for m, label in METRICS:
            fv, iv = mean(f, arm, size, "fp32", m), mean(f, arm, size, "int8", m)
            print(f"{arm:11}{size:>5}{n:>6}  {label:11}{fv:>9.4f}{iv:>9.4f}{iv - fv:>+14.4f}")
        print()

print("DEPLOYED POINT (base @ 320 px):")
for f in fams:
    if not (g.get((f, "base", 320, "fp32")) or g.get((f, "base", 320, "int8"))):
        continue
    print(f"  [{FAM_LABEL[f]}]" + ("   << headline (deployed quantiser)" if f == "nntool" else ""))
    for m, label in METRICS:
        fv, iv = mean(f, "base", 320, "fp32", m), mean(f, "base", 320, "int8", m)
        print(f"    {label:11} FP32 {fv:.4f}   INT8 {iv:.4f}   delta {iv - fv:+.4f}")

# LaTeX table for the paper: base recipe, FP32 vs INT8 per resolution (proxy rows) plus the
# NNTool deployed-point row. Build rows first so a partial sweep renders gaps as "--" and an
# empty result never clobbers a good table with 'nan' cells.
def _c(x, fmt="{:.3f}"):
    return "--" if math.isnan(x) else fmt.format(x)

def _pair(f, size, m, fmt="{:.3f}"):
    return f"{_c(mean(f, 'base', size, 'fp32', m), fmt)}\\,/\\,{_c(mean(f, 'base', size, 'int8', m), fmt)}"

def _row(f, size, label):
    vals = [mean(f, "base", size, p, m) for p in ("fp32", "int8") for m, _ in METRICS[:4]]
    if all(math.isnan(v) for v in vals):
        return None                                # nothing for this cell yet
    fm, im = mean(f, "base", size, "fp32", "map50"), mean(f, "base", size, "int8", "map50")
    dmap = "--" if (math.isnan(im) or math.isnan(fm)) else f"{100 * (im - fm):+.1f}"
    return (f"{label} & {_pair(f, size, 'map50')} & {_pair(f, size, 'micro_f1')} "
            f"& {_pair(f, size, 'count_mae_pct', '{:.1f}')} "
            f"& {_pair(f, size, 'net_bias_pct', '{:+.1f}')} & {dmap} \\\\\n")

tex_rows = [r for size in (192, 320, 512)
            if (r := _row("ort-proxy", size, f"${size}^2$"))]
nntool_row = _row("nntool", 320, "$320^2$ (NNTool)")

if not (tex_rows or nntool_row):
    print(f"\n[int8_summary] no base-recipe data yet -- NOT writing {OUT} (kept any existing table)")
else:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fo:
        fo.write("% generated by scripts/int8_summary.py from results/metrics/int8_validation.csv\n"
                 "% Per-resolution rows: ONNX-Runtime static PTQ (runnable proxy for NE16).\n"
                 "% '(NNTool)' row: GreenWaves NNTool SQ8/NE16 -- the DEPLOYED quantiser.\n")
        fo.write("\\begin{tabular}{@{}rccccc@{}}\n\\toprule\n")
        fo.write("Input & mAP@0.5 & $F_1$(count) & count-MAE & net bias & $\\Delta$mAP \\\\\n")
        fo.write(" (px) & FP32\\,/\\,INT8 & FP32\\,/\\,INT8 & FP32\\,/\\,INT8 "
                 "& FP32\\,/\\,INT8 & (pp) \\\\\n\\midrule\n")
        for row in tex_rows:
            fo.write(row)
        if nntool_row:
            if tex_rows:
                fo.write("\\midrule\n")
            fo.write(nntool_row)
        fo.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\n[done] wrote {OUT}")
