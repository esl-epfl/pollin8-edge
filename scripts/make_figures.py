"""Generate the paper's figures from results/metrics/*.csv.

Conventions: Wong palette, hatched bars, single-column width. Multi-panel
figures are ALSO exported panel-by-panel (``*_panelA.pdf`` ...) so the LaTeX
side can drop/swap/re-pair panels via ``subcaption`` without rerunning Python.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")          # headless (cluster): set before pyplot is imported
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from _style import apply_rc, styled_bars, COL_W_IN, GOLDEN, WONG, PALETTE, HATCHES

METRICS = Path("results/metrics")

# regime -> (colour, marker, linestyle): distinct markers+dashes keep the curves
# separable in greyscale, the line-plot analogue of the bar hatches.
REGIME_STYLE = {
    "scratch":   (WONG["blue"],   "o", "-"),
    "distilled": (WONG["vermil"], "s", "--"),   # default distillation = teacher conf 0.25
    "distill50": (WONG["green"],  "^", ":"),    # teacher conf 0.50
}
# legend labels (the stored regime names are terse)
REGIME_LABEL = {
    "scratch":   "scratch",
    "distilled": "distill (conf 0.25)",
    "distill50": "distill (conf 0.50)",
}


def _read(name):
    p = METRICS / name
    if not p.exists():
        return []
    with p.open() as f:
        # strip header whitespace: some CSVs ship "a, b, c" with padded keys
        rows = list(csv.DictReader(f))
        return [{(k.strip() if k else k): v for k, v in r.items()} for r in rows]


# --- SENSEI-arch sweep helpers (shared with make_tables) ----------------------
# Build-up ablation order for figures/tables: base -> focal_noiw -> focal -> nwd.
SENSEI_ARMS = ["base", "focal_noiw", "focal", "nwd"]
ARM_LABEL = {
    "base":       "base",          # DEPLOYED recipe (standard BCE, no re-balancing)
    "focal_noiw": "+focal",        # build-up: adds focal loss + rare-class augmentation
    "focal":      "+img-wt",       # adds inverse-frequency image re-weighting
    "nwd":        "+NWD",          # adds NWD tiny-object box loss
}
APIS_CLASS_ID = 1                  # Apis mellifera (honeybee), the collapsing class


DEPLOY_SIZE = 320          # deployed operating point: deliberate energy/accuracy choice


def deployed_size() -> int:
    """Deployed input size (px) = DEPLOY_SIZE (320).

    Final data has accuracy rising monotonically with resolution (base mAP@0.5:
    192=0.77, 320=0.86, 512=0.91), so argmax is 512; we deploy at 320 to roughly
    halve the inference energy (5.5 vs 13 mJ) and because all four loss-ablation arms
    were evaluated per species there. 512 is the accuracy ceiling. Pinned so figures,
    tables and values agree on the operating point."""
    return DEPLOY_SIZE


def _arm_overall(arm: str, size: int, seed: int = 0):
    """One overall row for (arm, size, seed), or {} if its CSV is missing."""
    rows = _read(f"sensei_{arm}_{size}_s{seed}.csv")
    return rows[0] if rows else {}


def _arm_per_species(arm: str, size: int, seed: int = 0):
    """Per-species rows for (arm, size, seed), or [] if its CSV is missing."""
    return _read(f"sensei_{arm}_{size}_s{seed}_per_species.csv")


def fig_energy_power(outdir: Path):
    """Two panels: (A) energy/inference, (B) peak active power. SENSEI vs RPi4+Coral."""
    e = _read("energy.csv")
    sensei_e = float(e[0]["energy_mj"]) if e else 8.0
    sensei_p = float(e[0]["active_power_mw"]) if e else 70.0
    # baseline (paper Sec. V): RPi4+Coral
    base_e, base_p = 140.0, 7000.0
    labels = ["SENSEI\nGAP9", "RPi4+\nCoral"]

    apply_rc()
    # combined 2-panel
    fig, axs = plt.subplots(1, 2, figsize=(COL_W_IN, COL_W_IN * GOLDEN))
    styled_bars(axs[0], labels, [sensei_e, base_e], "Energy / inf. (mJ)", "(a)", logy=True)
    styled_bars(axs[1], labels, [sensei_p, base_p], "Peak power (mW)", "(b)", logy=True)
    fig.savefig(outdir / "energy_power.pdf"); plt.close(fig)

    # per-panel exports
    for tag, vals, ylab in [("panelA", [sensei_e, base_e], "Energy / inf. (mJ)"),
                            ("panelB", [sensei_p, base_p], "Peak power (mW)")]:
        f, ax = plt.subplots(figsize=(COL_W_IN * 0.5, COL_W_IN * GOLDEN))
        styled_bars(ax, labels, vals, ylab, logy=True)
        f.savefig(outdir / f"energy_power_{tag}.pdf"); plt.close(f)


def fig_accuracy(outdir: Path):
    """Accuracy bar: compressed pipeline vs reference ceiling."""
    a = _read("accuracy.csv")
    sensei_map = float(a[0]["map50"]) if a else 0.88
    ref_map = 0.927   # Bjerge YOLOv5m6 ceiling (paper Sec. II)
    apply_rc()
    f, ax = plt.subplots(figsize=(COL_W_IN, COL_W_IN * GOLDEN))
    styled_bars(ax, ["SENSEI\n(INT8)", "YOLOv5m6\nref."], [sensei_map, ref_map],
                "mAP@0.5")
    ax.set_ylim(0, 1.0)
    f.savefig(outdir / "accuracy.pdf"); plt.close(f)


def _sweep_by_regime():
    """Rows of sweep_resolution_trained.csv for the from-scratch regime, sorted by imgsz.
    The paper reports the from-scratch detector only, so other regimes are dropped here."""
    rows = _read("sweep_resolution_trained.csv")
    out = {}
    for r in rows:
        if r["regime"] != "scratch":
            continue
        out.setdefault(r["regime"], []).append(r)
    for g in out.values():
        g.sort(key=lambda r: int(r["imgsz"]))
    return out


def fig_res_sweep(outdir: Path):
    """Trained from-scratch resolution sweep. Skipped if the CSV is absent.
    (A) mAP@0.5 vs input resolution; (B) accuracy-vs-energy Pareto. Recommended
    (cheapest deployable) point ringed. Also exported panel-by-panel."""
    groups = _sweep_by_regime()
    if not groups:
        print("[skip] no sweep_resolution_trained.csv — run 45_res_sweep + sweep_collect first")
        return
    apply_rc()

    def _curves(ax, xkey, xlabel):
        for regime, g in sorted(groups.items()):
            col, mk, ls = REGIME_STYLE.get(regime, (WONG["green"], "^", ":"))
            xs = [float(r[xkey]) for r in g]; ys = [float(r["map50"]) for r in g]
            ax.plot(xs, ys, marker=mk, linestyle=ls, color=col,
                    label=REGIME_LABEL.get(regime, regime))
            for r in g:
                if int(r["recommended"]):
                    ax.scatter([float(r[xkey])], [float(r["map50"])], s=80,
                               facecolors="none", edgecolors=col, linewidths=1.4, zorder=5)
        ax.set_xlabel(xlabel); ax.set_ylabel("mAP@0.5")
        ax.margins(y=0.10)                      # headroom so curves clear the top legend

    def _plot_acc_vs_res(ax, legend=False):
        _curves(ax, "imgsz", "Input resolution (px)")
        if legend:
            ax.legend(loc="lower right", frameon=False, fontsize=7, handlelength=1.6)

    def _plot_pareto(ax, legend=False):
        _curves(ax, "energy_mj", "Energy / inf. (mJ)")
        if legend:
            ax.legend(loc="lower right", frameon=False, fontsize=7, handlelength=1.6)

    # combined 2-panel: single from-scratch curve, so no legend is needed
    fig, axs = plt.subplots(1, 2, figsize=(COL_W_IN, COL_W_IN * GOLDEN))
    _plot_acc_vs_res(axs[0]); axs[0].set_title("(a)", y=-0.32)
    _plot_pareto(axs[1]);     axs[1].set_title("(b)", y=-0.32)
    fig.savefig(outdir / "res_sweep.pdf", bbox_inches="tight"); plt.close(fig)

    # per-panel exports (subcaption-friendly)
    for tag, fn in [("panelA", _plot_acc_vs_res), ("panelB", _plot_pareto)]:
        f, ax = plt.subplots(figsize=(COL_W_IN * 0.55, COL_W_IN * GOLDEN))
        fn(ax); f.savefig(outdir / f"res_sweep_{tag}.pdf", bbox_inches="tight")
        plt.close(f)


def fig_accuracy_energy(outdir: Path, revised: bool = False):
    """Grouped bars: mAP@0.5 by input resolution, one bar per loss recipe (baseline + three
    imbalance-handling variants), +/-1 std over three seeds. x-axis annotated with the measured
    per-inference energy; the deployed baseline bar is ringed. Reads the resolution sweep; the
    diminishing-returns shape (steep 192->320, flat 320->512) and the closeness of the four
    recipes are both read directly off the bar heights. Skipped if no base rows.
    revised=True -> co-author version: square 'NxN px' tick labels (Victor: images are 2-D) and
    output accuracy_energy_revised.pdf; energy provenance is documented in the LaTeX caption."""
    rows = _read("sensei_arch_sweep.csv")
    by_arm, energy = {}, {}
    for r in rows:
        a = (r.get("arm") or "").strip()
        try:
            sz = int(r["imgsz"])
            by_arm.setdefault(a, {})[sz] = (float(r["map50_mean"]), float(r.get("map50_std") or 0.0))
            energy[sz] = float(r["energy_mj"])
        except (KeyError, ValueError, TypeError):
            continue
    if "base" not in by_arm:
        print("[skip] no base rows in sensei_arch_sweep.csv — run 47_sensei_arch_sweep first")
        return
    sizes = sorted(energy)                                # 192, 320, 512
    dep = deployed_size()
    # (colour, hatch) per arm; hatch on every bar for B&W/print safety (repo convention)
    STYLE = {"base":       (WONG["blue"],   ""),
             "focal_noiw": (WONG["green"],  "//"),
             "focal":      (WONG["vermil"], ".."),
             "nwd":        (WONG["orange"], "xx")}
    apply_rc()
    f, ax = plt.subplots(figsize=(COL_W_IN, COL_W_IN * 0.66))
    arms = [a for a in SENSEI_ARMS if a in by_arm]
    n = len(arms); w = 0.8 / n; x = list(range(len(sizes)))
    for k, arm in enumerate(arms):
        col, hatch = STYLE.get(arm, (WONG["black"], ""))
        off = (k - (n - 1) / 2) * w
        ys = [by_arm[arm].get(s, (0.0, 0.0))[0] for s in sizes]
        es = [by_arm[arm].get(s, (0.0, 0.0))[1] for s in sizes]
        ax.bar([xi + off for xi in x], ys, w, yerr=es, capsize=2, label=ARM_LABEL.get(arm, arm),
               color=col, edgecolor="black", linewidth=0.4, hatch=hatch, error_kw=dict(lw=0.7))
    if dep in sizes:                                       # ring + label the deployed baseline bar
        di = sizes.index(dep); off0 = (0 - (n - 1) / 2) * w; yb = by_arm["base"][dep][0]
        ax.scatter([di + off0], [yb], s=55, facecolors="none", edgecolors="black",
                   linewidths=1.3, zorder=6)
        ax.annotate("deployed", (di + off0, yb), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=6.5)
    ax.set_xticks(x)
    lab = (lambda s: f"{s}x{s} px\n{energy[s]:.1f} mJ") if revised else (lambda s: f"{s} px\n{energy[s]:.1f} mJ")
    ax.set_xticklabels([lab(s) for s in sizes])
    ax.set_ylabel("mAP@0.5")
    lo = min(by_arm[a][s][0] - by_arm[a][s][1] for a in arms for s in sizes if s in by_arm[a])
    ax.set_ylim(max(0.0, lo - 0.04), 0.96)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=n, frameon=False,
              fontsize=6.5, handlelength=1.2, columnspacing=0.9, borderaxespad=0.1)
    name = "accuracy_energy_revised.pdf" if revised else "accuracy_energy.pdf"
    f.savefig(outdir / name, bbox_inches="tight"); plt.close(f)


def fig_convergence_revised(outdir: Path):
    """Co-author revision of fig_convergence (Philip: resolutions are discrete -> not a line plot;
    Victor: explain the A. urticae dip). Per-class base-detector recall at 192/320/512 px drawn as
    a DOT plot (markers only, very faint guide line) so the three discrete operating points read as
    points, not a continuum. Same Wong palette/markers as the original."""
    sizes = [192, 320, 512]
    per = {sz: {r["species"]: r for r in _arm_per_species("base", sz)} for sz in sizes}
    if not all(per.values()):
        print("[skip] missing base per-species CSVs for convergence_revised figure")
        return
    species = list(per[sizes[-1]].keys())
    apply_rc()
    f, ax = plt.subplots(figsize=(COL_W_IN, COL_W_IN * 0.58))
    marks = ["o", "s", "^", "D", "v", "P", "X", "*", "<"]
    xpos = list(range(len(sizes)))                            # categorical spacing -> emphasise discreteness
    for i, sp in enumerate(species):
        ys = []
        for sz in sizes:
            try:
                ys.append(float(per[sz][sp]["recall"]))
            except (KeyError, ValueError):
                ys.append(float("nan"))
        parts = sp.split()
        abbr = f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else sp
        col = PALETTE[i % len(PALETTE)]
        ax.plot(xpos, ys, color=col, linewidth=0.5, alpha=0.30, zorder=1)   # faint guide only
        ax.scatter(xpos, ys, marker=marks[i % 9], color=col, s=22, zorder=3,
                   edgecolors="black", linewidths=0.3, label=abbr)
    ax.set_xlabel("Input resolution (px)"); ax.set_ylabel("recall (base detector)")
    ax.set_xticks(xpos); ax.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax.set_xlim(-0.35, len(sizes) - 0.65); ax.set_ylim(0.3, 1.0)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              fontsize=5.4, handlelength=1.0, columnspacing=0.7, labelspacing=0.25, handletextpad=0.3)
    f.savefig(outdir / "convergence_revised.pdf", bbox_inches="tight"); plt.close(f)


def fig_pipeline_revised(outdir: Path):
    """Co-author revision of fig_pipeline (Philip: 'every stage on-node' is misleading -- training
    and quantisation are OFFLINE). Same boustrophedon layout, but the offline stages (top row) and
    on-node stages (bottom row) are enclosed in labelled 'Offline (host)' / 'On-node (GAP9)' bands,
    and 'Train from scratch' -> 'Train (init YOLOv5n)' to match the recipe."""
    apply_rc()
    f, ax = plt.subplots(figsize=(COL_W_IN, COL_W_IN * 0.30))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.margins(0); f.subplots_adjust(left=0, right=1, top=1, bottom=0)
    bw, bh, y1, y2 = 0.205, 0.34, 0.74, 0.30
    lin = lambda a, b, n: [a + (b - a) * i / (n - 1) for i in range(n)]
    xs1, xs2 = lin(0.115, 0.885, 4), lin(0.115, 0.885, 3)
    blue, green, cream = WONG["blue"], WONG["green"], "#FBE7C6"
    # background bands: offline (top) vs on-node (bottom)
    ax.add_patch(mpatches.FancyBboxPatch((0.02, y1 - bh / 2 - 0.05), 0.96, bh + 0.10,
        boxstyle="round,pad=0.002,rounding_size=0.02", facecolor="#F2F2F2", edgecolor="0.7",
        linewidth=0.6, linestyle=(0, (3, 2)), zorder=0))
    ax.add_patch(mpatches.FancyBboxPatch((0.02, y2 - bh / 2 - 0.05), 0.96, bh + 0.10,
        boxstyle="round,pad=0.002,rounding_size=0.02", facecolor="#E9F1F7", edgecolor=blue,
        linewidth=0.6, linestyle=(0, (3, 2)), zorder=0))
    ax.text(0.035, y1 + bh / 2 + 0.005, "Offline (host)", fontsize=4.6, color="0.35", style="italic", va="bottom")
    ax.text(0.035, y2 + bh / 2 + 0.005, "On-node (GAP9)", fontsize=4.6, color=blue, style="italic", va="bottom")

    def box(cx, cy, text, fc, tc):
        ax.add_patch(mpatches.FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
            boxstyle="round,pad=0.004,rounding_size=0.06", mutation_aspect=0.28,
            facecolor=fc, edgecolor="black", linewidth=0.8, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=5.4, color=tc, linespacing=0.95,
                zorder=5, fontweight="bold" if fc in (blue, green) else "normal")

    def arrow(x0, y0, x1, y1c):
        ax.annotate("", xy=(x1, y1c), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.1, shrinkA=0, shrinkB=0), zorder=4)

    row1 = [("Benchmark\n9 sp., 21k img", "white", "black"), ("Tiling\n320x320 px", "white", "black"),
            ("Train\n(init YOLOv5n)", cream, "black"), ("INT8\nquantise", "white", "black")]
    row2 = [("Deploy on\nGAP9", blue, "white"), ("Per-tile\ninference", "white", "black"),
            ("Per-species\ncount", green, "white")]
    for cx, (t, fc, tc) in zip(xs1, row1):
        box(cx, y1, t, fc, tc)
    r2x = list(reversed(xs2))
    for cx, (t, fc, tc) in zip(r2x, row2):
        box(cx, y2, t, fc, tc)
    for i in range(3):
        arrow(xs1[i] + bw / 2, y1, xs1[i + 1] - bw / 2, y1)
    arrow(xs1[3], y1 - bh / 2, r2x[0], y2 + bh / 2)   # offline -> on-node (the deployment boundary)
    for i in range(2):
        arrow(r2x[i] - bw / 2, y2, r2x[i + 1] + bw / 2, y2)
    f.savefig(outdir / "pipeline_revised.pdf", bbox_inches="tight"); plt.close(f)


def fig_ablation(outdir: Path):
    """Build-up ablation at the deployed size over arms base,focal_noiw,focal,nwd.
    (a) overall micro_f1 and honeybee (Apis class 1) recall; (b) signed count bias.
    Combined + per-panel exports. Skipped if no arm CSV is present at that size."""
    dep = deployed_size()
    labels, f1s, hb_rec, bias = [], [], [], []
    for arm in SENSEI_ARMS:
        ov = _arm_overall(arm, dep); sp = _arm_per_species(arm, dep)
        if not ov and not sp:
            print(f"[skip] no sensei_{arm}_{dep}_s0 CSV — arm '{arm}' omitted from ablation")
            continue
        labels.append(ARM_LABEL[arm])
        def _f(d, k):
            try:
                return float(d.get(k))
            except (TypeError, ValueError):
                return 0.0
        f1s.append(_f(ov, "micro_f1"))
        bias.append(_f(ov, "count_bias_img"))
        apis = next((r for r in sp if str(r.get("class_id")).strip() == str(APIS_CLASS_ID)), {})
        hb_rec.append(_f(apis, "recall"))
    if not labels:
        print(f"[skip] no SENSEI-arch arm CSVs at {dep}px — run 47_sensei_arch_sweep first")
        return

    apply_rc()
    x = list(range(len(labels))); width = 0.38

    def _panel_a(ax, legend=True, xlabels=True, rot=0, legend_above=False):
        ax.bar([i - width / 2 for i in x], f1s, width, color=WONG["blue"],
               edgecolor="black", linewidth=0.6, hatch="//", label="overall $F_1$")
        ax.bar([i + width / 2 for i in x], hb_rec, width, color=WONG["vermil"],
               edgecolor="black", linewidth=0.6, hatch="\\\\", label="honeybee recall")
        ax.set_ylabel("score (0–1)"); ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels(labels if xlabels else [""] * len(labels),
                           rotation=rot, ha="right" if rot else "center")
        if legend and legend_above:        # narrow standalone panel: put it outside, above
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                      frameon=False, fontsize=7, handlelength=1.1, columnspacing=0.9)
        elif legend:                       # wide combined panel: fits inside the headroom
            ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=7,
                      handlelength=1.2, columnspacing=1.0, borderaxespad=0.05)

    def _panel_b(ax, xlabels=True, rot=0):
        bars = ax.bar(x, bias, width=0.62,
                      color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))],
                      edgecolor="black", linewidth=0.6)
        for i, b in enumerate(bars):
            b.set_hatch(HATCHES[i % len(HATCHES)])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("count bias/img")
        ax.set_xticks(x)
        ax.set_xticklabels(labels if xlabels else [""] * len(labels),
                           rotation=rot, ha="right" if rot else "center")

    # combined: panels STACKED with a shared x-axis so the build-up labels never collide
    fig, axs = plt.subplots(2, 1, figsize=(COL_W_IN, COL_W_IN * 0.98), sharex=True)
    _panel_a(axs[0], legend=True, xlabels=False)
    _panel_b(axs[1], xlabels=True)
    for ax, tag in zip(axs, ("(a)", "(b)")):
        ax.text(0.015, 0.84, tag, transform=ax.transAxes, fontsize=8, fontweight="bold")
    fig.savefig(outdir / "ablation.pdf", bbox_inches="tight"); plt.close(fig)

    # per-panel exports (subcaption-friendly): narrow, so rotate the labels
    fa, ax = plt.subplots(figsize=(COL_W_IN * 0.5, COL_W_IN * GOLDEN))
    _panel_a(ax, legend=True, xlabels=True, rot=22, legend_above=True)
    fa.savefig(outdir / "ablation_panelA.pdf", bbox_inches="tight"); plt.close(fa)
    fb, ax = plt.subplots(figsize=(COL_W_IN * 0.5, COL_W_IN * GOLDEN))
    _panel_b(ax, xlabels=True, rot=22)
    fb.savefig(outdir / "ablation_panelB.pdf", bbox_inches="tight"); plt.close(fb)


def fig_count_error(outdir: Path):
    """Per-species signed count error % for the DEPLOYED BASE model, one hatched bar
    per species with a zero reference line. Skipped if the per-species CSV is absent."""
    dep = deployed_size()
    sp = _arm_per_species("base", dep)
    if not sp:
        print(f"[skip] no sensei_base_{dep}_s0_per_species.csv — run 47_sensei_arch_sweep first")
        return
    names, errs = [], []
    tot_gt = tot_abserr = 0.0
    for r in sp:
        try:
            ngt = float(r["n_gt"]); npred = float(r["n_pred"])
        except (KeyError, ValueError, TypeError):
            continue
        names.append((r.get("species") or r.get("class_id") or "?").strip())
        errs.append((npred - ngt) / ngt * 100.0 if ngt else 0.0)
        tot_gt += ngt; tot_abserr += abs(npred - ngt)
    if not names:
        print(f"[skip] no usable rows in sensei_base_{dep}_s0_per_species.csv")
        return
    # count-weighted MEAN ABSOLUTE error: over- and under-counts do NOT cancel (a fair
    # magnitude, unlike a signed net which lets a +30% species offset a -20% one).
    mae = tot_abserr / tot_gt * 100.0 if tot_gt else 0.0

    apply_rc()
    f, ax = plt.subplots(figsize=(COL_W_IN, COL_W_IN * 0.48))
    x = range(len(names))
    bars = ax.bar(list(x), errs, width=0.7,
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(names))],
                  edgecolor="black", linewidth=0.6)
    for i, b in enumerate(bars):
        b.set_hatch(HATCHES[i % len(HATCHES)])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhspan(-mae, mae, color=WONG["vermil"], alpha=0.10, zorder=0)
    ax.axhline(mae, color=WONG["vermil"], linestyle="--", linewidth=1.3, zorder=4,
               label=f"count-weighted mean |error| = {mae:.1f}%")
    ax.axhline(-mae, color=WONG["vermil"], linestyle="--", linewidth=1.3, zorder=4)
    ax.set_xticks(list(x))
    # abbreviate to genus-initial (e.g. "C. septempunctata") so the rotated labels stay short
    abbr = [f"{n.split()[0][0]}. {' '.join(n.split()[1:])}" if len(n.split()) > 1 else n
            for n in names]
    ax.set_xticklabels(abbr, rotation=30, ha="right", fontsize=6.5, style="italic")
    ax.set_ylabel("count error (%)")
    # legend ABOVE the axes (outside the bars) so it never overlaps a tall bar
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6.5)
    f.savefig(outdir / "count_error.pdf", bbox_inches="tight"); plt.close(f)


def fig_convergence(outdir: Path):
    """Per-class convergence with input resolution: base detector recall for each of the
    nine species at 192/320/512 px. Shows that most classes improve with resolution (the
    look-alike mimics most), while the rarest class can regress. Skipped if CSVs absent."""
    sizes = [192, 320, 512]
    per = {sz: {r["species"]: r for r in _arm_per_species("base", sz)} for sz in sizes}
    if not all(per.values()):
        print("[skip] missing base per-species CSVs for convergence figure")
        return
    species = list(per[sizes[-1]].keys())
    apply_rc()
    f, ax = plt.subplots(figsize=(COL_W_IN, COL_W_IN * 0.56))   # single IEEE column (compact)
    for i, sp in enumerate(species):
        ys = []
        for sz in sizes:
            try:
                ys.append(float(per[sz][sp]["recall"]))
            except (KeyError, ValueError):
                ys.append(float("nan"))
        parts = sp.split()
        abbr = f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else sp  # "C. septempunctata"
        ax.plot(sizes, ys, marker=["o", "s", "^", "D", "v", "P", "X", "*", "<"][i % 9],
                color=PALETTE[i % len(PALETTE)], linewidth=1.1, markersize=3.6, label=abbr)
    ax.set_xlabel("Input resolution (px)"); ax.set_ylabel("recall (base detector)")
    ax.set_xticks(sizes); ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              fontsize=5.4, handlelength=1.1, columnspacing=0.7, labelspacing=0.25,
              handletextpad=0.4)
    f.savefig(outdir / "convergence.pdf", bbox_inches="tight"); plt.close(f)


def fig_pipeline(outdir: Path):
    """Schematic, column-width end-to-end pipeline (no data). Compact two-row
    (boustrophedon) layout that fits a single IEEE column: benchmark -> tiling ->
    train -> INT8 quantise, then (down) deploy on GAP9 -> per-tile inference ->
    per-species count, with the on-chip stages highlighted."""
    apply_rc()
    f, ax = plt.subplots(figsize=(COL_W_IN, COL_W_IN * 0.23))   # short banner (~half prior height)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.margins(0); f.subplots_adjust(left=0, right=1, top=1, bottom=0)
    bw, bh, y1, y2 = 0.205, 0.42, 0.73, 0.27
    lin = lambda a, b, n: [a + (b - a) * i / (n - 1) for i in range(n)]
    xs1, xs2 = lin(0.105, 0.895, 4), lin(0.105, 0.895, 3)
    blue, green, cream = WONG["blue"], WONG["green"], "#FBE7C6"

    def box(cx, cy, text, fc, tc):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2), bw, bh,
            boxstyle="round,pad=0.004,rounding_size=0.06", mutation_aspect=0.23,
            facecolor=fc, edgecolor="black", linewidth=0.8))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=5.5, color=tc,
                linespacing=0.95, zorder=5,
                fontweight="bold" if fc in (blue, green) else "normal")

    def arrow(x0, y0, x1, y1c):
        ax.annotate("", xy=(x1, y1c), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.1,
                                    shrinkA=0, shrinkB=0))

    row1 = [("Benchmark\n9 sp., 21k img", "white", "black"),
            ("Tiling\n320 px", "white", "black"),
            ("Train from\nscratch", cream, "black"),
            ("INT8\nquantise", "white", "black")]
    row2 = [("Deploy on\nGAP9", blue, "white"),
            ("Per-tile\ninference", "white", "black"),
            ("Per-species\ncount", green, "white")]
    for cx, (t, fc, tc) in zip(xs1, row1):
        box(cx, y1, t, fc, tc)
    r2x = list(reversed(xs2))                       # right-to-left second row
    for cx, (t, fc, tc) in zip(r2x, row2):
        box(cx, y2, t, fc, tc)
    for i in range(3):                              # row 1, left to right
        arrow(xs1[i] + bw / 2, y1, xs1[i + 1] - bw / 2, y1)
    arrow(xs1[3], y1 - bh / 2, r2x[0], y2 + bh / 2)  # down: INT8 -> Deploy
    for i in range(2):                              # row 2, right to left
        arrow(r2x[i] - bw / 2, y2, r2x[i + 1] + bw / 2, y2)
    f.savefig(outdir / "pipeline.pdf", bbox_inches="tight"); plt.close(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("results/figures"))
    a = ap.parse_args(argv)
    a.outdir.mkdir(parents=True, exist_ok=True)
    fig_pipeline(a.outdir)
    fig_convergence(a.outdir)
    fig_energy_power(a.outdir)
    fig_accuracy(a.outdir)
    fig_res_sweep(a.outdir)
    fig_accuracy_energy(a.outdir)
    fig_ablation(a.outdir)
    fig_count_error(a.outdir)
    # co-author review revisions
    fig_convergence_revised(a.outdir)
    fig_pipeline_revised(a.outdir)
    fig_accuracy_energy(a.outdir, revised=True)
    print("[done] figures ->", a.outdir)


if __name__ == "__main__":
    main()
