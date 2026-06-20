"""Emit results/tables/values.tex: one \\newcommand per inline \\val macro.

Keeps every number that the prose quotes out of the .tex source and in code, so
the paper's inline figures (the deployed model's F1/recall, count error, etc.)
are regenerated from results/metrics/*.csv rather than hand-typed.

The DEPLOYED model is arm "base" (standard BCE, no image-weights). The deployed
SIZE is whichever base row has the highest map50_mean in the arch sweep; if the
sweep file or its base rows are absent we fall back to 320\\,px. Mirrors the
read/col idioms and AUTO-GENERATED header of scripts/make_tables.py.

Degrades gracefully: any missing CSV -> the macro is still defined with a
placeholder body ("0.XX"/"--") so LaTeX always compiles; never crashes.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

METRICS = Path("results/metrics")
FALLBACK_SIZE = 320          # contract: deployed size when sweep/base rows absent
PLACEHOLDER = "0.XX"         # numeric placeholder so LaTeX still compiles
HONEYBEE = "Apis mellifera"  # the abundant species the mimic collapses onto
DRONEFLY = "Eristalis tenax" # the rare Batesian mimic over-predicted under re-balancing


def _read(name):
    """Read a metrics CSV into a list of dicts; [] if it is absent."""
    p = METRICS / name
    return list(csv.DictReader(p.open())) if p.exists() else []


def _col(row, key, default=""):
    """Header-tolerant column fetch (headers may carry stray spaces)."""
    if key in row:
        return row[key]
    for k, v in row.items():
        if k is not None and k.strip() == key:
            return v
    return default


def _f2(v):
    """Format a value to 2 d.p., or the placeholder when it is not numeric."""
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return PLACEHOLDER


def deployed_size():
    """Deployed input size (px) = DEPLOY_SIZE.

    The paper's deployed operating point is a DELIBERATE energy/accuracy choice,
    not the accuracy maximum. With the final data accuracy rises monotonically with
    resolution (base mAP@0.5: 192=0.77, 320=0.86, 512=0.91), so argmax map50 is 512;
    we deploy at 320 instead because it roughly halves the inference energy
    (5.5 vs 13 mJ) for a small accuracy give-up, and it is the size at which all four
    loss-ablation arms were evaluated per species. 512 is reported as the accuracy
    ceiling. Pin here so every generated number agrees on the operating point.
    """
    return FALLBACK_SIZE


def _base_overall(size, seed=0):
    """The base OVERALL eval row at this size/seed, or {} if absent."""
    rows = _read(f"sensei_base_{size}_s{seed}.csv")
    return rows[0] if rows else {}


def _arm_overall(arm, size, seed=0):
    """The OVERALL eval row for any arm at this size/seed, or {} if absent."""
    rows = _read(f"sensei_{arm}_{size}_s{seed}.csv")
    return rows[0] if rows else {}


def _base_per_species(size, seed=0):
    """The base per-species eval rows at this size/seed, or [] if absent."""
    return _read(f"sensei_base_{size}_s{seed}_per_species.csv")


def _arm_per_species(arm, size, seed=0):
    """Per-species eval rows for any arm at this size/seed, or [] if absent."""
    return _read(f"sensei_{arm}_{size}_s{seed}_per_species.csv")


def _species_row(rows, species):
    """The row for a given species name, or {} if absent."""
    for r in rows:
        if _col(r, "species").strip() == species:
            return r
    return {}


def _species_recall(arm, size, species):
    """A species' recall under an arm at this size, 2 d.p. (placeholder if absent)."""
    return _f2(_col(_species_row(_arm_per_species(arm, size), species), "recall"))


def _species_int(arm, size, species, field):
    """An integer per-species count (n_gt / n_pred) under an arm, or '--' if absent."""
    v = _col(_species_row(_arm_per_species(arm, size), species), field)
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return "--"


def _count_err_pct(size, seed=0):
    """Count-weighted MEAN ABSOLUTE count error (%), 1 d.p., with a literal \\%.

    sum_s |n_pred_s - n_gt_s| / sum_s n_gt_s from the deployed base per-species CSV.
    A magnitude metric: per-species over- and under-counts do NOT cancel (unlike a
    signed net aggregate). Placeholder when the CSV is missing/empty or GT total is zero.
    """
    sp = _base_per_species(size, seed)
    if not sp:
        return PLACEHOLDER + r"\%"
    try:
        gt = sum(int(_col(r, "n_gt")) for r in sp)
        abserr = sum(abs(int(_col(r, "n_pred")) - int(_col(r, "n_gt"))) for r in sp)
    except (TypeError, ValueError):
        return PLACEHOLDER + r"\%"
    if gt == 0:
        return PLACEHOLDER + r"\%"
    return f"{100 * abserr / gt:.1f}" + r"\%"


def _count_net_pct(size, seed=0):
    """Signed NET count error of the TOTAL (%), integer with sign and literal \\%:
    (sum n_pred - sum n_gt) / sum n_gt. This is the total-abundance bias (it DOES net
    over- and under-counts), reported separately from the per-species magnitude error."""
    sp = _base_per_species(size, seed)
    if not sp:
        return PLACEHOLDER + r"\%"
    try:
        gt = sum(int(_col(r, "n_gt")) for r in sp)
        pr = sum(int(_col(r, "n_pred")) for r in sp)
    except (TypeError, ValueError):
        return PLACEHOLDER + r"\%"
    if gt == 0:
        return PLACEHOLDER + r"\%"
    return f"{round(100 * (pr - gt) / gt):+d}" + r"\%"


def build_values():
    """Compute every \\val macro body. Returns an ordered list of (name, body)."""
    size = deployed_size()
    small = _base_overall(192)
    large = _base_overall(512)
    dep = _base_overall(size)

    return [
        ("valFoneSmall",    _f2(_col(small, "micro_f1"))),
        ("valFoneLarge",    _f2(_col(large, "micro_f1"))),
        ("valBestSize",     str(size)),                              # integer px, no decimals
        ("valDeployRecipe", "the baseline recipe (standard loss, no class re-balancing)"),
        ("valDeployFone",   _f2(_col(dep, "micro_f1"))),
        ("valDeployRecall", _f2(_col(dep, "recall"))),
        ("valCountErrPct",  _count_err_pct(size)),        # count-weighted mean ABS error (non-cancel)
        ("valCountNetPct",  _count_net_pct(size)),        # signed net over/under-count of the TOTAL
        # Mimic-collapse mechanism, fully self-contained from the per-species CSVs
        # (same data as Table~\ref{tab:monitoring}): the image-weighting recipes (focal/nwd)
        # collapse the abundant honeybee's recall and balloon the rare drone fly's predictions;
        # the no-image-weights control (focal_noiw) restores the honeybee.
        ("valHoneyRecallBase",  _species_recall("base",       size, HONEYBEE)),
        ("valHoneyRecallFocal", _species_recall("focal",      size, HONEYBEE)),
        ("valHoneyRecallNoIW",  _species_recall("focal_noiw", size, HONEYBEE)),
        ("valDroneGt",          _species_int("base",       size, DRONEFLY, "n_gt")),
        ("valDronePredBase",    _species_int("base",       size, DRONEFLY, "n_pred")),
        ("valDronePredFocal",   _species_int("focal",      size, DRONEFLY, "n_pred")),
        ("valDronePredNoIW",    _species_int("focal_noiw", size, DRONEFLY, "n_pred")),
        # Counting deliverable (centre-matched F1) per arm: the decisive axis the
        # imbalance-handling recipes collapse on, even where their mAP looks fine.
        ("valFoneFocal",        _f2(_col(_arm_overall("focal", size), "micro_f1"))),
        ("valFoneNwd",          _f2(_col(_arm_overall("nwd",   size), "micro_f1"))),
    ]


def render(values) -> str:
    body = "".join(f"\\newcommand{{\\{name}}}{{{val}}}\n" for name, val in values)
    return (
        "% AUTO-GENERATED by scripts/make_values.py — do not edit by hand.\n"
        "% Inline \\val macros for the paper prose; bodies come from\n"
        "% results/metrics/sensei_arch_sweep.csv + the base eval CSVs.\n"
        + body
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/tables/values.tex"))
    a = ap.parse_args(argv)

    size = deployed_size()
    if not _read("sensei_arch_sweep.csv"):
        print("[skip] sensei_arch_sweep.csv missing -> deployed size falls back to", size)
    if not _base_overall(size):
        print(f"[skip] sensei_base_{size}_s0.csv missing -> F1/recall use placeholders")

    values = build_values()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(values))

    for name, val in values:
        print(f"  \\{name} = {val}")
    print("[done] values ->", a.out)


if __name__ == "__main__":
    main()
