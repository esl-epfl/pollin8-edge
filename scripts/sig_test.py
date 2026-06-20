"""Significance of the loss-recipe ablation: is the baseline's accuracy lead real (n=3 seeds)?

Welch's two-sample t-test (unequal variance) of base vs each imbalance-handling variant at every
input size, computed from the per-(arm,size) mean / std / n in sensei_arch_sweep.csv. The sweep
stores a POPULATION std (divided by n); we convert to the sample std (n-1) the t-test needs.
Student-t tail probability is evaluated with a stdlib regularized incomplete beta (no SciPy).

Output: results/metrics/significance.csv  (size, variant, base_map, var_map, delta, t, df, p,
significant) — one row per (size, variant), so the ablation caption can cite "all p < 0.01".
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path

METRICS = Path("results/metrics")
BASELINE = "base"


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta (Numerical Recipes betacf)."""
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    """Regularized incomplete beta I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_two_sided_p(t, df):
    """Two-sided p-value of Student-t with df degrees of freedom."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)        # = P(|T| >= |t|)


def welch(m1, s1, n1, m2, s2, n2):
    """Welch's t and Welch-Satterthwaite df from sample means/stds/n."""
    v1, v2 = s1 * s1 / n1, s2 * s2 / n2
    se = math.sqrt(v1 + v2)
    if se == 0:
        return float("inf"), float(n1 + n2 - 2)
    t = (m1 - m2) / se
    df = (v1 + v2) ** 2 / (v1 * v1 / (n1 - 1) + v2 * v2 / (n2 - 1)) if (n1 > 1 and n2 > 1) else 1.0
    return t, df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default=str(METRICS / "sensei_arch_sweep.csv"))
    ap.add_argument("--out", default=str(METRICS / "significance.csv"))
    a = ap.parse_args(argv)

    rows = list(csv.DictReader(open(a.sweep)))
    by_size = {}
    for r in rows:
        try:
            by_size.setdefault(int(r["imgsz"]), {})[r["arm"].strip()] = (
                float(r["map50_mean"]), float(r["map50_std"]), int(r["n_seeds"]))
        except (KeyError, ValueError):
            continue

    out = []
    for size in sorted(by_size):
        arms = by_size[size]
        if BASELINE not in arms:
            continue
        m1, sd1, n1 = arms[BASELINE]
        s1 = sd1 * math.sqrt(n1 / (n1 - 1)) if n1 > 1 else sd1   # population -> sample std
        for arm, (m2, sd2, n2) in arms.items():
            if arm == BASELINE:
                continue
            s2 = sd2 * math.sqrt(n2 / (n2 - 1)) if n2 > 1 else sd2
            t, df = welch(m1, s1, n1, m2, s2, n2)
            p = _t_two_sided_p(t, df)
            out.append(dict(size=size, variant=arm, base_map=round(m1, 4),
                            var_map=round(m2, 4), delta=round(m1 - m2, 4),
                            t=round(t, 3), df=round(df, 2),
                            p_value=f"{p:.2e}", significant=bool(p < 0.01)))

    if not out:
        raise SystemExit(f"[sig_test] no baseline rows in {a.sweep}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    allp = [float(r["p_value"]) for r in out]
    print(f"[done] {len(out)} comparisons -> {a.out}")
    print(f"  base vs all variants: max p = {max(allp):.2e}  ({'ALL p<0.01' if max(allp) < 0.01 else 'some not significant — inspect'})")
    for r in out:
        print(f"  {r['size']}px  base({r['base_map']}) vs {r['variant']}({r['var_map']})"
              f"  Δ={r['delta']:+.3f}  t={r['t']}  df={r['df']}  p={r['p_value']}"
              f"{'  *' if r['significant'] else ''}")


if __name__ == "__main__":
    main()
