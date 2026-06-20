"""Normalised confusion matrix, centre-matched (consistent with the monitoring metrics).

Each prediction is matched to the nearest ground-truth centre within `match_dist`,
regardless of class, so cross-species confusions are visible; the (true, predicted) class
pair is tallied. A ``missed`` column holds undetected ground truth (false negatives) and a
``background`` row holds false positives. Rows are normalised to sum to 1 (the standard
per-true-class / recall view). Saves a PDF figure with species names.

The geometry (build_confusion, row_normalise) is pure and unit-tested; Ultralytics is
imported lazily only to run the prediction pass.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
from . import monitor_metrics as MM

# abbreviated Bjerge-1201 names for compact axis labels
SPECIES = ["C. septempunctata", "A. mellifera", "B. lapidarius", "B. terrestris",
           "E. corolla", "E. balteatus", "A. urticae", "V. vulgaris", "E. tenax"]


def build_confusion(per_image, n, match_dist):
    """per_image: [(preds, gts)] with preds/gts = [(cls,xc,yc), ...] (preds conf-filtered).
    Returns an (n+1)x(n+1) count matrix: rows 0..n-1 = true classes, row n = background
    (false positives); cols 0..n-1 = predicted classes, col n = missed (false negatives)."""
    M = [[0] * (n + 1) for _ in range(n + 1)]
    for preds, gts in per_image:
        pairs = []
        for i, (pc, px, py) in enumerate(preds):
            for j, (gc, gx, gy) in enumerate(gts):
                d = math.hypot(px - gx, py - gy)
                if d <= match_dist:
                    pairs.append((d, i, j))
        pairs.sort()                       # class-agnostic, nearest-first, greedy 1-to-1
        used_p, used_g = set(), set()
        for _d, i, j in pairs:
            if i in used_p or j in used_g:
                continue
            used_p.add(i); used_g.add(j)
            M[gts[j][0]][preds[i][0]] += 1      # true gt class -> predicted class
        for j, (gc, _x, _y) in enumerate(gts):
            if j not in used_g:
                M[gc][n] += 1                   # undetected -> "missed" column
        for i, (pc, _x, _y) in enumerate(preds):
            if i not in used_p:
                M[n][pc] += 1                   # false positive -> "background" row
    return M


def row_normalise(M):
    out = []
    for row in M:
        s = sum(row)
        out.append([(v / s if s else 0.0) for v in row])
    return out


def _normalised_for(model, imgs, lbl_dir, imgsz, conf, match_dist, n, backend="ultralytics"):
    cache = MM.predict_cache(model, imgs, lbl_dir, imgsz, conf, backend)
    return row_normalise(build_confusion(MM._strip_conf(cache, conf), n, match_dist))


def _draw(ax, Mn, labels, n, title=None, ticks=True):
    im = ax.imshow(Mn, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n + 1)); ax.set_yticks(range(n + 1))
    if ticks:
        ax.set_xticklabels(labels + ["missed"], rotation=45, ha="right", fontsize=5)
        ax.set_yticklabels(labels + ["bg"], fontsize=5)
    else:
        ax.set_xticklabels([]); ax.set_yticklabels([])
    if title:
        ax.set_title(title, fontsize=8)
    for i in range(n + 1):
        for j in range(n + 1):
            v = Mn[i][j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4,
                        color="white" if v > 0.5 else "black")
    return im


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/pico_tiled/weights/best.pt")
    ap.add_argument("--data", default="configs/insects1201.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--conf", type=float, default=0.45,
                    help="operating confidence (use the monitoring-tuned value)")
    ap.add_argument("--match-dist", type=float, default=0.05)
    ap.add_argument("--out", default="results/figures/confusion_matrix.pdf")
    # sweep (multi-panel) mode: one confusion matrix per trained resolution
    ap.add_argument("--sweep-dir", default=None,
                    help="dir holding sweep_<regime>_<size>/weights/best.pt (multi-panel mode)")
    ap.add_argument("--regime", default="scratch")
    ap.add_argument("--sizes", type=int, nargs="*", default=[128, 160, 224, 320])
    ap.add_argument("--backend", default="ultralytics",
                    choices=["ultralytics", "yolov5-classic"])
    ap.add_argument("--yolov5-repo", default=None,
                    help="local clone of ultralytics/yolov5 (for --backend yolov5-classic)")
    a = ap.parse_args(argv)
    if a.backend == "yolov5-classic" and not a.yolov5_repo:
        raise SystemExit("[confusion] --backend yolov5-classic requires --yolov5-repo <local clone>")

    img_dir, lbl_dir, names = MM._resolve(a.data, a.split, None)
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in MM.IMG_EXTS)
    if not imgs:
        raise SystemExit(f"[confusion] no images under {img_dir}")
    classes = sorted(names.keys()) if isinstance(names, dict) else list(range(9))
    n = len(classes)
    labels = SPECIES if n == 9 else [str(c) for c in classes]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if a.sweep_dir:                          # ---- multi-panel: all trained sizes ----
        sizes = a.sizes
        ncol = 2; nrow = (len(sizes) + 1) // 2
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 4.4, nrow * 4.2))
        axs = axs.flatten()
        im = None
        for k, sz in enumerate(sizes):
            w = Path(a.sweep_dir) / f"sweep_{a.regime}_{sz}" / "weights" / "best.pt"
            if not w.is_file():
                print(f"[confusion] missing {w}; skipping {sz}px"); axs[k].axis("off"); continue
            model = MM.build_model(str(w), a.backend, a.yolov5_repo)
            Mn = _normalised_for(model, imgs, lbl_dir, sz, a.conf, a.match_dist, n, a.backend)
            im = _draw(axs[k], Mn, labels, n, title=f"{a.regime} @ {sz}\\,px",
                       ticks=(k % ncol == 0 or k >= len(sizes) - ncol))
        for k in range(len(sizes), len(axs)):
            axs[k].axis("off")
        if im is not None:
            fig.colorbar(im, ax=axs.tolist(), fraction=0.03, pad=0.02)
        out = a.out.replace(".pdf", "_sweep.pdf")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"[done] confusion matrices for {a.regime} {sizes} -> {out}")
        return

    # ---- single model ----
    model = MM.build_model(a.weights, a.backend, a.yolov5_repo)
    Mn = _normalised_for(model, imgs, lbl_dir, a.imgsz, a.conf, a.match_dist, n, a.backend)
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    im = _draw(ax, Mn, labels, n)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"[done] normalised confusion matrix ({a.split}, conf={a.conf}) -> {a.out}")


if __name__ == "__main__":
    main()
