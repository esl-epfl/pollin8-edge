"""Monitoring-aligned metrics for insect detection/counting (localisation-agnostic).

mAP@0.5 penalises loose boxes even when an insect is correctly found and counted. For
a monitoring system whose deliverable is detection, counting and tracking---not precise
box position---we report instead:

  * centre-matched F1 (precision/recall): a prediction is a true positive when its CENTRE
    lies within `match_dist` (normalised) of an unmatched ground-truth centre of the SAME
    class (greedy one-to-one), as used for centroid detectors such as FOMO. No IoU.
  * counting error: per-image MAE/RMSE between predicted and true counts (overall and
    per species) --- the quantity ecological trend analysis consumes.
  * per-species R^2: coefficient of determination between predicted and true per-image
    counts, i.e. how well abundance trends are tracked.

The matching/counting/R^2 are pure functions (unit-tested); Ultralytics is imported lazily
only to run the prediction pass. No retraining: this is a single forward pass over the
locked test split. Emits results/metrics/monitoring.csv and monitoring_per_species.csv.
"""
from __future__ import annotations
import argparse, csv, math
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}
OUT = Path("results/metrics")


# ---- pure metric algebra (unit-tested) --------------------------------------
def match_centres(preds, gts, match_dist):
    """Greedy one-to-one, class-aware centre matching.

    preds, gts: lists of (cls, xc, yc) in normalised coords.
    Returns (tp_by_class, fp_by_class, fn_by_class) dicts.
    """
    pairs = []
    for i, (pc, px, py) in enumerate(preds):
        for j, (gc, gx, gy) in enumerate(gts):
            if pc == gc:
                d = math.hypot(px - gx, py - gy)
                if d <= match_dist:
                    pairs.append((d, i, j))
    pairs.sort()                               # nearest first
    used_p, used_g = set(), set()
    tp = {}
    for _d, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        c = preds[i][0]; tp[c] = tp.get(c, 0) + 1
    fp, fn = {}, {}
    for i, (pc, _x, _y) in enumerate(preds):
        if i not in used_p:
            fp[pc] = fp.get(pc, 0) + 1
    for j, (gc, _x, _y) in enumerate(gts):
        if j not in used_g:
            fn[gc] = fn.get(gc, 0) + 1
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def r2_score(true, pred):
    """Coefficient of determination of pred vs true (per-image counts)."""
    n = len(true)
    if n == 0:
        return float("nan")
    mt = sum(true) / n
    ss_tot = sum((t - mt) ** 2 for t in true)
    ss_res = sum((t - p) ** 2 for t, p in zip(true, pred))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def aggregate(per_image, classes, match_dist):
    """per_image: list of (preds, gts) with preds/gts = [(cls,xc,yc),...].
    Returns (overall_dict, per_species_rows)."""
    TP = {c: 0 for c in classes}; FP = dict(TP); FN = dict(TP)
    true_counts = {c: [] for c in classes}; pred_counts = {c: [] for c in classes}
    img_abs, img_sq, img_signed = [], [], []   # per-image TOTAL-count errors (abs, sq, signed)
    for preds, gts in per_image:
        tp, fp, fn = match_centres(preds, gts, match_dist)
        for c in classes:
            TP[c] += tp.get(c, 0); FP[c] += fp.get(c, 0); FN[c] += fn.get(c, 0)
            pc = sum(1 for k, *_ in preds if k == c)
            gc = sum(1 for k, *_ in gts if k == c)
            pred_counts[c].append(pc); true_counts[c].append(gc)
        tot_err = sum(1 for _ in preds) - sum(1 for _ in gts)   # +ve = over-count
        img_abs.append(abs(tot_err)); img_sq.append(tot_err ** 2); img_signed.append(tot_err)

    rows = []
    for c in classes:
        p, r, f1 = prf(TP[c], FP[c], FN[c])
        errs = [pp - tt for pp, tt in zip(pred_counts[c], true_counts[c])]
        n = len(errs) or 1
        rows.append(dict(
            class_id=c, n_gt=sum(true_counts[c]), n_pred=sum(pred_counts[c]),
            tp=TP[c], fp=FP[c], fn=FN[c],
            precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
            count_mae=round(sum(abs(e) for e in errs) / n, 4),
            count_rmse=round(math.sqrt(sum(e * e for e in errs) / n), 4),
            r2=round(r2_score(true_counts[c], pred_counts[c]), 4),
        ))
    sumTP, sumFP, sumFN = sum(TP.values()), sum(FP.values()), sum(FN.values())
    mp, mr, micro_f1 = prf(sumTP, sumFP, sumFN)
    macro_f1 = sum(x["f1"] for x in rows) / (len(rows) or 1)
    ni = len(img_abs) or 1
    overall = dict(
        micro_f1=round(micro_f1, 4), precision=round(mp, 4), recall=round(mr, 4),
        macro_f1=round(macro_f1, 4),
        count_mae_img=round(sum(img_abs) / ni, 4),
        count_rmse_img=round(math.sqrt(sum(img_sq) / ni), 4),
        count_bias_img=round(sum(img_signed) / ni, 4),   # signed: +ve over-counts, -ve under
        n_images=len(img_abs),
    )
    return overall, rows


# ---- I/O + prediction pass ---------------------------------------------------
def _labels_dir(img_dir: Path) -> Path:
    s = str(img_dir)
    # YOLO convention: .../images[/...] -> .../labels[/...]
    if s.endswith("/images"):
        return Path(s[: -len("/images")] + "/labels")
    return Path(s.replace("/images/", "/labels/")) if "/images/" in s else img_dir.parent / "labels"


def _read_gt(p: Path):
    out = []
    if p.exists():
        for ln in p.read_text().splitlines():
            if ln.strip():
                c, cx, cy, *_ = ln.split()
                out.append((int(float(c)), float(cx), float(cy)))
    return out


# ---- confidence-threshold tuning (pure over cached predictions) -------------
def _strip_conf(per_image_c, conf):
    """Filter cached preds to conf>=threshold and drop the conf field -> aggregate() input."""
    return [([(c, x, y) for (c, x, y, cf) in preds_c if cf >= conf], gts)
            for preds_c, gts in per_image_c]


def _parse_grid(spec):
    lo, hi, step = (float(x) for x in spec.split(":"))
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def tune_conf(per_image_c, classes, match_dist, grid, metric="f1"):
    """Pick the confidence operating point on this split. Returns (best_conf, sweep_rows).
    Pure: re-thresholds cached predictions, so the model runs only once.

    metric="f1" (default): MAXIMISE centre-matched micro-F1. This is the robust choice —
      on a grid-tiled set most tiles are empty background, so minimising counting MAE
      degenerates to "predict nothing" (conf->1, F1->0). F1 cannot collapse that way.
      Counting MAE/bias are still reported at the chosen conf.
    metric="count_mae": MINIMISE per-image counting MAE (only sensible on object-dense
      splits without many empty tiles)."""
    rows = []
    for t in grid:
        ov, _ = aggregate(_strip_conf(per_image_c, t), classes, match_dist)
        rows.append(dict(conf=t, count_mae_img=ov["count_mae_img"],
                         count_bias_img=ov["count_bias_img"], micro_f1=ov["micro_f1"],
                         macro_f1=ov["macro_f1"]))
    if metric == "count_mae":
        best = min(rows, key=lambda r: (r["count_mae_img"], -r["micro_f1"]))
    else:  # "f1"
        best = max(rows, key=lambda r: (r["micro_f1"], -r["count_mae_img"]))
    return best["conf"], rows


def _resolve(data, split, images):
    if images:
        img_dir = Path(images)
    else:
        import yaml
        y = yaml.safe_load(Path(data).read_text())
        img_dir = Path(y[split])
    names = None
    if data and Path(data).exists():
        import yaml
        names = yaml.safe_load(Path(data).read_text()).get("names")
    return img_dir, _labels_dir(img_dir), names


def _predict_per_image(model, imgs, lbl_dir, imgsz, conf_floor):
    """Run the model ONCE at a low floor; cache (cls,xc,yc,conf) + GT per image so a
    confidence sweep can re-threshold in memory without re-running inference."""
    out = []
    # max_det caps boxes/tile so caching low-conf predictions over many empty tiles stays
    # bounded in memory (tiles rarely hold >100 insects). stream=True yields results in
    # INPUT order, so we pair each with its source path `p` and read GT by p.stem — NOT by
    # r.path, which on a batched list source does not reliably map back to the label file
    # (that bug silently returned empty GT -> F1=0 everywhere).
    results = model.predict(source=[str(p) for p in imgs], imgsz=imgsz, conf=conf_floor,
                            max_det=100, stream=True, verbose=False)
    for p, r in zip(imgs, results):
        preds = [(int(c), float(x), float(y), float(cf))
                 for c, cf, (x, y, _w, _h) in zip(r.boxes.cls.tolist(),
                                                  r.boxes.conf.tolist(), r.boxes.xywhn.tolist())]
        out.append((preds, _read_gt(lbl_dir / (p.stem + ".txt"))))
    return out


# ---- classic (anchor-based) YOLOv5 backend -----------------------------------
# The SENSEI-architecture sweep is trained with the classic ultralytics/yolov5 repo, whose
# v7 checkpoints the modern `ultralytics` package cannot load. We load them via torch.hub
# (source="local") and read the same (cls, xc, yc, conf) tuples so every downstream metric is
# backend-agnostic.
def _load_classic(weights, repo):
    import torch
    m = torch.hub.load(repo or ".", "custom", path=str(weights), source="local", verbose=False)
    return m


def _predict_per_image_classic(model, imgs, lbl_dir, imgsz, conf_floor, batch=32):
    """Same cache contract as _predict_per_image, via the classic torch.hub AutoShape model.
    Runs once at conf_floor; predictions are re-thresholded later in memory."""
    model.conf = float(conf_floor)
    model.max_det = 100
    out = []
    for k in range(0, len(imgs), batch):
        chunk = imgs[k:k + batch]
        res = model([str(p) for p in chunk], size=imgsz)
        for p, df in zip(chunk, res.pandas().xywhn):   # normalised xc,yc,w,h,conf,class
            preds = [(int(row["class"]), float(row["xcenter"]), float(row["ycenter"]),
                      float(row["confidence"])) for _, row in df.iterrows()]
            out.append((preds, _read_gt(lbl_dir / (p.stem + ".txt"))))
    return out


def build_model(weights, backend, yolov5_repo=None):
    if backend == "yolov5-classic":
        return _load_classic(weights, yolov5_repo)
    from ultralytics import YOLO  # lazy
    return YOLO(weights)


def predict_cache(model, imgs, lbl_dir, imgsz, conf_floor, backend):
    if backend == "yolov5-classic":
        return _predict_per_image_classic(model, imgs, lbl_dir, imgsz, conf_floor)
    return _predict_per_image(model, imgs, lbl_dir, imgsz, conf_floor)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/pico_tiled/weights/best.pt")
    ap.add_argument("--data", default="configs/insects1201.yaml",
                    help="data yaml (for the split path + class names)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--images", default=None,
                    help="override: predict on this image dir (labels inferred as ../labels)")
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence threshold (overridden by --tune-on); always reported")
    ap.add_argument("--tune-on", default=None, choices=[None, "val", "test"],
                    help="tune the conf threshold on this split (minimise count MAE) and apply "
                         "it to --split. Use 'val' to keep the test split leakage-free.")
    ap.add_argument("--conf-grid", default="0.05:0.9:0.05",
                    help="lo:hi:step grid for the conf sweep")
    ap.add_argument("--tune-metric", default="f1", choices=["f1", "count_mae"],
                    help="operating-point objective: 'f1' (robust; default) maximises "
                         "centre-matched F1; 'count_mae' minimises counting MAE (degenerate "
                         "on background-dominated tiled sets)")
    ap.add_argument("--conf-floor", type=float, default=0.05,
                    help="predictions are run once at this floor, then re-thresholded; keep it "
                         "= the grid minimum so fewer junk boxes are cached (lower memory)")
    ap.add_argument("--match-dist", type=float, default=0.05,
                    help="centre-match radius in normalised coords (FOMO-style point matching)")
    ap.add_argument("--out", default=str(OUT / "monitoring.csv"))
    ap.add_argument("--out-species", default=str(OUT / "monitoring_per_species.csv"))
    ap.add_argument("--out-sweep", default=str(OUT / "monitoring_conf_sweep.csv"))
    ap.add_argument("--provenance", default="measured")
    ap.add_argument("--backend", default="ultralytics",
                    choices=["ultralytics", "yolov5-classic"],
                    help="'yolov5-classic' loads anchor-based v7 checkpoints (SENSEI-arch "
                         "sweep) via torch.hub; 'ultralytics' is the modern package")
    ap.add_argument("--yolov5-repo", default=None,
                    help="local clone of ultralytics/yolov5 (required for --backend yolov5-classic)")
    a = ap.parse_args(argv)
    if a.backend == "yolov5-classic" and not a.yolov5_repo:
        raise SystemExit("[monitor] --backend yolov5-classic requires --yolov5-repo <local clone>")

    img_dir, lbl_dir, names = _resolve(a.data, a.split, a.images)
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"[monitor] no images under {img_dir}")
    classes = sorted(names.keys()) if isinstance(names, dict) else list(range(9))

    model = build_model(a.weights, a.backend, a.yolov5_repo)

    # Stage 1 (optional): tune the operating point on a separate split (no test leakage).
    conf = a.conf
    if a.tune_on:
        t_dir, t_lbl, _ = _resolve(a.data, a.tune_on, None)
        t_imgs = sorted(p for p in t_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
        t_cache = predict_cache(model, t_imgs, t_lbl, a.imgsz, a.conf_floor, a.backend)
        conf, sweep = tune_conf(t_cache, classes, a.match_dist, _parse_grid(a.conf_grid),
                                metric=a.tune_metric)
        Path(a.out_sweep).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out_sweep, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep[0])); w.writeheader(); w.writerows(sweep)
        print(f"[tune] conf* = {conf} (min count-MAE on '{a.tune_on}') -> {a.out_sweep}")

    # Stage 2: evaluate the target split at the chosen threshold.
    cache = predict_cache(model, imgs, lbl_dir, a.imgsz, a.conf_floor, a.backend)
    overall, rows = aggregate(_strip_conf(cache, conf), classes, a.match_dist)
    overall.update(conf=conf, tuned_on=(a.tune_on or "fixed"), match_dist=a.match_dist,
                   imgsz=a.imgsz, provenance=a.provenance)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(overall)); w.writeheader(); w.writerow(overall)
    if isinstance(names, dict):
        name_of = lambda c: names.get(c, str(c))
    elif isinstance(names, (list, tuple)):
        name_of = lambda c: names[c] if 0 <= c < len(names) else str(c)
    else:
        name_of = lambda c: str(c)
    for row in rows:
        row["species"] = name_of(row["class_id"]); row["provenance"] = a.provenance
    fields = ["class_id", "species", "n_gt", "n_pred", "tp", "fp", "fn",
              "precision", "recall", "f1", "count_mae", "count_rmse", "r2", "provenance"]
    with open(a.out_species, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})
    print(f"[done] monitoring on '{a.split}' @ conf={conf} (tuned_on={overall['tuned_on']}): "
          f"micro-F1={overall['micro_f1']} macro-F1={overall['macro_f1']} "
          f"count-MAE/img={overall['count_mae_img']} bias={overall['count_bias_img']} "
          f"-> {a.out}, {a.out_species}")


if __name__ == "__main__":
    main()
