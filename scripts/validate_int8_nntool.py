"""Validate INT8 accuracy with the REAL deployment quantiser: GreenWaves NNTool (SQ8/NE16).

`validate_int8.py` measures the FP32->INT8 delta with ONNX-Runtime PTQ, an explicit PROXY
for the on-device NE16/nntool quantiser. This script measures the same four headline
quantities -- mAP@0.5, centre-matched micro-F1, count-weighted MAE %, net bias % -- with
NNTool itself, i.e. the quantiser whose output actually runs on GAP9. Differences from the
proxy pipeline, chosen so the delta isolates *quantisation* error:

  * FP32 reference = NNTool FLOAT execution of the SAME adjusted/fused graph (not the
    classic-ONNX eval), so pt->tflite conversion error cannot masquerade as quantisation
    error. The float leg is cross-checked against the recorded classic FP32 row.
  * The confidence threshold is FROZEN at the FP32 deployed operating point (read from
    results/metrics/sensei_<arm>_<size>_s<seed>.csv) instead of re-tuned per precision;
    the full conf-grid sensitivity is emitted for free from the cached predictions.
  * NMS runs through the classic yolov5 repo when available (identical semantics to the
    recorded rows); mAP uses a val.py-equivalent 101-point-interp AP in numpy, applied
    identically to both precisions.

NNTool's executer is a numpy interpreter (~seconds/tile), so the locked test split is
sharded across a SLURM array. Three modes share one CLI:

  --mode prepare              load+fuse graph, float-parity gate vs best_static.onnx,
                              collect calibration statistics, quantise, per-node QSNR
                              -> <cache>/{meta.json,stats.pkl,quant_hash.txt}
                              -> results/metrics/quant_qsnr.csv (appended)
  --mode run --shard I --num-shards N [--limit K] [--skip-float]
                              evaluate shard I of the test tiles (float + quantised)
                              -> <dump>/shard_III_of_N.pkl.gz
  --mode merge [--conf X] [--allow-partial]
                              coverage/hash gates, metrics, 3 rows appended to
                              results/metrics/int8_validation.csv
                              (provenance nntool-sq8-ne16(<version>)) + the conf-grid
                              sensitivity -> results/metrics/int8_nntool_detail.csv

Cluster driver: scripts/slurm/51_nntool_validate.sbatch. Full runbook:
docs/training_kuma.md section 6b.
"""
from __future__ import annotations
import argparse, csv, gzip, json, pickle, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from insect_gap9 import monitor_metrics as MM              # noqa: E402
from insect_gap9 import gapflow_real as GF                 # noqa: E402

OUT = Path("results/metrics")
KEYS = ["map50", "map5095", "micro_f1", "recall", "count_mae_pct", "net_bias_pct", "model_mb"]
MAP_NMS = dict(conf_thres=0.001, iou_thres=0.6, multi_label=True, max_det=300)    # val.py
COUNT_NMS = dict(conf_thres=0.05, iou_thres=0.45, multi_label=False, max_det=100)  # AutoShape
IOUV = np.linspace(0.5, 0.95, 10)
NA, NC = 3, 9                                              # anchors/scale, classes
NO = 5 + NC


# ---- pure geometry / NMS / AP (numpy twins of the classic yolov5 semantics) --
def xywh2xyxy(x):
    y = np.array(x, dtype=np.float32, copy=True)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def box_iou_np(a, b):
    """IoU matrix between (m,4) and (n,4) xyxy boxes."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-16)


def _greedy_nms(boxes, scores, iou_thres):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        iou = box_iou_np(boxes[i][None], boxes[order[1:]])[0]
        order = order[1:][iou <= iou_thres]
    return keep


def nms_np(pred, conf_thres, iou_thres, multi_label, max_det, max_wh=7680, max_nms=30000):
    """classic non_max_suppression for ONE image. pred (N,5+nc) xywh-pixels ->
    (n,6) [x1,y1,x2,y2,conf,cls]."""
    x = pred[pred[:, 4] > conf_thres]
    if not x.size:
        return np.zeros((0, 6), np.float32)
    x = x.copy()
    x[:, 5:] *= x[:, 4:5]                                  # conf = obj * cls
    box = xywh2xyxy(x[:, :4])
    if multi_label:
        i, j = np.where(x[:, 5:] > conf_thres)
        x = np.concatenate((box[i], x[i, 5 + j][:, None], j[:, None].astype(np.float32)), 1)
    else:
        j = x[:, 5:].argmax(1)
        conf = x[np.arange(len(x)), 5 + j]
        x = np.concatenate((box, conf[:, None], j[:, None].astype(np.float32)), 1)
        x = x[conf > conf_thres]
    if not len(x):
        return np.zeros((0, 6), np.float32)
    x = x[x[:, 4].argsort()[::-1][:max_nms]]
    c = x[:, 5:6] * max_wh                                 # class-offset boxes (per-class NMS)
    keep = _greedy_nms(x[:, :4] + c, x[:, 4], iou_thres)[:max_det]
    return x[keep]


def make_nms(yolov5_repo):
    """Prefer the classic repo's non_max_suppression (bit-identical to the recorded rows);
    fall back to the numpy twin. Returns (fn(pred_1xNxno, **NMS)->(n,6) np, impl_name)."""
    try:
        if yolov5_repo:
            sys.path.insert(0, str(yolov5_repo))
        import torch
        from utils.general import non_max_suppression as _nms
        def nms(pred, conf_thres, iou_thres, multi_label, max_det):
            t = torch.from_numpy(np.ascontiguousarray(pred))
            out = _nms(t, conf_thres=conf_thres, iou_thres=iou_thres,
                       multi_label=multi_label, max_det=max_det)[0]
            return out.cpu().numpy().astype(np.float32)
        return nms, "yolov5-classic"
    except Exception as e:
        print(f"[nntool] classic NMS unavailable ({e}) -> numpy fallback (same semantics)")
        def nms(pred, conf_thres, iou_thres, multi_label, max_det):
            return nms_np(np.asarray(pred, np.float32)[0], conf_thres, iou_thres,
                          multi_label, max_det)
        return nms, "numpy-fallback"


def match_predictions(det, labels, iouv=IOUV):
    """val.py process_batch: det (n,6) xyxy/conf/cls, labels (m,5) cls/xyxy ->
    correct (n, len(iouv)) bool."""
    correct = np.zeros((det.shape[0], len(iouv)), bool)
    if det.shape[0] == 0 or labels.shape[0] == 0:
        return correct
    iou = box_iou_np(labels[:, 1:5], det[:, :4])
    cls_ok = labels[:, 0:1] == det[:, 5][None, :]
    for k, t in enumerate(iouv):
        gi, di = np.where((iou >= t) & cls_ok)
        if gi.size:
            m = np.stack([gi, di, iou[gi, di]], 1)
            if gi.size > 1:
                m = m[m[:, 2].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), k] = True
    return correct


_trapz = getattr(np, "trapezoid", None) or np.trapz    # np.trapz removed in numpy 2.x


def _compute_ap(recall, precision):
    """val.py compute_ap (101-point interpolation)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x))


def ap_per_class_np(tp, conf, pred_cls, target_cls, eps=1e-16):
    """val.py ap_per_class, mAP part only. tp (N,10) bool, conf/pred_cls (N,),
    target_cls (M,) -> (ap (nc,10), unique_classes)."""
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    ap = np.zeros((unique_classes.shape[0], tp.shape[1]))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        if not i.any() or nt[ci] == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (nt[ci] + eps)
        precision = tpc / (tpc + fpc)
        for j in range(tp.shape[1]):
            ap[ci, j] = _compute_ap(recall[:, j], precision[:, j])
    return ap, unique_classes


# ---- raw prediction tensor (per graph format) --------------------------------
def looks_normalised(raw) -> bool:
    """True when box coords are in [0,1]-ish units (tflite) rather than pixels (onnx)."""
    v = np.abs(np.asarray(raw)[..., :4])
    return float(np.percentile(v, 99)) <= 1.5 if v.size else True


def pixel_scale(raw, fmt, imgsz):
    """tflite decode emits NORMALISED xywh; DetectMultiBackend multiplies by the input
    size -- replicate that so both formats land in identical pixel space."""
    y = np.array(raw, dtype=np.float32, copy=True)
    if fmt == "tflite":
        y[..., :4] *= imgsz
    return y


def order_heads(outs, no_na=NA * NO):
    """Sort truncated-graph head maps P3->P5 (largest spatial first), matching the
    strides/anchors order in anchors.json. Layout-agnostic (NCHW or NHWC)."""
    def spatial(t):
        s = [int(d) for d in t.shape[1:] if int(d) != no_na]
        return max(s) if s else 0
    return sorted(outs, key=lambda t: -spatial(t))


def decode_heads(head_maps, anchors_px, strides, nc=NC, na=NA):
    """Replicate classic Detect.forward on raw head convs (rung-3 fallback):
    xy=(2*sig-0.5+grid)*stride, wh=(2*sig)^2*anchor_px -> (1, sum(na*H*W), 5+nc) pixels.
    Row order matches the yolov5 export decode, so parity is elementwise."""
    no = 5 + nc
    outs = []
    for x, apx, s in zip(head_maps, anchors_px, strides):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[None]
        if x.shape[1] == na * no:                          # NCHW (1, na*no, ny, nx)
            _, _, ny, nx = x.shape
            x = x.reshape(1, na, no, ny, nx).transpose(0, 1, 3, 4, 2)
        else:                                              # NHWC (1, ny, nx, na*no)
            _, ny, nx, _ = x.shape
            x = x.reshape(1, ny, nx, na, no).transpose(0, 3, 1, 2, 4)
        y = 1.0 / (1.0 + np.exp(-x))                       # sigmoid over ALL channels
        xv, yv = np.meshgrid(np.arange(nx, dtype=np.float32),
                             np.arange(ny, dtype=np.float32))
        grid = np.stack((xv, yv), 2).reshape(1, 1, ny, nx, 2)
        apx = np.asarray(apx, np.float32).reshape(1, na, 1, 1, 2)
        y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid) * float(s)
        y[..., 2:4] = (y[..., 2:4] * 2.0) ** 2 * apx
        outs.append(y.reshape(1, -1, no))
    return np.concatenate(outs, 1)


def raw_prediction(model, steps, fmt, imgsz, anchors=None):
    """Final (1, N, 5+nc) tensor in PIXEL xywh from an execute() step list."""
    outs = GF.graph_outputs(model, steps)
    if fmt == "onnx-trunc":
        return decode_heads(order_heads(outs), anchors["anchors_px"], anchors["strides"])
    if len(outs) != 1:
        raise SystemExit(f"[nntool] fmt={fmt} expects ONE decoded output, graph has "
                         f"{len(outs)} -- truncated export mislabelled? use --graph onnx-trunc")
    return pixel_scale(outs[0], fmt, imgsz)


# ---- GT readers --------------------------------------------------------------
def read_gt_boxes(txt: Path, imgsz: int) -> np.ndarray:
    """YOLO label file -> (m,5) [cls, x1,y1,x2,y2] in pixels (for the mAP path;
    the counting path reuses monitor_metrics._read_gt centres)."""
    rows = []
    if txt.exists():
        for ln in txt.read_text().splitlines():
            if ln.strip():
                c, cx, cy, w, h = (float(v) for v in ln.split()[:5])
                rows.append([c, cx * imgsz, cy * imgsz, w * imgsz, h * imgsz])
    if not rows:
        return np.zeros((0, 5), np.float32)
    a = np.asarray(rows, np.float32)
    return np.concatenate([a[:, :1], xywh2xyxy(a[:, 1:5])], 1)


# ---- shard partition + merge gates (pure) ------------------------------------
def shard_slice(items, shard, num_shards):
    """Strided slice: keeps shard content balanced and the union exactly the input."""
    if not 0 <= shard < num_shards:
        raise SystemExit(f"[nntool] shard {shard} outside 0..{num_shards - 1}")
    return items[shard::num_shards]


def check_shards(payloads, full_stems, allow_partial):
    """Coverage / duplicate / quantisation-identity gates before any row is written."""
    if not payloads:
        raise SystemExit("[nntool] no shard dumps found -- run --mode run first")
    stems = [s for p in payloads for s in p["stems"]]
    if len(stems) != len(set(stems)):
        raise SystemExit("[nntool] duplicate tiles across shard dumps -- stale dump dir? "
                         "clear it and re-run the array")
    hashes = {p["quant_hash"] for p in payloads}
    if len(hashes) != 1:
        raise SystemExit(f"[nntool] shards quantised differently (hashes {hashes}) -- "
                         "re-run the whole array from one prepare")
    extra = set(stems) - set(full_stems)
    if extra:
        raise SystemExit(f"[nntool] shards contain {len(extra)} tiles not in the test "
                         f"split (e.g. {sorted(extra)[:3]}) -- wrong --data?")
    missing = set(full_stems) - set(stems)
    if missing and not allow_partial:
        raise SystemExit(f"[nntool] only {len(stems)}/{len(full_stems)} test tiles covered "
                         "-- wait for all array shards, or pass --allow-partial "
                         "(smoke: results go to int8_validation_smoke.csv, NOT the paper CSV)")
    return sorted(missing)


def count_pcts(species_rows):
    """Count-weighted MAE % and signed net bias % -- the exact validate_int8 formulas."""
    gt = sum(float(r["n_gt"]) for r in species_rows) or 1.0
    mae = sum(abs(float(r["n_pred"]) - float(r["n_gt"])) for r in species_rows) / gt
    bias = sum(float(r["n_pred"]) - float(r["n_gt"]) for r in species_rows) / gt
    return round(100 * mae, 2), round(100 * bias, 2)


def read_frozen_conf(arm, size, seed, override=None, metrics_dir=OUT):
    """The FP32 deployed operating point conf* (frozen for BOTH precisions, so the delta
    is quantisation error, not an operating-point shift)."""
    if override is not None:
        return float(override)
    p = Path(metrics_dir) / f"sensei_{arm}_{size}_s{seed}.csv"
    if not p.exists():
        raise SystemExit(f"[nntool] frozen-conf source {p} missing -- run "
                         "49_sensei_eval.sbatch for this run first, or pass --conf")
    return float(next(csv.DictReader(p.open()))["conf"])


def eval_precision(map_stats, count_cache, classes, conf, match_dist):
    """All headline metrics for one precision from the merged shard payloads."""
    if map_stats:
        stats = [np.concatenate(x, 0) for x in zip(*map_stats)]
    else:
        stats = [np.zeros((0, len(IOUV)), bool), np.zeros(0), np.zeros(0), np.zeros(0)]
    if stats[0].size and stats[3].size:
        ap, _ = ap_per_class_np(*stats)
        map50, map5095 = float(ap[:, 0].mean()), float(ap.mean())
    else:
        map50 = map5095 = float("nan")
    overall, rows = MM.aggregate(MM._strip_conf(count_cache, conf), classes, match_dist)
    mae, bias = count_pcts(rows)
    return dict(map50=round(map50, 4), map5095=round(map5095, 4),
                micro_f1=overall["micro_f1"], recall=overall["recall"],
                count_mae_pct=mae, net_bias_pct=bias)


# ---- shared setup ------------------------------------------------------------
def _classes(names):
    return sorted(names.keys()) if isinstance(names, dict) else list(range(NC))


def _test_images(data, limit):
    img_dir, lbl_dir, names = MM._resolve(data, "test", None)
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in MM.IMG_EXTS)
    if not imgs:
        raise SystemExit(f"[nntool] no test images under {img_dir}")
    return (imgs[:limit] if limit else imgs), lbl_dir, names


def _load_anchors(weights: Path, fmt: str):
    if fmt != "onnx-trunc":
        return None
    p = Path(weights).with_name("anchors.json")
    if not p.exists():
        raise SystemExit(
            f"[nntool] {p} missing (needed to decode the truncated graph). On a torch host:\n"
            "  python - <<'EOF'\n"
            "  import json, torch; m = torch.load('best.pt', map_location='cpu')['model'].float()\n"
            "  d = m.model[-1]; a = (d.anchors * d.stride.view(-1,1,1)).tolist()\n"
            "  json.dump({'strides': d.stride.tolist(), 'anchors_px': a}, open('anchors.json','w'))\n"
            "  EOF")
    return json.loads(p.read_text())


def _prepare_model(a, need_quant=True):
    """Resolve -> load -> fuse (-> stats -> quantise) and hand back everything the mode
    needs. Statistics come from the cache when present; a recollect is deterministic
    (same sorted calibration list), and run-mode asserts the quantisation hash anyway."""
    ver = GF.nntool_version()
    if ver is None:
        raise SystemExit("[nntool] GreenWaves NNTool not importable -- run "
                         "scripts/slurm/00b_nntool_env.sh on the login node "
                         "(NOT `pip install nntool`, which is an unrelated package)")
    graph_path, fmt = GF.resolve_graph(Path(a.weights), Path(a.graph_path) if a.graph_path else None)
    if a.graph != "auto" and a.graph != fmt:
        raise SystemExit(f"[nntool] --graph {a.graph} but resolved {graph_path} ({fmt})")
    model = GF.load_prepared_graph(graph_path)
    hwc, imgsz = GF.input_is_hwc(model), GF.input_size(model)
    if imgsz != a.imgsz:
        raise SystemExit(f"[nntool] graph input {imgsz}px != --imgsz {a.imgsz} -- wrong export?")
    cache = Path(a.cache_dir)
    if need_quant:
        stats = GF.load_stats(cache / "stats.pkl")
        if stats is None:
            paths = GF.calib_images(Path(a.calib_dir), a.n_calib)
            print(f"[nntool] (re)collecting statistics on {len(paths)} calibration tiles")
            stats = GF.collect_stats(model, GF.CalibLoader(paths, imgsz, hwc))
            GF.save_stats(stats, cache / "stats.pkl")
        GF.quantize_sq8_ne16(model, stats)
    return model, graph_path, fmt, hwc, imgsz, ver, cache


# ---- modes -------------------------------------------------------------------
def mode_prepare(a):
    model, graph_path, fmt, hwc, imgsz, ver, cache = _prepare_model(a, need_quant=False)
    cache.mkdir(parents=True, exist_ok=True)
    report = str(model.show())
    (cache / "graph_report.txt").write_text(report)
    print(report)
    print(f"[nntool] graph={graph_path} fmt={fmt} hwc={hwc} imgsz={imgsz} nntool={ver}")

    anchors = _load_anchors(Path(a.weights), fmt)
    calib = GF.calib_images(Path(a.calib_dir), a.n_calib)

    # Float-parity gate: the prepared NNTool graph must reproduce the static-ONNX export
    # (coords compared in NORMALISED units) before its quantisation delta means anything.
    static = Path(a.weights).with_name(Path(a.weights).stem + "_static.onnx")
    if a.skip_parity:
        print("[nntool] WARNING: --skip-parity -- float graph unverified against ONNX")
    elif not static.exists() and fmt == "onnx":
        print(f"[nntool] parity reference is the loaded graph itself ({graph_path})")
    elif not static.exists():
        raise SystemExit(f"[nntool] parity reference {static} missing -- export it "
                         "(login node, see docs/training_kuma.md 6b) or --skip-parity")
    else:
        worst = _parity(model, fmt, static, calib[:3], imgsz, hwc, anchors)
        gate = "PASS" if worst <= a.parity_tol else "FAIL"
        print(f"[nntool] float parity vs {static.name}: max|delta|={worst:.2e} "
              f"(tol {a.parity_tol:g}) -> {gate}")
        if worst > a.parity_tol:
            raise SystemExit("[nntool] parity gate failed -- fall to the next export rung "
                             "(tflite -> onnx -> onnx-trunc), see docs/training_kuma.md 6b")

    print(f"[nntool] collecting statistics on {len(calib)} calibration tiles")
    stats = GF.collect_stats(model, GF.CalibLoader(calib, imgsz, hwc))
    GF.save_stats(stats, cache / "stats.pkl")
    GF.quantize_sq8_ne16(model, stats)
    qhash = GF.quant_config_hash(model)
    (cache / "quant_hash.txt").write_text(qhash)

    # Per-node QSNR over the first --qsnr-images calibration tiles.
    acc = {}
    for p in calib[:a.qsnr_images]:
        img = GF.preprocess(p, imgsz, hwc)
        GF.qsnr_update(acc, GF.execute_float(model, img), GF.execute_quant(model, img))
    rows = GF.qsnr_rows(acc, model)
    _append_qsnr(a, rows, ver)
    worst10 = sorted(rows, key=lambda r: r["qsnr_db"])[:10]
    print("[nntool] worst-10 QSNR nodes:")
    for r in worst10:
        print(f"  {r['qsnr_db']:>7.2f} dB  step {r['step_idx']:>3}  {r['op_type']:<18} {r['node_name']}")

    try:
        act_items, param_items = (int(x) for x in model.total_memory_usage)
    except Exception:
        act_items = param_items = 0
    meta = dict(graph=str(graph_path), fmt=fmt, hwc=hwc, imgsz=imgsz, quant_hash=qhash,
                nntool_version=ver, n_calib=len(calib),
                params=param_items, weights_bytes_int8=param_items,
                peak_activation_bytes=act_items,
                fits_l2_1p5mb=bool(param_items + act_items < GF.L2_BUDGET_BYTES))
    (cache / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[nntool] prepare done: hash={qhash[:12]}... params={param_items} "
          f"act={act_items} -> {cache}")


def _parity(model, fmt, static_onnx, tiles, imgsz, hwc, anchors):
    try:
        from validate_int8 import _cap_ort_threads
        _cap_ort_threads()
    except Exception:
        pass
    import onnxruntime as ort
    sess = ort.InferenceSession(str(static_onnx), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    worst = 0.0
    for p in tiles:
        nn = raw_prediction(model, GF.execute_float(model, GF.preprocess(p, imgsz, hwc)),
                            fmt, imgsz, anchors)
        onx = np.asarray(sess.run(None, {iname: GF.preprocess(p, imgsz, False)[None]})[0],
                         np.float32)
        if nn.shape != onx.shape:
            raise SystemExit(f"[nntool] parity shape mismatch {nn.shape} vs {onx.shape} "
                             f"for {static_onnx.name} -- different export settings?")
        nn, onx = nn.copy(), onx.copy()
        nn[..., :4] /= imgsz
        onx[..., :4] /= imgsz
        worst = max(worst, float(np.max(np.abs(nn - onx))))
    return worst


def mode_run(a):
    model, graph_path, fmt, hwc, imgsz, ver, cache = _prepare_model(a, need_quant=True)
    want_hash = (cache / "quant_hash.txt")
    if not want_hash.exists():
        raise SystemExit(f"[nntool] {want_hash} missing -- run --mode prepare first")
    qhash = GF.quant_config_hash(model)
    if qhash != want_hash.read_text().strip():
        raise SystemExit("[nntool] quantisation hash differs from prepare -- stats cache "
                         "stale or nntool version changed; re-run --mode prepare")
    anchors = _load_anchors(Path(a.weights), fmt)
    imgs, lbl_dir, _names = _test_images(a.data, a.limit)
    mine = shard_slice(imgs, a.shard, a.num_shards)
    nms, nms_impl = make_nms(a.yolov5_repo)
    precisions = ["int8"] if a.skip_float else ["fp32", "int8"]
    print(f"[nntool] shard {a.shard}/{a.num_shards}: {len(mine)} tiles "
          f"({'int8 only' if a.skip_float else 'float+int8'}), nms={nms_impl}")

    layout_checked = False
    stats = {prec: dict(map_stats=[], count_cache=[]) for prec in precisions}
    for k, p in enumerate(mine):
        img = GF.preprocess(p, imgsz, hwc)
        gt_pts = MM._read_gt(lbl_dir / (p.stem + ".txt"))
        gt_box = read_gt_boxes(lbl_dir / (p.stem + ".txt"), imgsz)
        execs = {}
        if "fp32" in precisions:
            execs["fp32"] = GF.execute_float(model, img)
        execs["int8"] = GF.execute_quant(model, img)
        for prec, steps in execs.items():
            if fmt != "onnx-trunc" and not layout_checked:
                rawout = GF.graph_outputs(model, steps)[0]
                if looks_normalised(rawout) != (fmt == "tflite"):
                    raise SystemExit(f"[nntool] output units disagree with fmt={fmt} "
                                     "(normalised vs pixel) -- wrong graph?")
                layout_checked = True
            pred = raw_prediction(model, steps, fmt, imgsz, anchors)
            det = nms(pred, **MAP_NMS)
            stats[prec]["map_stats"].append((match_predictions(det, gt_box),
                                             det[:, 4].copy(), det[:, 5].copy(),
                                             gt_box[:, 0].copy()))
            detc = nms(pred, **COUNT_NMS)
            preds = [(int(c), float((x1 + x2) / 2 / imgsz), float((y1 + y2) / 2 / imgsz),
                      float(cf)) for x1, y1, x2, y2, cf, c in detc]
            stats[prec]["count_cache"].append((preds, gt_pts))
        if (k + 1) % 50 == 0:
            print(f"[nntool] shard {a.shard}: {k + 1}/{len(mine)}")

    dump_dir = Path(a.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(shard=a.shard, num_shards=a.num_shards, stems=[p.stem for p in mine],
                   imgsz=imgsz, graph_fmt=fmt, graph=str(graph_path), quant_hash=qhash,
                   nntool_version=ver, nms_impl=nms_impl, limit=a.limit,
                   precisions=precisions, stats=stats)
    out = dump_dir / f"shard_{a.shard:03d}_of_{a.num_shards}.pkl.gz"
    with gzip.open(out, "wb") as f:
        pickle.dump(payload, f)
    print(f"[nntool] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def mode_merge(a):
    dump_dir = Path(a.dump_dir)
    files = sorted(dump_dir.glob("shard_*.pkl.gz"))
    payloads = []
    for fp in files:
        with gzip.open(fp, "rb") as f:
            payloads.append(pickle.load(f))
    imgs, _lbl, names = _test_images(a.data, None)
    missing = check_shards(payloads, [p.stem for p in imgs], a.allow_partial)
    if missing:
        print(f"[nntool] PARTIAL merge: {len(missing)} tiles missing (smoke mode)")
    classes = _classes(names)
    meta = json.loads((Path(a.cache_dir) / "meta.json").read_text())
    ver, fmt = meta["nntool_version"], meta["fmt"]
    conf = read_frozen_conf(a.arm, a.imgsz, a.seed, a.conf)
    print(f"[nntool] merging {len(payloads)} shards, frozen conf*={conf} "
          f"(fp32 operating point), nntool={ver}")

    merged = {}
    for prec in ("fp32", "int8"):
        ms = [t for p in payloads for t in p["stats"].get(prec, {}).get("map_stats", [])]
        cc = [t for p in payloads for t in p["stats"].get(prec, {}).get("count_cache", [])]
        if cc:
            merged[prec] = (ms, cc)

    results, prov = {}, f"nntool-sq8-ne16({ver})"
    fp32_prov = prov
    if "fp32" in merged:
        results["fp32"] = eval_precision(*merged["fp32"], classes, conf, a.match_dist)
        results["fp32"]["model_mb"] = round(Path(meta["graph"]).stat().st_size / 1e6, 3) \
            if Path(meta["graph"]).exists() else float("nan")
    else:                                                  # --skip-float emergency lever
        results["fp32"] = _reused_fp32_row(a)
        fp32_prov = "classic-onnx-fp32-ref(reused)"
        print("[nntool] float leg skipped -> fp32 row REUSED from the classic eval "
              "(delta then includes conversion error; provenance says so)")
    results["int8"] = eval_precision(*merged["int8"], classes, conf, a.match_dist)
    results["int8"]["model_mb"] = round(meta["weights_bytes_int8"] / 1e6, 3)
    delta = {k: round(results["int8"][k] - results["fp32"][k], 4) for k in KEYS}

    out_csv = Path(a.out) if not missing else Path(a.out).with_name("int8_validation_smoke.csv")
    if missing and str(out_csv) != str(a.out):
        print(f"[nntool] partial coverage -> writing {out_csv} instead of {a.out}")
    new = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arm", "size", "seed", "precision", *KEYS, "provenance"])
        w.writerow([a.arm, a.imgsz, a.seed, "fp32", *[results["fp32"][k] for k in KEYS], fp32_prov])
        w.writerow([a.arm, a.imgsz, a.seed, "int8", *[results["int8"][k] for k in KEYS], prov])
        w.writerow([a.arm, a.imgsz, a.seed, "delta(int8-fp32)", *[delta[k] for k in KEYS], prov])

    _write_detail(a, merged, classes, conf, prov)
    _crosscheck_fp32(a, results["fp32"])
    print(f"[done] {out_csv}: ΔmAP@0.5={delta['map50']:+.4f}  ΔF1={delta['micro_f1']:+.4f}  "
          f"Δcount_MAE={delta['count_mae_pct']:+.2f}pp  Δnet_bias={delta['net_bias_pct']:+.2f}pp")


def _reused_fp32_row(a):
    """--skip-float: reuse the recorded classic FP32 row (never silently relabelled)."""
    src = Path(a.out)
    if src.exists():
        for r in csv.DictReader(src.open()):
            if (r["arm"], r["size"], r["seed"], r["precision"]) == \
                    (a.arm, str(a.imgsz), str(a.seed), "fp32"):
                return {k: float(r[k]) for k in KEYS}
    raise SystemExit("[nntool] --skip-float needs an existing fp32 row in "
                     f"{a.out} for {a.arm}_{a.imgsz}_s{a.seed} (run validate_int8.py) ")


def _write_detail(a, merged, classes, conf_star, prov):
    """Operating-point sensitivity, free from the cached floor-0.05 predictions: the
    quantisation delta at EVERY conf grid point (is the delta stable, or threshold luck?)."""
    grid = MM._parse_grid(a.conf_grid)
    path = OUT / "int8_nntool_detail.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arm", "size", "seed", "precision", "conf", "frozen",
                        "micro_f1", "recall", "count_mae_pct", "net_bias_pct", "provenance"])
        for prec, (_ms, cc) in merged.items():
            for t in grid:
                ov, rows = MM.aggregate(MM._strip_conf(cc, t), classes, a.match_dist)
                mae, bias = count_pcts(rows)
                w.writerow([a.arm, a.imgsz, a.seed, prec, t, int(abs(t - conf_star) < 1e-9),
                            ov["micro_f1"], ov["recall"], mae, bias, prov])
    print(f"[nntool] conf-grid sensitivity -> {path}")


def _crosscheck_fp32(a, fp32):
    """Bound the pt->graph conversion error: NNTool-float vs the recorded classic row."""
    p = OUT / f"sensei_{a.arm}_{a.imgsz}_s{a.seed}.csv"
    if not p.exists():
        return
    ref = next(csv.DictReader(p.open()))
    d = fp32["micro_f1"] - float(ref["micro_f1"])
    flag = "OK" if abs(d) <= 0.02 else "WARN: conversion error above 0.02 -- inspect export"
    print(f"[nntool] float cross-check vs {p.name}: micro_f1 {fp32['micro_f1']:.4f} "
          f"vs classic {float(ref['micro_f1']):.4f} (Δ{d:+.4f}) [{flag}]")


def _append_qsnr(a, rows, ver):
    path = OUT / "quant_qsnr.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arm", "size", "seed", "step_idx", "node_name", "op_type",
                        "qsnr_db", "out_scale", "out_zp", "n_images", "provenance"])
        for r in rows:
            w.writerow([a.arm, a.imgsz, a.seed, r["step_idx"], r["node_name"], r["op_type"],
                        r["qsnr_db"], r["out_scale"], r["out_zp"], r["n_images"],
                        f"nntool-sq8-ne16({ver})"])
    print(f"[nntool] per-node QSNR ({len(rows)} steps) -> {path}")


# ---- CLI ---------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["prepare", "run", "merge"])
    ap.add_argument("--weights", required=True, help="runs/sensei_arch/<run>/weights/best.pt")
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--data", required=True, help="tiled.yaml (test split + names)")
    ap.add_argument("--calib-dir", required=True, help="val tiles (same set as validate_int8)")
    ap.add_argument("--yolov5-repo", default=None, help="classic clone (preferred NMS impl)")
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--graph", default="auto", choices=["auto", "tflite", "onnx", "onnx-trunc"])
    ap.add_argument("--graph-path", default=None, help="explicit exported graph (overrides ladder)")
    ap.add_argument("--cache-dir", default=None, help="default <weights_dir>/nntool_cache")
    ap.add_argument("--dump-dir", default=None, help="default <run_dir>/nntool_eval")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first K test tiles only")
    ap.add_argument("--skip-float", action="store_true",
                    help="emergency: int8 leg only; fp32 row reused from the classic eval "
                         "(provenance-flagged -- the delta then includes conversion error)")
    ap.add_argument("--conf", type=float, default=None,
                    help="override the frozen conf* (default: read from the run's eval CSV)")
    ap.add_argument("--conf-grid", default="0.05:0.9:0.05")
    ap.add_argument("--match-dist", type=float, default=0.05)
    ap.add_argument("--qsnr-images", type=int, default=32)
    ap.add_argument("--parity-tol", type=float, default=2e-3)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--out", default=str(OUT / "int8_validation.csv"))
    a = ap.parse_args(argv)

    w = Path(a.weights)
    a.cache_dir = a.cache_dir or str(w.parent / "nntool_cache")
    a.dump_dir = a.dump_dir or str(w.parent.parent / "nntool_eval")
    OUT.mkdir(parents=True, exist_ok=True)
    {"prepare": mode_prepare, "run": mode_run, "merge": mode_merge}[a.mode](a)


if __name__ == "__main__":
    main()
