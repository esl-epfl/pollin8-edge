"""Validate INT8 (ONNX-Runtime PTQ) detection/counting accuracy vs the FP32 model.

The paper reports accuracy on the trained FP32 model and energy/latency on the silicon INT8
deployment (Wiese 2025). This script quantifies what INT8 *costs in accuracy*, using ONNX-Runtime
static post-training quantisation as a runnable proxy for the on-device NE16/nntool quantiser
(same INT8 bit-width and PTQ family; calibration/rounding differ, so treat the delta as
representative, not bit-identical to silicon).

Pipeline for one run (runs/sensei_arch/<arm>_<size>_s<seed>/weights/best.pt):
  1. export FP32 ONNX via the classic yolov5 export.py
  2. ORT static-quantise (QDQ, per-channel) calibrated on val tiles -> INT8 ONNX
  3. evaluate BOTH ONNX models through the existing harness:
       - mAP@0.5 / mAP@0.5:0.95 : ultralytics YOLO(onnx).val
       - centre-matched F1, recall, count-weighted MAE, net bias : insect_gap9.monitor_metrics
  4. emit results/metrics/int8_validation.csv with fp32, int8 and delta(int8-fp32) rows.

Run on SCITAS via scripts/slurm/50_int8_validate.sbatch. Needs: ultralytics, onnx, onnxruntime.
"""
from __future__ import annotations
import argparse, csv, math, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("results/metrics")


def _cap_ort_threads():
    """SLURM cgroups limit cores, but onnxruntime sizes its CPU thread-pool to the whole NODE and
    pins each thread -> floods the log with `pthread_setaffinity_np failed` and oversubscribes the
    allocation. Cap intra-op threads to $SLURM_CPUS_PER_TASK (also speeds the CPU eval). Applies to
    every onnxruntime session in this process, incl. Ultralytics' ONNX backend."""
    import os
    n = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("OMP_NUM_THREADS") or 4)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    try:
        import onnxruntime as ort
        if getattr(ort.InferenceSession, "_thread_capped", False):
            return
        _orig = ort.InferenceSession
        def _capped(model, sess_options=None, providers=None, **kw):
            so = sess_options or ort.SessionOptions()
            if getattr(so, "intra_op_num_threads", 0) in (0, None):
                so.intra_op_num_threads = n
            return _orig(model, sess_options=so, providers=providers, **kw)
        _capped._thread_capped = True
        ort.InferenceSession = _capped
        print(f"[int8] capped onnxruntime intra-op threads to {n}")
    except Exception as e:
        print(f"[int8] thread-cap skipped ({e})")


# ---- preprocessing + calibration data reader ---------------------------------
def _preprocess(path: Path, imgsz: int) -> np.ndarray:
    """Square val tiles -> (1,3,imgsz,imgsz) float32 in [0,1], RGB, CHW (matches yolov5 export)."""
    im = Image.open(path).convert("RGB").resize((imgsz, imgsz))
    a = np.asarray(im, dtype=np.float32) / 255.0          # HWC
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])  # 1,C,H,W


def _onnx_input_name(onnx_path: Path) -> str:
    import onnx
    return onnx.load(str(onnx_path)).graph.input[0].name


def _make_reader(calib_dir: Path, input_name: str, imgsz: int, n: int):
    from onnxruntime.quantization import CalibrationDataReader
    paths = sorted([p for ext in ("*.jpg", "*.png", "*.jpeg") for p in calib_dir.rglob(ext)])[:n]
    if not paths:
        raise SystemExit(f"[int8] no calibration images under {calib_dir}")
    print(f"[int8] calibrating on {len(paths)} tiles from {calib_dir}")

    class _R(CalibrationDataReader):
        def __init__(self): self._it = iter(paths)
        def get_next(self):
            p = next(self._it, None)
            return None if p is None else {input_name: _preprocess(p, imgsz)}
    return _R()


# ---- stages ------------------------------------------------------------------
def export_onnx(weights: Path, imgsz: int, yolov5_repo: Path) -> Path:
    onnx_path = weights.with_suffix(".onnx")
    cmd = [sys.executable, str(Path(yolov5_repo) / "export.py"),
           "--weights", str(weights), "--include", "onnx", "--imgsz", str(imgsz), "--opset", "13",
           "--dynamic"]   # dynamic batch: the counting path (AutoShape) sends batches up to 32, which a
                          # frozen batch=1 ONNX rejects. val.py forces batch 1 and quantize_static
                          # calibrates at (1,3,N,N), so both still work with a dynamic graph.
    print("[int8] export:", " ".join(cmd)); subprocess.run(cmd, check=True)
    if not onnx_path.exists():
        raise SystemExit(f"[int8] expected ONNX at {onnx_path} not found")
    return onnx_path


def quantize_int8(fp32_onnx: Path, calib_dir: Path, imgsz: int, n_calib: int) -> Path:
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
    prep = fp32_onnx.with_name(fp32_onnx.stem + "_prep.onnx")
    src = fp32_onnx
    try:                                                  # shape-infer/clean for robust quant
        from onnxruntime.quantization.preprocess import quant_pre_process
        quant_pre_process(str(fp32_onnx), str(prep)); src = prep
    except Exception as e:
        print(f"[int8] quant_pre_process skipped ({e})")
    int8_onnx = fp32_onnx.with_name(fp32_onnx.stem + "_int8.onnx")
    quantize_static(str(src), str(int8_onnx), _make_reader(calib_dir, _onnx_input_name(src), imgsz, n_calib),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
    print(f"[int8] wrote {int8_onnx} ({int8_onnx.stat().st_size/1e6:.2f} MB)")
    return int8_onnx


def map_eval(onnx_path: Path, data_yaml: str, imgsz: int, split: str, yolov5_repo: str) -> tuple[float, float]:
    """mAP via the CLASSIC yolov5 val.py (loads ONNX through DetectMultiBackend). The modern
    ultralytics package mis-parses the anchor-based head (confusion-matrix IndexError), so the ONNX
    must be evaluated with the same classic repo used for the .pt in 49_sensei_eval."""
    run = onnx_path.parent.parent.name                       # <arm>_<size>_s<seed>: unique per sweep job
    out = subprocess.run(
        [sys.executable, str(Path(yolov5_repo) / "val.py"), "--weights", str(onnx_path),
         "--data", data_yaml, "--task", split, "--imgsz", str(imgsz),
         "--conf-thres", "0.001", "--iou-thres", "0.6", "--batch-size", "32",
         "--project", str(OUT.parent / "_int8val"), "--name", f"{run}_{onnx_path.stem}", "--exist-ok"],
        capture_output=True, text=True, check=True)
    m50 = m5095 = float("nan")
    for line in ((out.stdout or "") + "\n" + (out.stderr or "")).splitlines():
        p = line.split()
        if len(p) >= 7 and p[0] == "all":            # summary row: all <imgs> <inst> P R mAP50 mAP50-95
            try: m50, m5095 = float(p[-2]), float(p[-1])
            except ValueError: pass
    if math.isnan(m50) or math.isnan(m5095):                 # no parseable 'all' row -> fail loud, not silent nan
        raise SystemExit(f"[int8] could not parse val.py mAP summary for {onnx_path.name}; output format "
                         f"may have drifted or 0 instances matched.\n--- val.py stdout ---\n{out.stdout}\n"
                         f"--- val.py stderr ---\n{out.stderr}")
    return round(m50, 4), round(m5095, 4)


def counting_eval(onnx_path: Path, data_yaml: str, imgsz: int, tag: str, yolov5_repo: str) -> dict:
    """Centre-matched counting metrics via monitor_metrics with the CLASSIC backend (loads ONNX
    through DetectMultiBackend, like the .pt path); count-weighted MAE + net bias over species."""
    uid = f"{onnx_path.parent.parent.name}_{tag}"            # unique per sweep job (temp files share cwd)
    ov = OUT / f"_int8tmp_{uid}.csv"; sp = OUT / f"_int8tmp_{uid}_per_species.csv"
    argv = ["monitor_metrics", "--backend", "yolov5-classic", "--yolov5-repo", yolov5_repo,
            "--weights", str(onnx_path),
            "--data", data_yaml, "--split", "test", "--tune-on", "val", "--tune-metric", "f1",
            "--imgsz", str(imgsz), "--match-dist", "0.05", "--out", str(ov), "--out-species", str(sp),
            "--out-sweep", str(OUT / f"_int8tmp_{uid}_sweep.csv")]
    # run monitor_metrics in a subprocess (isolation) but apply the same onnxruntime thread cap first
    pre = ("import sys; from validate_int8 import _cap_ort_threads as _c; _c(); sys.argv = %r; "
           "import runpy; runpy.run_module('insect_gap9.monitor_metrics', run_name='__main__')" % argv)
    print("[int8] counting:", " ".join(argv)); subprocess.run([sys.executable, "-c", pre], check=True)
    orow = next(csv.DictReader(ov.open()))
    srows = list(csv.DictReader(sp.open()))
    sg = sum(float(r["n_gt"]) for r in srows) or 1.0
    mae = sum(abs(float(r["n_pred"]) - float(r["n_gt"])) for r in srows) / sg
    bias = sum(float(r["n_pred"]) - float(r["n_gt"]) for r in srows) / sg
    return dict(micro_f1=float(orow["micro_f1"]), recall=float(orow["recall"]),
                count_mae_pct=round(100 * mae, 2), net_bias_pct=round(100 * bias, 2))


def _eval(onnx_path: Path, data_yaml: str, imgsz: int, yolov5_repo: str) -> dict:
    m50, m5095 = map_eval(onnx_path, data_yaml, imgsz, "test", yolov5_repo)
    c = counting_eval(onnx_path, data_yaml, imgsz, onnx_path.stem, yolov5_repo)
    return dict(map50=m50, map5095=m5095, model_mb=round(onnx_path.stat().st_size / 1e6, 3), **c)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="runs/sensei_arch/<arm>_<size>_s<seed>/weights/best.pt")
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--arm", required=True); ap.add_argument("--seed", required=True)
    ap.add_argument("--data", required=True, help="tiled.yaml (train/val/test tile dirs)")
    ap.add_argument("--calib-dir", required=True, help="val tiles dir for calibration, e.g. $DATA/val_tiled/images")
    ap.add_argument("--yolov5-repo", required=True)
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--out", default=str(OUT / "int8_validation.csv"))
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    _cap_ort_threads()

    fp32 = export_onnx(Path(a.weights), a.imgsz, Path(a.yolov5_repo))
    int8 = quantize_int8(fp32, Path(a.calib_dir), a.imgsz, a.n_calib)

    print("[int8] === evaluating FP32 ONNX ==="); efp = _eval(fp32, a.data, a.imgsz, a.yolov5_repo)
    print("[int8] === evaluating INT8 ONNX ==="); eint = _eval(int8, a.data, a.imgsz, a.yolov5_repo)
    keys = ["map50", "map5095", "micro_f1", "recall", "count_mae_pct", "net_bias_pct", "model_mb"]
    delta = {k: round(eint[k] - efp[k], 4) for k in keys}

    new = not Path(a.out).exists()
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["arm", "size", "seed", "precision", *keys, "provenance"])
        prov = "onnxruntime-ptq(proxy for NE16)"
        w.writerow([a.arm, a.imgsz, a.seed, "fp32", *[efp[k] for k in keys], prov])
        w.writerow([a.arm, a.imgsz, a.seed, "int8", *[eint[k] for k in keys], prov])
        w.writerow([a.arm, a.imgsz, a.seed, "delta(int8-fp32)", *[delta[k] for k in keys], prov])
    print(f"[done] {a.out}: ΔmAP@0.5={delta['map50']:+.4f}  ΔF1={delta['micro_f1']:+.4f}  "
          f"Δcount_MAE={delta['count_mae_pct']:+.2f}pp  Δnet_bias={delta['net_bias_pct']:+.2f}pp")


if __name__ == "__main__":
    main()
