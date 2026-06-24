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
           "--weights", str(weights), "--include", "onnx", "--imgsz", str(imgsz), "--opset", "13"]
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


def map_eval(onnx_path: Path, data_yaml: str, imgsz: int, split: str) -> tuple[float, float]:
    from ultralytics import YOLO
    r = YOLO(str(onnx_path)).val(data=data_yaml, split=split, imgsz=imgsz, workers=2, verbose=False)
    return round(float(r.box.map50), 4), round(float(r.box.map), 4)


def counting_eval(onnx_path: Path, data_yaml: str, imgsz: int, tag: str) -> dict:
    """Run monitor_metrics (ultralytics backend accepts ONNX via AutoBackend); parse F1/recall and
    compute count-weighted MAE + net bias (sum over species) to match the paper's metrics."""
    ov = OUT / f"_int8tmp_{tag}.csv"; sp = OUT / f"_int8tmp_{tag}_per_species.csv"
    cmd = [sys.executable, "-m", "insect_gap9.monitor_metrics", "--backend", "ultralytics",
           "--weights", str(onnx_path), "--data", data_yaml, "--split", "test",
           "--tune-on", "val", "--tune-metric", "f1", "--imgsz", str(imgsz), "--match-dist", "0.05",
           "--out", str(ov), "--out-species", str(sp), "--out-sweep", str(OUT / f"_int8tmp_{tag}_sweep.csv")]
    print("[int8] counting:", " ".join(cmd)); subprocess.run(cmd, check=True)
    orow = next(csv.DictReader(ov.open()))
    srows = list(csv.DictReader(sp.open()))
    sg = sum(float(r["n_gt"]) for r in srows) or 1.0
    mae = sum(abs(float(r["n_pred"]) - float(r["n_gt"])) for r in srows) / sg
    bias = sum(float(r["n_pred"]) - float(r["n_gt"]) for r in srows) / sg
    return dict(micro_f1=float(orow["micro_f1"]), recall=float(orow["recall"]),
                count_mae_pct=round(100 * mae, 2), net_bias_pct=round(100 * bias, 2))


def _eval(onnx_path: Path, data_yaml: str, imgsz: int) -> dict:
    m50, m5095 = map_eval(onnx_path, data_yaml, imgsz, "test")
    c = counting_eval(onnx_path, data_yaml, imgsz, onnx_path.stem)
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

    fp32 = export_onnx(Path(a.weights), a.imgsz, Path(a.yolov5_repo))
    int8 = quantize_int8(fp32, Path(a.calib_dir), a.imgsz, a.n_calib)

    print("[int8] === evaluating FP32 ONNX ==="); efp = _eval(fp32, a.data, a.imgsz)
    print("[int8] === evaluating INT8 ONNX ==="); eint = _eval(int8, a.data, a.imgsz)
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
