"""Real GAPflow/NNTool primitives: SQ8/NE16 post-training quantisation + error analysis.

This is the module `quantize.py` and `simulate_gap9.py` dispatch to when the GreenWaves
NNTool is importable. It wraps the exact deployment recipe used for the SENSEI GAP9
bring-up (load_graph -> adjust_order -> fusions('scaled_match_group') -> collect_statistics
-> quantize {scheme SQ8, use_ne16, hwc}) and adds what the accuracy validation needs:
float/quantised execution of the same prepared graph, a per-node QSNR accumulator, and a
quantisation-config hash so sharded eval jobs can assert they quantised identically.

Import is safe WITHOUT nntool: everything nntool-facing imports lazily, so the pure parts
(preprocessing, calibration selection, QSNR algebra) stay unit-testable on any machine.
NOTE: the PyPI package named "nntool" is an UNRELATED project; the real GreenWaves NNTool
comes from the GAP SDK (`pip install <gap_sdk>/tools/nntool`) -- see nntool_version().
"""
from __future__ import annotations
import hashlib, math, os
from pathlib import Path

import numpy as np
from PIL import Image

# Same extension set + sort as validate_int8._make_reader, so the ORT proxy and NNTool
# calibrate on the SAME first-n tiles.
CALIB_EXTS = ("*.jpg", "*.png", "*.jpeg")
# SQ8 is the 8-bit accuracy scheme; NE16 is only GAP9's on-chip accelerator EXECUTION mode for
# that same scheme (accuracy-equivalent). This SDK clone's NE16 numpy-emulation zeroes the graph
# output, so default to plain SQ8 for the accuracy measurement and make NE16 opt-in (NNTOOL_NE16=1).
SQ8_OPTIONS = {"scheme": "SQ8"}
if os.environ.get("NNTOOL_NE16", "") == "1":
    SQ8_OPTIONS.update(use_ne16=True, hwc=True)   # SENSEI deploy notebook (silicon SDK / SIF)
L2_BUDGET_BYTES = 1_500_000        # GAP9 L2 (matches model_stats.py)
QSNR_CAP_DB = 120.0                # reported when the quantisation error is exactly zero


# ---- environment -------------------------------------------------------------
def nntool_version() -> str | None:
    """Version of the REAL GreenWaves NNTool, or None if it is not installed.

    Raises ImportError (with a fix) if `import nntool` finds the unrelated PyPI package:
    silently treating the impostor as the SDK would crash deep inside the eval.
    """
    try:
        from nntool.api import NNGraph  # noqa: F401
    except ImportError:
        try:
            import nntool  # noqa: F401
        except ImportError:
            return None
        raise ImportError(
            "'import nntool' succeeds but 'nntool.api' does not: this is the unrelated "
            "PyPI 'nntool' package, not GreenWaves NNTool. Install the real one with "
            "`pip install <gap_sdk clone>/tools/nntool` (scripts/slurm/00b_nntool_env.sh).")
    try:
        import importlib.metadata as im
        return im.version("nntool")
    except Exception:
        return os.environ.get("NNTOOL_VERSION", "unknown")


# ---- graph discovery + preparation ------------------------------------------
def resolve_graph(weights: Path, explicit: Path | None = None) -> tuple[Path, str]:
    """Find the exported inference graph next to <weights> (best.pt).

    Ladder order = deployment fidelity: best-fp32.tflite (SENSEI-proven, anchor decode
    in-graph) -> best_static.onnx (static batch-1) -> best_trunc.onnx (three raw head
    convs; decode must then run in float Python, which EXCLUDES decode quantisation).
    Returns (path, fmt) with fmt in {tflite, onnx, onnx-trunc}. Export commands are in
    docs/training_kuma.md section 6b; exports run on the LOGIN node, never in-job.
    """
    weights = Path(weights)
    if explicit is not None:
        p = Path(explicit)
        fmt = ("tflite" if p.suffix == ".tflite"
               else "onnx-trunc" if "trunc" in p.stem else "onnx")
        if not p.exists():
            raise SystemExit(f"[nntool] --graph-path {p} does not exist")
        return p, fmt
    stem = weights.stem                                   # "best"
    candidates = [(weights.with_name(f"{stem}-fp32.tflite"), "tflite"),
                  (weights.with_name(f"{stem}_static.onnx"), "onnx"),
                  (weights.with_name(f"{stem}_trunc.onnx"), "onnx-trunc")]
    for p, fmt in candidates:
        if p.exists():
            return p, fmt
    raise SystemExit(
        f"[nntool] no exported graph next to {weights}. On the LOGIN node run one of:\n"
        f"  python $YOLOV5_REPO/export.py --weights {weights} --include tflite --imgsz <px>\n"
        f"  python $YOLOV5_REPO/export.py --weights <staged copy> --include onnx --opset 13 "
        f"--imgsz <px>   # then mv to {candidates[1][0]}\n"
        f"(FP32 export, no --int8: NNTool quantises from its own statistics.)")


def load_prepared_graph(graph_path: Path):
    """Load + adjust + fuse exactly as the SENSEI deploy notebook (pre-quantisation)."""
    from nntool.api import NNGraph
    model = NNGraph.load_graph(str(graph_path), load_quantization=False)
    model.name = Path(graph_path).stem.replace("-", "_")
    model.adjust_order()
    model.fusions('scaled_match_group')                   # == `fusions --scale8`
    model.adjust_order()
    return model


def _input_shape(model) -> list[int]:
    node = model.input_nodes()[0]
    dims = node.out_dims[0]
    return list(getattr(dims, "shape", dims))


def input_is_hwc(model) -> bool:
    """True when the (adjusted) graph expects channel-last input."""
    return int(_input_shape(model)[-1]) in (1, 3)


def input_size(model) -> int:
    """Spatial input size in px (square input assumed, as tiled)."""
    shape = _input_shape(model)
    spatial = [d for d in shape if int(d) not in (1, 3)]
    return int(spatial[0]) if spatial else int(max(shape))


# ---- calibration loader ------------------------------------------------------
def calib_images(calib_dir: Path, n: int) -> list[Path]:
    """First n tiles, sorted -- IDENTICAL selection to validate_int8._make_reader."""
    paths = sorted([p for ext in CALIB_EXTS for p in Path(calib_dir).rglob(ext)])[:n]
    if not paths:
        raise SystemExit(f"[nntool] no calibration images under {calib_dir}")
    return paths


def preprocess(path: Path, imgsz: int, hwc: bool) -> np.ndarray:
    """Same pixel values as validate_int8._preprocess (RGB, /255, float32); layout per
    graph; single sample without batch dim (nntool execute takes one image)."""
    im = Image.open(path).convert("RGB").resize((imgsz, imgsz))
    a = np.asarray(im, dtype=np.float32) / 255.0          # HWC
    return np.ascontiguousarray(a if hwc else a.transpose(2, 0, 1))


class CalibLoader:
    """Restartable iterator over preprocessed calibration tiles (collect_statistics
    consumes an iterable; shaped like the SENSEI notebook's MyDataLoader)."""

    def __init__(self, paths, imgsz: int, hwc: bool):
        self._paths, self._imgsz, self._hwc = list(paths), imgsz, hwc
        self._idx = 0

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= len(self._paths):
            raise StopIteration()
        p = self._paths[self._idx]
        self._idx += 1
        return preprocess(p, self._imgsz, self._hwc)


# ---- statistics + quantisation ----------------------------------------------
def collect_stats(model, loader):
    return model.collect_statistics(loader)


def quantize_sq8_ne16(model, statistics):
    model.quantize(statistics, graph_options=dict(SQ8_OPTIONS))


def save_stats(statistics, path: Path) -> bool:
    import pickle
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(statistics, f)
        return True
    except Exception as e:                                 # stats are a cache, not a result
        print(f"[nntool] stats save skipped ({e})")
        return False


def load_stats(path: Path):
    """Cached statistics or None; the caller recollects (deterministic calib list) and
    asserts quant_config_hash equality, so a stale/broken pickle can never skew results."""
    import pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def quant_config_hash(model) -> str:
    """sha256 over every node's output scale/zero-point: two processes that quantised
    identically agree on this; sharded eval asserts it before contributing rows."""
    h = hashlib.sha256()
    qset = model.quantization or {}
    for key in sorted(qset.keys(), key=str):
        qrec = qset[key]
        for q in (getattr(qrec, "out_qs", None) or []):
            if q is None:
                continue
            h.update(str(key).encode())
            h.update(np.asarray(getattr(q, "scale", 0.0), dtype=np.float64).tobytes())
            h.update(np.asarray(getattr(q, "zero_point", 0), dtype=np.float64).tobytes())
    return h.hexdigest()


# ---- execution ---------------------------------------------------------------
def _exec(model, img, **kw):
    try:
        return model.execute(img, **kw)
    except (TypeError, ValueError):                        # some versions want a list of inputs
        return model.execute([img], **kw)


def execute_float(model, img):
    return _exec(model, img)


def execute_quant(model, img):
    """Quantised execution, dequantised back to float at every step -- directly
    comparable to execute_float per step (the QSNR input) and at the graph output."""
    return _exec(model, img, quantize=True, dequantize=True)


def graph_outputs(model, steps) -> list[np.ndarray]:
    """The graph's output tensors from an execute() step list, batch dim guaranteed."""
    outs = []
    for node in model.output_nodes():
        t = np.asarray(steps[node.step_idx][0], dtype=np.float32)
        outs.append(t if t.ndim >= 3 else t[None])
    return outs


def step_names(model) -> dict[int, tuple[str, str]]:
    """step_idx -> (node_name, op_type) for QSNR reporting."""
    out = {}
    for node in model.nodes():
        idx = getattr(node, "step_idx", None)
        if idx is not None:
            out[int(idx)] = (str(node.name), str(getattr(node, "op_name", type(node).__name__)))
    return out


# ---- per-node QSNR (pure algebra, unit-tested) -------------------------------
def qsnr_update(acc: dict, fsteps, qsteps) -> dict:
    """Accumulate (sum f^2, sum (f-q)^2, n_images) per step over one image.

    fsteps/qsteps: execute() step lists (float / quantised-dequantised). Steps whose
    tensors are missing or shape-mismatched (fused-away, debug) are skipped.
    """
    for i in range(min(len(fsteps), len(qsteps))):
        try:
            f = np.asarray(fsteps[i][0], dtype=np.float64)
            q = np.asarray(qsteps[i][0], dtype=np.float64)
        except (TypeError, IndexError):
            continue
        if f.shape != q.shape or f.size == 0:
            continue
        sf2, se2, n = acc.get(i, (0.0, 0.0, 0))
        acc[i] = (sf2 + float((f * f).sum()), se2 + float(((f - q) ** 2).sum()), n + 1)
    return acc


def qsnr_db(sum_f2: float, sum_e2: float, cap: float = QSNR_CAP_DB) -> float:
    if sum_e2 <= 0.0:
        return cap                                        # bit-exact under quantisation
    if sum_f2 <= 0.0:
        return -cap
    return min(cap, 10.0 * math.log10(sum_f2 / sum_e2))


def qsnr_rows(acc: dict, model) -> list[dict]:
    """One row per accumulated step: name, op, QSNR(dB), output scale/zero-point."""
    names = step_names(model)
    qset = model.quantization or {}
    rows = []
    for i in sorted(acc):
        sf2, se2, n = acc[i]
        name, op = names.get(i, (f"step_{i}", "?"))
        scale = zp = ""
        try:
            q = qset[name].out_qs[0]
            scale = float(np.asarray(q.scale).ravel()[0])
            zp = float(np.asarray(q.zero_point).ravel()[0])
        except Exception:
            pass                                          # fused-away / unquantised step
        rows.append(dict(step_idx=i, node_name=name, op_type=op,
                         qsnr_db=round(qsnr_db(sf2, se2), 2),
                         out_scale=scale, out_zp=zp, n_images=n))
    return rows


# ---- the seams called by quantize.py / simulate_gap9.py ----------------------
def quantize_with_nntool(weights, calib, n_calib, out) -> dict:
    """Post-training SQ8/NE16 quantisation of the exported deployment graph; returns the
    real memory footprint (contract of quantize.py's analytic_footprint, plus provenance
    fields). Writes out/quant_report.txt and out/qsnr.csv (first 32 calibration tiles)."""
    import csv
    ver = nntool_version()
    if ver is None:
        raise SystemExit("[nntool] GreenWaves NNTool is not importable -- run "
                         "scripts/slurm/00b_nntool_env.sh (login node) first")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    graph_path, fmt = resolve_graph(Path(weights))
    model = load_prepared_graph(graph_path)
    hwc, imgsz = input_is_hwc(model), input_size(model)
    paths = calib_images(Path(calib), int(n_calib))
    stats = collect_stats(model, CalibLoader(paths, imgsz, hwc))
    quantize_sq8_ne16(model, stats)
    (out / "quant_report.txt").write_text(str(model.show()))

    acc: dict = {}
    for p in paths[:32]:
        img = preprocess(p, imgsz, hwc)
        qsnr_update(acc, execute_float(model, img), execute_quant(model, img))
    rows = qsnr_rows(acc, model)
    with (out / "qsnr.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["step_idx"])
        w.writeheader(); w.writerows(rows)

    try:
        act_items, param_items = (int(x) for x in model.total_memory_usage)
    except Exception:
        act_items = param_items = 0
    return dict(params=param_items,
                weights_bytes_int8=param_items,           # 1 byte/param at INT8
                peak_activation_bytes=act_items,
                fits_l2_1p5mb=bool(param_items + act_items < L2_BUDGET_BYTES),
                scheme="SQ8", ne16=True, hwc=hwc, imgsz=imgsz,
                graph=str(graph_path), graph_fmt=fmt,
                calib_n=len(paths), nntool_version=ver)


def run_gvsoc(model_dir) -> dict:
    raise NotImplementedError(
        "GVSOC is a manual multi-day bring-up (AutoTiler project + SDK container), not an "
        "automated call -- follow docs/gvsoc_deployment.md; simulate_gap9.py meanwhile "
        "falls back to the measured sweep row.")
