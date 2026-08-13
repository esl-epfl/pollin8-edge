"""Tests for the NNTool (deployed-quantiser) INT8 validation -- the pure parts.

Everything here runs WITHOUT the GreenWaves NNTool installed (and without the classic
yolov5 clone): gapflow_real imports nntool lazily, and validate_int8_nntool's geometry/
NMS/AP/shard logic is plain numpy. The one end-to-end smoke test skips unless nntool and
the exported artifacts are present.
"""
import csv, os, subprocess, sys
from pathlib import Path

import numpy as np
import pytest

from insect_gap9 import gapflow_real as GF
import validate_int8_nntool as VN


# ---- QSNR algebra ------------------------------------------------------------
def test_qsnr_math():
    assert GF.qsnr_db(100.0, 1.0) == pytest.approx(20.0)
    assert GF.qsnr_db(1.0, 0.0) == GF.QSNR_CAP_DB          # bit-exact -> capped
    assert GF.qsnr_db(0.0, 1.0) == -GF.QSNR_CAP_DB
    f = np.ones(10, np.float32)
    q = f + 0.1                                            # SNR = 10/0.1 = 100 -> 20 dB
    acc = GF.qsnr_update({}, [[f]], [[q]])
    sf2, se2, n = acc[0]
    assert n == 1 and GF.qsnr_db(sf2, se2) == pytest.approx(20.0, abs=1e-4)
    GF.qsnr_update(acc, [[f]], [[q]])                      # second image: ratio unchanged
    sf2, se2, n = acc[0]
    assert n == 2 and GF.qsnr_db(sf2, se2) == pytest.approx(20.0, abs=1e-4)
    # shape-mismatched (fused-away) steps are skipped, not crashed on
    acc2 = GF.qsnr_update({}, [[f]], [[np.ones(3)]])
    assert acc2 == {}


# ---- decode (rung-3 fallback) ------------------------------------------------
def test_decode_hand_fixture():
    """1 anchor, 1 class, 2x2 grid, zero logits: sigmoid=0.5 everywhere ->
    xy=(2*0.5-0.5+grid)*stride, wh=(2*0.5)^2*anchor -- computed by hand."""
    na, nc, s = 1, 1, 8.0
    x = np.zeros((1, na * (5 + nc), 2, 2), np.float32)     # NCHW
    out = VN.decode_heads([x], [[(10.0, 20.0)]], [s], nc=nc, na=na)
    assert out.shape == (1, 4, 6)
    exp_xy = np.array([[4, 4], [12, 4], [4, 12], [12, 12]], np.float32)  # (0.5+grid)*8
    np.testing.assert_allclose(out[0, :, :2], exp_xy, atol=1e-5)
    np.testing.assert_allclose(out[0, :, 2:4], np.tile([10, 20], (4, 1)), atol=1e-5)
    np.testing.assert_allclose(out[0, :, 4:], 0.5, atol=1e-6)


def test_decode_layout_agnostic():
    """The same head map in NCHW and NHWC layouts must decode identically."""
    rng = np.random.default_rng(0)
    na, nc = 3, 9
    nchw = rng.normal(size=(1, na * (5 + nc), 4, 4)).astype(np.float32)
    nhwc = nchw.transpose(0, 2, 3, 1)
    apx = [[(10, 13), (16, 30), (33, 23)]]
    a = VN.decode_heads([nchw], apx, [8.0], nc=nc, na=na)
    b = VN.decode_heads([nhwc], apx, [8.0], nc=nc, na=na)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_order_heads():
    p3 = np.zeros((1, 42, 40, 40)); p5 = np.zeros((1, 10, 10, 42)); p4 = np.zeros((1, 42, 20, 20))
    ordered = VN.order_heads([p5, p3, p4])
    assert [max(t.shape[1:]) for t in ordered] == [42, 42, 42]  # sanity: channel dim present
    assert [t.shape for t in ordered] == [(1, 42, 40, 40), (1, 42, 20, 20), (1, 10, 10, 42)]


# ---- pixel scaling -----------------------------------------------------------
def test_pixel_scaling_and_units():
    imgsz = 320
    rng = np.random.default_rng(1)
    norm = np.concatenate([rng.uniform(0, 1, (1, 50, 4)),      # normalised xywh (tflite)
                           rng.uniform(0, 1, (1, 50, 10))], -1).astype(np.float32)
    pix = norm.copy(); pix[..., :4] *= imgsz                   # pixel xywh (onnx)
    np.testing.assert_allclose(VN.pixel_scale(norm, "tflite", imgsz), pix, atol=1e-5)
    np.testing.assert_allclose(VN.pixel_scale(pix, "onnx", imgsz), pix, atol=1e-5)
    assert VN.looks_normalised(norm) and not VN.looks_normalised(pix)


# ---- NMS + mAP twins ---------------------------------------------------------
def _pred(rows):
    """rows of (x, y, w, h, obj, *cls_probs) -> (N, 5+nc) array."""
    return np.asarray(rows, np.float32)


def test_nms_np_semantics():
    nc9 = [0.0] * 9
    # two same-class boxes overlapping (IoU ~0.82) -> keep the higher-conf one;
    # a different-class box on top of them survives (per-class NMS via class offset).
    a = [100, 100, 40, 40, 0.9] + [0.9] + nc9[:8]
    b = [104, 100, 40, 40, 0.8] + [0.9] + nc9[:8]
    c = [100, 100, 40, 40, 0.7] + [0.0, 0.9] + nc9[:7]
    out = VN.nms_np(_pred([a, b, c]), 0.05, 0.45, multi_label=False, max_det=100)
    assert out.shape[0] == 2
    assert {int(r[5]) for r in out} == {0, 1}
    assert out[0, 4] == pytest.approx(0.81, abs=1e-5)          # conf = obj * cls
    # multi_label: one box above threshold for two classes -> two rows
    d = [50, 50, 20, 20, 0.9, 0.8, 0.7] + [0.0] * 7
    out2 = VN.nms_np(_pred([d]), 0.05, 0.45, multi_label=True, max_det=100)
    assert out2.shape[0] == 2 and {int(r[5]) for r in out2} == {0, 1}
    out3 = VN.nms_np(_pred([d]), 0.05, 0.45, multi_label=False, max_det=100)
    assert out3.shape[0] == 1 and int(out3[0, 5]) == 0
    # empty in, empty out
    assert VN.nms_np(np.zeros((0, 14), np.float32), 0.05, 0.45, False, 100).shape == (0, 6)


def test_match_predictions_iou_thresholds():
    gt = np.array([[0, 100, 100, 140, 140]], np.float32)       # cls, xyxy
    exact = np.array([[100, 100, 140, 140, 0.9, 0]], np.float32)
    assert VN.match_predictions(exact, gt).all()               # IoU=1 -> all 10 thresholds
    # IoU = 0.6 (40x40 gt vs 40x40 shifted): inter 24x40=960/(1600+1600-960)=0.4286 -> pick
    # a shift giving IoU just above 0.5: shift 8px -> inter 32*40=1280/1920=0.667
    shifted = np.array([[108, 100, 148, 140, 0.9, 0]], np.float32)
    c = VN.match_predictions(shifted, gt)[0]
    iou = 1280.0 / 1920.0
    assert c.tolist() == [t <= iou for t in VN.IOUV]
    wrong_cls = np.array([[100, 100, 140, 140, 0.9, 1]], np.float32)
    assert not VN.match_predictions(wrong_cls, gt).any()
    # two dets on one gt: only the better-IoU det is credited
    two = np.array([[100, 100, 140, 140, 0.8, 0], [108, 100, 148, 140, 0.9, 0]], np.float32)
    c2 = VN.match_predictions(two, gt)
    assert c2[0, 0] and not c2[1, 0]                           # at IoU 0.5


def test_ap_per_class_np():
    # one class, one gt, one perfect detection -> AP 1.0 at every IoU threshold
    tp = np.ones((1, 10), bool)
    ap, ucls = VN.ap_per_class_np(tp, np.array([0.9]), np.array([0.0]), np.array([0.0]))
    assert ucls.tolist() == [0.0]
    assert ap[0, 0] == pytest.approx(1.0, abs=0.02)            # 101-pt interp edge
    # all-miss -> AP 0
    ap0, _ = VN.ap_per_class_np(np.zeros((1, 10), bool), np.array([0.9]),
                                np.array([0.0]), np.array([0.0]))
    assert ap0[0, 0] == pytest.approx(0.0, abs=1e-6)


# ---- shard partition + merge gates -------------------------------------------
def test_shard_partition():
    items = list(range(103))
    parts = [VN.shard_slice(items, i, 8) for i in range(8)]
    flat = [x for p in parts for x in p]
    assert sorted(flat) == items and len(flat) == len(set(flat))
    assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1
    with pytest.raises(SystemExit):
        VN.shard_slice(items, 8, 8)


def _payload(stems, qhash="h1"):
    return dict(stems=stems, quant_hash=qhash, nntool_version="x")


def test_merge_gates():
    full = ["a", "b", "c", "d"]
    ok = [_payload(["a", "c"]), _payload(["b", "d"])]
    assert VN.check_shards(ok, full, allow_partial=False) == []
    with pytest.raises(SystemExit):                            # duplicate tile
        VN.check_shards([_payload(["a", "b"]), _payload(["b", "c"])], full, False)
    with pytest.raises(SystemExit):                            # mixed quantisation
        VN.check_shards([_payload(["a"]), _payload(["b"], qhash="h2")], full, False)
    with pytest.raises(SystemExit):                            # foreign tile
        VN.check_shards([_payload(["a", "z"])], full, False)
    with pytest.raises(SystemExit):                            # partial without the flag
        VN.check_shards([_payload(["a", "b"])], full, False)
    assert VN.check_shards([_payload(["a", "b"])], full, True) == ["c", "d"]


# ---- counting formulas + frozen conf ------------------------------------------
def test_count_pcts_matches_validate_int8():
    """Same algebra as validate_int8.counting_eval:153-155 (count-weighted, not per-image)."""
    rows = [dict(n_gt=10, n_pred=13), dict(n_gt=30, n_pred=24)]
    mae, bias = VN.count_pcts(rows)
    gt = 40.0
    assert mae == pytest.approx(round(100 * (3 + 6) / gt, 2))          # 22.5, no cancelling
    assert bias == pytest.approx(round(100 * (3 - 6) / gt, 2))         # -7.5, signed net


def test_frozen_conf_read(tmp_path):
    p = tmp_path / "sensei_base_320_s0.csv"
    p.write_text("micro_f1,recall,conf,tuned_on\n0.8171,0.8345,0.4,val\n")
    assert VN.read_frozen_conf("base", 320, "0", None, metrics_dir=tmp_path) == 0.4
    assert VN.read_frozen_conf("base", 320, "0", 0.3, metrics_dir=tmp_path) == 0.3
    with pytest.raises(SystemExit):
        VN.read_frozen_conf("base", 512, "0", None, metrics_dir=tmp_path)


# ---- gapflow_real pure parts --------------------------------------------------
def test_resolve_graph_ladder(tmp_path):
    w = tmp_path / "best.pt"
    w.touch()
    with pytest.raises(SystemExit):                            # nothing exported yet
        GF.resolve_graph(w)
    onnx = tmp_path / "best_static.onnx"
    onnx.touch()
    assert GF.resolve_graph(w) == (onnx, "onnx")
    tfl = tmp_path / "best-fp32.tflite"
    tfl.touch()
    assert GF.resolve_graph(w) == (tfl, "tflite")              # tflite outranks onnx
    trunc = tmp_path / "best_trunc.onnx"
    trunc.touch()
    assert GF.resolve_graph(w, explicit=trunc) == (trunc, "onnx-trunc")


def test_preprocess_and_loader(tmp_path):
    from PIL import Image
    p = tmp_path / "t.png"                                 # lossless -> exact pixel values
    Image.new("RGB", (320, 320), (255, 0, 0)).save(p)
    hwc = GF.preprocess(p, 320, hwc=True)
    chw = GF.preprocess(p, 320, hwc=False)
    assert hwc.shape == (320, 320, 3) and chw.shape == (3, 320, 320)
    assert hwc.dtype == np.float32 and hwc.max() <= 1.0
    assert hwc[0, 0, 0] == pytest.approx(1.0) and hwc[0, 0, 1] == pytest.approx(0.0)
    loader = GF.CalibLoader([p, p], 320, hwc=True)
    assert sum(1 for _ in loader) == 2
    assert sum(1 for _ in loader) == 2                         # restartable

    (tmp_path / "sub").mkdir()
    q = tmp_path / "sub" / "a.png"
    Image.new("RGB", (8, 8)).save(q)
    assert GF.calib_images(tmp_path, 10) == [q, p]             # sorted rglob, same as ORT reader


def test_nntool_absent_or_impostor_guard():
    """On machines without the GreenWaves SDK, nntool_version() is None (or raises the
    explicit impostor message if the unrelated PyPI package is installed)."""
    try:
        v = GF.nntool_version()
    except ImportError as e:
        assert "PyPI" in str(e)
    else:
        assert v is None or isinstance(v, str)


# ---- int8_summary provenance families -----------------------------------------
def test_summary_keeps_both_families(tmp_path):
    csv_p = tmp_path / "int8_validation.csv"
    hdr = "arm,size,seed,precision,map50,map5095,micro_f1,recall,count_mae_pct,net_bias_pct,model_mb,provenance\n"
    csv_p.write_text(
        hdr
        + "base,320,0,fp32,0.8623,0.4567,0.8171,0.8345,8.5,4.0,1.2,onnxruntime-ptq(proxy for NE16)\n"
        + "base,320,0,int8,0.8501,0.4413,0.8050,0.8210,9.1,3.1,0.33,onnxruntime-ptq(proxy for NE16)\n"
        + "base,320,0,delta(int8-fp32),-0.0122,-0.0154,-0.0121,-0.0135,0.6,-0.9,-0.87,onnxruntime-ptq(proxy for NE16)\n"
        + "base,320,0,fp32,0.8619,0.4560,0.8168,0.8340,8.6,4.1,1.4,nntool-sq8-ne16(6.0.0)\n"
        + "base,320,0,int8,0.8478,0.4390,0.8011,0.8180,9.4,2.9,0.31,nntool-sq8-ne16(6.0.0)\n"
        + "base,320,0,delta(int8-fp32),-0.0141,-0.0170,-0.0157,-0.0160,0.8,-1.2,-1.09,nntool-sq8-ne16(6.0.0)\n")
    script = Path(__file__).resolve().parent.parent / "scripts" / "int8_summary.py"
    r = subprocess.run([sys.executable, str(script), str(csv_p)], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "proxy for NE16" in r.stdout and "deployed quantiser" in r.stdout
    # the proxy INT8 mAP must NOT be overwritten by the nntool row (the old clobber bug)
    assert "0.8501" in r.stdout and "0.8478" in r.stdout
    tex = (tmp_path / "results/tables/int8_compare.tex").read_text()
    assert "(NNTool)" in tex and "net bias" in tex
    assert "0.850" in tex and "0.848" in tex


def test_make_values_int8_delta(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = tmp_path / "results/metrics"
    m.mkdir(parents=True)
    hdr = "arm,size,seed,precision,map50,map5095,micro_f1,recall,count_mae_pct,net_bias_pct,model_mb,provenance\n"
    (m / "int8_validation.csv").write_text(
        hdr
        + "base,320,0,delta(int8-fp32),-0.0122,-0.0154,-0.0121,-0.0135,0.6,-0.9,-0.87,onnxruntime-ptq(proxy for NE16)\n"
        + "base,320,0,delta(int8-fp32),-0.0141,-0.0170,-0.0157,-0.0160,0.8,-1.2,-1.09,nntool-sq8-ne16(6.0.0)\n")
    import importlib
    mv = importlib.import_module("make_values")
    assert mv._int8_delta("map50", 320) == pytest.approx(-0.0141)      # nntool preferred
    vals = dict(mv.build_values())
    assert vals["valIntEightDeltaMap"] == "-0.014"
    assert vals["valIntEightDeltaMae"] == "+0.8"
    (m / "int8_validation.csv").unlink()
    assert mv._int8_delta("map50", 320) is None                        # degrades gracefully
    assert dict(mv.build_values())["valIntEightDeltaMap"] == mv.PLACEHOLDER


# ---- end-to-end smoke (needs the SDK + exported artifacts) ---------------------
def test_end_to_end_smoke():
    pytest.importorskip("nntool.api", reason="GreenWaves NNTool not installed")
    if not os.environ.get("POLLIN8_NNTOOL_SMOKE"):
        pytest.skip("set POLLIN8_NNTOOL_SMOKE=1 with $WEIGHTS/$DATA staged to run the "
                    "50-tile prepare->run->merge smoke (see docs/training_kuma.md 6b)")
    # The cluster smoke path is exercised via 51_nntool_validate.sbatch MODE=run
    # NNTOOL_LIMIT=50; this hook exists for SDK-equipped hosts.
