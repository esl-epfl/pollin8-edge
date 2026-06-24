"""Tests for the deployability exporters + figures/tables (fallback mode)."""
import csv, importlib
from pathlib import Path
import pytest


def _read(p):
    return list(csv.DictReader(Path(p).open()))


def test_model_stats_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from insect_gap9 import model_stats
    model_stats.main(["--weights", "nope.pt", "--imgsz", "160", "--precision", "int8"])
    ms = _read(tmp_path / "results/metrics/model_stats.csv")[0]
    assert int(ms["params"]) == 314774            # yolov5p_sensei param count (backbone+neck 306,584 + anchor head 8,190)
    assert ms["fits_l2"] in ("True", "true")
    assert int(ms["macs"]) > 0
    layers = _read(tmp_path / "results/metrics/layers.csv")
    assert len(layers) == 25 and layers[-1]["type"] == "Detect"


def test_resolution_sweep_monotonic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from insect_gap9 import sweep
    sweep.main(["--sizes", "96", "160", "320"])
    rows = _read(tmp_path / "results/metrics/sweep_resolution.csv")
    assert len(rows) == 3
    maps = [float(r["map50"]) for r in rows]
    ens = [float(r["energy_mj"]) for r in rows]
    assert maps[0] < maps[-1]          # accuracy rises with resolution
    assert ens[0] < ens[-1]            # energy rises with resolution


def test_deploy_profile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from insect_gap9 import deploy_profile
    deploy_profile.main(["--latency-ms", "5.68", "--period-s", "30"])
    br = _read(tmp_path / "results/metrics/energy_breakdown.csv")
    phases = {r["phase"] for r in br}
    assert {"capture", "compute", "radio", "sleep"} <= phases
    trace = _read(tmp_path / "results/metrics/power_trace.csv")
    assert len(trace) > 5 and float(trace[-1]["t_ms"]) > 0


def test_deploy_figures_and_new_tables(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)
    # minimal CSVs the figures/tables read
    m = tmp_path / "results/metrics"; m.mkdir(parents=True)
    (m / "accuracy.csv").write_text("split,imgsz,map50,map5095,precision,recall,provenance\ntest,160,0.88,0.61,0.9,0.86,simulated\n")
    (m / "energy.csv").write_text("platform,energy_mj,active_power_mw,avg_power_mw_dutycycle,monthly_wh_upperbound,provenance\nGAP9,0.40,70,0.12,50.4,simulated\n")
    from insect_gap9 import model_stats, sweep, deploy_profile
    model_stats.main(["--weights", "nope.pt"]); sweep.main(["--sizes", "96", "160", "320"])
    deploy_profile.main([])
    (m / "sota_refs.csv").write_text("system,platform,energy_mj,latency_ms,params,task,citation_key,provenance\n"
                                     "TinyissimoYOLO,GAP9,0.15,2.12,422000,VOC,moosmann2023flexible,literature\n")
    mdf = importlib.import_module("make_deploy_figures")
    mt = importlib.import_module("make_tables")
    mdf.main(["--outdir", "results/figures"]); mt.main(["--outdir", "results/tables"])
    figs = {p.name for p in (tmp_path / "results/figures").glob("*.pdf")}
    for need in ("pareto_acc_energy.pdf", "sota_energy.pdf", "memory_footprint.pdf",
                 "power_trace.pdf", "layer_profile.pdf", "sweep_resolution_panelA.pdf"):
        assert need in figs, f"missing {need}"
    assert (tmp_path / "results/tables/model_complexity.tex").exists()
    assert "tab:layers" in (tmp_path / "results/tables/layer_table.tex").read_text()


def test_compare_runs(tmp_path, monkeypatch):
    """scratch vs distilled: accuracy differs, deployment identical."""
    monkeypatch.chdir(tmp_path)
    m = tmp_path / "results/metrics"; m.mkdir(parents=True)
    (m / "accuracy_scratch.csv").write_text("split,imgsz,map50,map5095,precision,recall,provenance\ntest,160,0.51,0.20,0.56,0.52,measured\n")
    (m / "accuracy_distilled.csv").write_text("split,imgsz,map50,map5095,precision,recall,provenance\ntest,160,0.57,0.24,0.60,0.55,measured\n")
    (m / "model_stats.csv").write_text("input_h,input_w,precision,params,model_bytes,macs,peak_activation_bytes,code_bytes,l2_budget_bytes,fits_l2,provenance\n160,160,int8,314774,314774,71875000,180000,200000,1500000,True,analytic\n")
    (m / "latency.csv").write_text("platform,cycles,latency_ms,provenance\nGAP9,2100000,5.68,simulated\n")
    (m / "energy.csv").write_text("platform,energy_mj,active_power_mw,avg_power_mw_dutycycle,monthly_wh_upperbound,provenance\nGAP9,0.40,70,0.12,50.4,simulated\n")
    from insect_gap9 import compare_runs
    compare_runs.main(["--scratch", "results/metrics/accuracy_scratch.csv",
                       "--distilled", "results/metrics/accuracy_distilled.csv"])
    import csv as _csv
    rows = list(_csv.DictReader((m / "comparison.csv").open()))
    by = {r["metric"]: r for r in rows}
    assert by["mAP@0.5"]["scratch"] == "0.51" and by["mAP@0.5"]["distilled"] == "0.57"
    assert float(by["mAP@0.5"]["delta"]) > 0                       # distillation improves accuracy
    assert by["Params"]["scratch"] == by["Params"]["distilled"]    # deployment identical
    assert "tab:distill" in (tmp_path / "results/tables/comparison.tex").read_text()
