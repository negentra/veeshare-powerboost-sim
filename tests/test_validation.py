"""Smoke test for validation module."""

import pandas as pd

from src.aggregate import build_traceability
from src.validate import run_validation


def test_run_validation_smoke():
    headline = pd.DataFrame([
        {"metric_id": f"M{i:02d}", "cluster": 0, "metric_name": "x",
         "value": 1.0, "unit": "x", "note": "x"}
        for i in range(1, 21)
    ])
    # Set ranges so all critical checks pass
    for mid, val in [
        ("M03", 100), ("M04", 100), ("M17", 100), ("M18", 100), ("M19", 50),
        ("M10", 5), ("M20", 4),
    ]:
        headline.loc[headline["metric_id"] == mid, "value"] = val
    c1 = pd.DataFrame({
        "n_nodes": [100], "repetition": [1], "throughput_mean": [5],
        "latency_p95_ms": [100],
    })
    c2 = pd.DataFrame({
        "penetration_pct": [50, 50], "regime": ["baseline", "coordinated"],
        "transformer_overload_hours": [10, 1],
        "total_energy_delivered_kwh": [100, 99], "unmet_demand_kwh": [1, 2],
    })
    c3 = pd.DataFrame({"model": ["pinn"], "mae_kw": [5]})
    trace = build_traceability()
    ok, report = run_validation(
        headline_df=headline, c1_df=c1, c2_df=c2, c3_df=c3,
        traceability_df=trace, figure_paths=[], wall_time_seconds=10.0,
        mode="smoke",
    )
    # V13 will fail (no figures) — that's fine for this unit test;
    # we just verify the structure
    assert "checks" in report
    assert len(report["checks"]) == 16


def _valid_headline():
    headline = pd.DataFrame([
        {"metric_id": f"M{i:02d}", "cluster": 0, "metric_name": "x",
         "value": 1.0, "unit": "x", "note": "x"}
        for i in range(1, 21)
    ])
    for mid, val in [
        ("M03", 100), ("M04", 100), ("M17", 100), ("M18", 100), ("M19", 50),
        ("M10", 5), ("M20", 4),
    ]:
        headline.loc[headline["metric_id"] == mid, "value"] = val
    return headline


def test_validation_unmet_demand_threshold_is_five_percent():
    c1 = pd.DataFrame({
        "n_nodes": [100, 500], "repetition": [1, 1],
        "throughput_mean": [5, 10], "latency_p95_ms": [100, 100],
    })
    c2 = pd.DataFrame({
        "penetration_pct": [50], "regime": ["coordinated"],
        "transformer_overload_hours": [0],
        "total_energy_delivered_kwh": [900], "unmet_demand_kwh": [100],
    })
    ok, report = run_validation(
        headline_df=_valid_headline(), c1_df=c1, c2_df=c2,
        c3_df=pd.DataFrame(), traceability_df=build_traceability(),
        figure_paths=[], wall_time_seconds=10.0, mode="smoke",
    )
    v09 = next(c for c in report["checks"] if c["id"] == "V09")
    assert v09["status"] == "FAIL"
    assert v09["severity"] == "advisory"


def test_validation_accepts_non_decreasing_system_throughput():
    c1 = pd.DataFrame({
        "n_nodes": [100, 500, 1000], "repetition": [1, 1, 1],
        "throughput_mean": [3.0, 15.0, 30.0],
        "latency_p95_ms": [100, 100, 100],
    })
    c2 = pd.DataFrame({
        "penetration_pct": [50], "regime": ["coordinated"],
        "transformer_overload_hours": [0],
        "total_energy_delivered_kwh": [1000], "unmet_demand_kwh": [0],
    })
    ok, report = run_validation(
        headline_df=_valid_headline(), c1_df=c1, c2_df=c2,
        c3_df=pd.DataFrame(), traceability_df=build_traceability(),
        figure_paths=[], wall_time_seconds=10.0, mode="smoke",
    )
    v05 = next(c for c in report["checks"] if c["id"] == "V05")
    assert v05["status"] == "PASS"
