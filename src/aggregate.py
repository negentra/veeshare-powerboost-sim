"""Aggregate per-cluster outputs into the 20-row headline_numbers.csv
and the 20-row audit_traceability.csv.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


HEADLINE_ORDER = [
    # (metric_id, cluster, metric_name, unit, note)
    ("M01", 1, "throughput_at_1k_nodes", "tx/sec", "mean across reps at N=1000"),
    ("M02", 1, "throughput_at_5k_nodes", "tx/sec", "mean across reps at N=5000"),
    ("M03", 1, "latency_p95_at_1k_nodes", "ms", "P2P consensus p95"),
    ("M04", 1, "latency_p95_at_5k_nodes", "ms", "P2P consensus p95"),
    ("M05", 1, "max_stable_node_count", "nodes", "max N where p95 latency < 2000 ms"),
    ("M06", 2, "peak_load_reduction_pct_at_50pen", "%", "(baseline-coord)/baseline at 50% pen"),
    ("M07", 2, "overload_hours_reduction_pct_at_50pen", "%", "same formula on overload hours"),
    ("M08", 2, "unmet_demand_increase_kwh_at_50pen", "kWh", "coord minus baseline; small acceptable"),
    ("M09", 2, "peak_to_avg_ratio_improvement_at_50pen", "ratio",
     "baseline_ratio / coord_ratio"),
    ("M10", 3, "pinn_mae_fixed_budget", "kW", "mean across seeds, fixed-budget regime"),
    ("M11", 3, "lstm_mae", "kW", "mean across seeds"),
    ("M12", 3, "xgboost_mae", "kW", "mean across seeds"),
    ("M13", 3, "pinn_inference_latency", "ms",
     "extended-regime model, mean ms/sample"),
    ("M14", 3, "pinn_physics_violation_rate", "%",
     "violations / test samples * 100"),
    ("M15", 0, "current_trl", "TRL", "hardcoded; justified in SUMMARY §5"),
    ("M16", 0, "target_trl_post_audit", "TRL", "hardcoded; justified in SUMMARY §5"),
    ("M17", 1, "dso_downstream_latency_p95_at_1k_nodes", "ms", "DSO broadcast to all nodes p95"),
    ("M18", 1, "dso_upstream_aggregation_latency_p95_at_1k_nodes", "ms",
     "nodes to DSO; p95 at 95% submission threshold"),
    ("M19", 1, "independent_verifiability_pct_at_1k_nodes", "%",
     "fraction of tx with witness_set >= 3"),
    ("M20", 3, "pinn_mae_extended_training", "kW",
     "extended-budget PINN regime, mean across seeds"),
]


TRACEABILITY = [
    # (metric_id, application_phrase, audit_scope_item, linked_gap, linked_req, linked_hurdle)
    ("M01", "scalability of the DePIN-based communication protocols", "1", "Gap-1", "Req-1", "Hurdle-1"),
    ("M02", "scalability of the DePIN-based communication protocols", "1", "Gap-1", "Req-1", "Hurdle-1"),
    ("M03", "scalability of the DePIN-based communication protocols", "1", "Gap-1", "Req-1", "Hurdle-1"),
    ("M04", "scalability of the DePIN-based communication protocols", "1", "Gap-1", "Req-1", "Hurdle-1"),
    ("M05", "scalability of the DePIN-based communication protocols", "1", "Gap-1", "Req-1", "Hurdle-1"),
    ("M06", "load balancing during peak demand", "2", "none", "Req-3", "Hurdle-2"),
    ("M07", "load balancing during peak demand + reduce grid congestion", "2", "none", "Req-3", "Hurdle-2"),
    ("M08", "load balancing during peak demand", "2", "Gap-4", "Req-3", "Hurdle-2"),
    ("M09", "load balancing during peak demand", "2", "none", "Req-3", "Hurdle-2"),
    ("M10", "digital maturity of our PINN", "3", "Gap-3", "none", "Hurdle-3"),
    ("M11", "digital maturity of our PINN", "3", "none", "none", "none"),
    ("M12", "digital maturity of our PINN", "3", "none", "none", "none"),
    ("M13", "digital maturity of our PINN", "3", "none", "Req-4", "Hurdle-3"),
    ("M14", "digital maturity of our PINN", "3", "Gap-5", "Req-4", "Hurdle-3"),
    ("M15", "innovation gaps and infrastructure requirements", "0", "none", "none", "none"),
    ("M16", "Solution Box readiness", "0", "none", "none", "none"),
    ("M17", "between charging points and the central grid management system", "1", "Gap-2", "Req-2", "none"),
    ("M18", "between charging points and the central grid management system", "1", "Gap-2", "Req-2", "none"),
    ("M19", "transparent, decentralized transactions", "(audit purpose)", "none", "Req-5", "none"),
    ("M20", "digital maturity of our PINN", "3", "none", "Req-4", "none"),
]


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def build_headline(
    df_c1: pd.DataFrame, df_c2: pd.DataFrame, df_c3: pd.DataFrame,
    *, headline_n: int = 1000, headline_pen: int = 50,
    test_sample_count: int | None = None,
) -> pd.DataFrame:
    """Build the 20-row headline_numbers.csv from cluster outputs."""

    # ---- Cluster 1 ----
    main = df_c1[df_c1["repetition"] != 99]  # exclude resilience row(s)
    def by_n(field, n):
        s = main[main["n_nodes"] == n][field]
        return _safe_mean(s)

    available_ns = sorted(main["n_nodes"].unique())
    n_for_headline = headline_n if headline_n in available_ns else max(available_ns)
    n_high = 5000 if 5000 in available_ns else max(available_ns)

    M01 = by_n("throughput_mean", n_for_headline)
    M02 = by_n("throughput_mean", n_high)
    M03 = by_n("latency_p95_ms", n_for_headline)
    M04 = by_n("latency_p95_ms", n_high)

    # M05: max N where p95 latency < 2000ms
    by_n_p95 = main.groupby("n_nodes")["latency_p95_ms"].mean().reset_index()
    stable = by_n_p95[by_n_p95["latency_p95_ms"] < 2000.0]["n_nodes"]
    M05 = int(stable.max()) if not stable.empty else int(main["n_nodes"].min())

    M17 = by_n("dso_downstream_latency_p95_ms", n_for_headline)
    M18 = by_n("dso_upstream_aggregation_latency_p95_ms", n_for_headline)
    M19 = by_n("independent_verifiability_pct", n_for_headline)

    # ---- Cluster 2 ----
    pen = headline_pen if headline_pen in df_c2["penetration_pct"].unique() else int(df_c2["penetration_pct"].iloc[0])
    sub = df_c2[df_c2["penetration_pct"] == pen]
    base = sub[sub["regime"] == "baseline"]
    coord = sub[sub["regime"] == "coordinated"]

    def _avg(d, col):
        return _safe_mean(d[col])

    baseline_peak = _avg(base, "peak_load_kw")
    coord_peak = _avg(coord, "peak_load_kw")
    baseline_overload = _avg(base, "transformer_overload_hours")
    coord_overload = _avg(coord, "transformer_overload_hours")
    baseline_unmet = _avg(base, "unmet_demand_kwh")
    coord_unmet = _avg(coord, "unmet_demand_kwh")
    baseline_p2a = _avg(base, "peak_to_avg_ratio")
    coord_p2a = _avg(coord, "peak_to_avg_ratio")

    M06 = (baseline_peak - coord_peak) / baseline_peak * 100.0 if baseline_peak else 0.0
    if baseline_overload > 0:
        M07 = (baseline_overload - coord_overload) / baseline_overload * 100.0
    else:
        # If no overload at baseline, define M07 as 0 (no improvement to claim)
        M07 = 0.0
    M08 = coord_unmet - baseline_unmet
    M09 = baseline_p2a / coord_p2a if coord_p2a else 0.0

    # ---- Cluster 3 ----
    pinn_fb = df_c3[(df_c3["model"] == "pinn") & (df_c3["regime"] == "fixed_budget")]
    pinn_ext = df_c3[(df_c3["model"] == "pinn") & (df_c3["regime"] == "extended")]
    lstm = df_c3[df_c3["model"] == "lstm"]
    xgb = df_c3[df_c3["model"] == "xgboost"]

    M10 = _avg(pinn_fb, "mae_kw")
    M11 = _avg(lstm, "mae_kw")
    M12 = _avg(xgb, "mae_kw")
    M13 = _avg(pinn_ext, "inference_latency_ms_per_sample")
    if not pinn_ext.empty:
        viol = _avg(pinn_ext, "physics_constraint_violations")
        if test_sample_count and test_sample_count > 0:
            M14 = float(viol) / float(test_sample_count) * 100.0
        else:
            # Direct unit-level calls may not have the test-set size. The main
            # pipeline passes it and therefore emits the documented rate.
            M14 = float(viol)
    else:
        M14 = float("nan")

    M15 = 3.0
    M16 = 5.0

    M20 = _avg(pinn_ext, "mae_kw")

    values = {
        "M01": M01, "M02": M02, "M03": M03, "M04": M04, "M05": float(M05),
        "M06": M06, "M07": M07, "M08": M08, "M09": M09,
        "M10": M10, "M11": M11, "M12": M12, "M13": M13, "M14": M14,
        "M15": M15, "M16": M16, "M17": M17, "M18": M18, "M19": M19, "M20": M20,
    }


    rows = []
    for mid, cluster, name, unit, note in HEADLINE_ORDER:
        rows.append({
            "metric_id": mid, "cluster": cluster, "metric_name": name,
            "value": values[mid], "unit": unit, "note": note,
        })
    return pd.DataFrame(rows)


def build_traceability() -> pd.DataFrame:
    rows = []
    for mid, phrase, scope, gap, req, hurdle in TRACEABILITY:
        rows.append({
            "metric_id": mid,
            "application_phrase": phrase,
            "audit_scope_item": scope,
            "linked_gap": gap,
            "linked_requirement": req,
            "linked_hurdle": hurdle,
        })
    return pd.DataFrame(rows)
