"""Validation gates — 15 sanity checks per spec §4.1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


CRITICAL_IDS = {
    "V01", "V02", "V03", "V04", "V06", "V07", "V08", "V11", "V13", "V14", "V15",
}
ADVISORY_IDS = {"V05", "V09", "V10", "V12"}


def _check(id_: str, description: str, condition: bool, observed=None,
           severity: str = None) -> Dict:
    if severity is None:
        severity = "critical" if id_ in CRITICAL_IDS else "advisory"
    return {
        "id": id_,
        "description": description,
        "status": "PASS" if condition else "FAIL",
        "observed": observed,
        "severity": severity,
    }


def run_validation(
    *,
    headline_df: pd.DataFrame,
    c1_df: pd.DataFrame,
    c2_df: pd.DataFrame,
    c3_df: pd.DataFrame,
    traceability_df: pd.DataFrame,
    figure_paths: List[Path],
    wall_time_seconds: float,
    mode: str,
) -> Tuple[bool, Dict]:
    """Run all validation checks. Returns (all_critical_passed, report_dict)."""
    checks: List[Dict] = []

    # V01: 20 rows
    checks.append(_check("V01", "headline_numbers.csv has exactly 20 rows",
                         len(headline_df) == 20, observed=len(headline_df)))
    # V02: no nulls in value
    n_null = int(headline_df["value"].isna().sum())
    checks.append(_check("V02", "headline_numbers.csv has no nulls in value",
                         n_null == 0, observed=n_null))

    val = headline_df.set_index("metric_id")["value"]

    # V03, V04
    m01 = float(val.get("M01", float("nan")))
    m02 = float(val.get("M02", float("nan")))
    checks.append(_check("V03", "M01 (throughput at 1k) > 0",
                         m01 > 0, observed=m01))
    checks.append(_check("V04", "M02 (throughput at 5k) > 0",
                         m02 > 0, observed=m02))

    # V05: system-wide throughput should not drop as node count increases
    # under the configured offered-load model.
    main = c1_df[c1_df["repetition"] != 99]
    grouped = main.groupby("n_nodes")["throughput_mean"].mean().sort_index()
    if len(grouped) >= 2:
        diffs = grouped.diff().dropna()
        non_decreasing = bool((diffs >= -1e-9).all())
    else:
        non_decreasing = True
    checks.append(_check("V05", "throughput non-decreasing with node count (advisory)",
                         non_decreasing, observed=grouped.to_dict()))

    # V06, V07, V08
    def _range(metric_id, lo, hi):
        v = float(val.get(metric_id, float("nan")))
        return lo <= v <= hi, v
    ok06, v_m03 = _range("M03", 10, 30000)
    ok06_b, v_m04 = _range("M04", 10, 30000)
    checks.append(_check("V06", "M03/M04 P2P latency in [10, 30000] ms",
                         ok06 and ok06_b, observed={"M03": v_m03, "M04": v_m04}))
    ok07_a, v_m17 = _range("M17", 10, 30000)
    ok07_b, v_m18 = _range("M18", 10, 30000)
    checks.append(_check("V07", "M17/M18 DSO latency in [10, 30000] ms",
                         ok07_a and ok07_b, observed={"M17": v_m17, "M18": v_m18}))
    ok08, v_m19 = _range("M19", 0, 100)
    checks.append(_check("V08", "M19 verifiability % in [0, 100]",
                         ok08, observed=v_m19))

    # V09: unmet demand <= 5% of total demand at 50% pen, coord (advisory)
    sub = c2_df[(c2_df["penetration_pct"] == 50) & (c2_df["regime"] == "coordinated")]
    if sub.empty:
        # fall back to whatever penetration is present
        sub = c2_df[c2_df["regime"] == "coordinated"]
    if not sub.empty:
        unmet_frac = sub["unmet_demand_kwh"].mean() / max(
            1e-6,
            sub["total_energy_delivered_kwh"].mean()
            + sub["unmet_demand_kwh"].mean(),
        )
        checks.append(_check("V09", "Cluster 2 unmet demand <= 5% (advisory)",
                             unmet_frac <= 0.05, observed=unmet_frac))
    else:
        checks.append(_check("V09", "Cluster 2 unmet demand check skipped",
                             True, observed="no coord rows"))

    # V10: M07 > 0 (advisory)
    m07 = float(val.get("M07", float("nan")))
    checks.append(_check("V10", "M07 overload reduction > 0 (advisory; 0 OK if baseline=0)",
                         m07 >= 0, observed=m07))

    # V11: PINN MAE finite and > 0
    m10 = float(val.get("M10", float("nan")))
    m20 = float(val.get("M20", float("nan")))
    checks.append(_check("V11", "M10 (PINN fixed) finite and > 0",
                         (m10 == m10) and m10 > 0, observed=m10))

    # V12: extended training MAE <= fixed-budget MAE (advisory)
    if (m20 == m20) and (m10 == m10):
        checks.append(_check("V12", "M20 (PINN extended) <= M10 (PINN fixed) (advisory)",
                             m20 <= m10 + 0.01, observed={"M10": m10, "M20": m20}))
    else:
        checks.append(_check("V12", "M20 vs M10 comparison skipped (NaN)",
                             True, observed={"M10": m10, "M20": m20}))

    # V13: figures exist in PNG and SVG
    expected_stems = [
        "fig1_protocol_throughput",
        "fig2_protocol_and_dso_latency",
        "fig3_load_baseline_vs_coordinated",
        "fig4_pinn_vs_baselines",
    ]
    fig_dir = figure_paths[0].parent if figure_paths else Path(".")
    all_present = all(
        (fig_dir / f"{s}.png").exists() and (fig_dir / f"{s}.svg").exists()
        for s in expected_stems
    )
    checks.append(_check("V13", "All 4 figures exist in PNG and SVG",
                         all_present, observed=[s for s in expected_stems]))

    # V14: traceability covers all 20 metrics
    expected_metrics = set([f"M{i:02d}" for i in range(1, 21)])
    actual_metrics = set(traceability_df["metric_id"].unique())
    checks.append(_check("V14", "audit_traceability.csv covers all 20 metrics",
                         expected_metrics.issubset(actual_metrics),
                         observed=sorted(actual_metrics)))

    # V15: wall time within budget per mode
    # Budgets are realistic estimates; V15 uses a 50% grace to avoid
    # false positives from short OS hiccups. With multi-task on,
    # audit-mode work is ~37-50 min on a typical 4-core laptop.
    budgets = {"smoke": 120.0, "audit": 3600.0, "full": 5400.0}
    budget = budgets.get(mode, 3600.0)
    checks.append(_check("V15", f"Wall time within {mode} budget",
                         wall_time_seconds <= budget * 1.5,  # 50% grace
                         observed={"wall_seconds": wall_time_seconds, "budget": budget}))

    # V16: uncertainty columns
    has_uncertainty = "ci95_low" in headline_df.columns
    checks.append(_check("V16", "headline_numbers.csv includes uncertainty columns",
                         has_uncertainty, observed=has_uncertainty,
                         severity="advisory"))


    # Roll-up
    critical_failed = [
        c for c in checks if c["severity"] == "critical" and c["status"] == "FAIL"
    ]
    advisory_failed = [
        c for c in checks if c["severity"] == "advisory" and c["status"] == "FAIL"
    ]
    all_critical_passed = len(critical_failed) == 0
    report = {
        "mode": mode,
        "wall_time_seconds": wall_time_seconds,
        "critical_failed_count": len(critical_failed),
        "advisory_failed_count": len(advisory_failed),
        "all_critical_passed": all_critical_passed,
        "checks": checks,
    }
    return all_critical_passed, report
