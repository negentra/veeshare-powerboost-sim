"""Figure generation for the four required plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.plots import ACCENT, GREY, PRIMARY, WARM, apply_style, save_both


def make_fig1_throughput(df_c1: pd.DataFrame, out_dir: Path) -> Path:
    apply_style()
    main = df_c1[df_c1["repetition"] != 99]
    grp = main.groupby("n_nodes")["throughput_mean"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        grp["n_nodes"], grp["mean"], yerr=grp["std"].fillna(0),
        fmt="o-", color=PRIMARY, ecolor=ACCENT, capsize=4, label="Mean throughput",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of nodes (log scale)")
    ax.set_ylabel("Throughput (tx/sec)")
    ax.set_title("DePIN protocol throughput vs node count")
    ax.legend()
    stem = out_dir / "fig1_protocol_throughput"
    save_both(fig, str(stem))
    plt.close(fig)
    return stem.with_suffix(".png")


def make_fig2_latency(df_c1: pd.DataFrame, out_dir: Path) -> Path:
    apply_style()
    main = df_c1[df_c1["repetition"] != 99]
    grp = main.groupby("n_nodes").agg(
        p50=("latency_p50_ms", "mean"),
        p95=("latency_p95_ms", "mean"),
        down=("dso_downstream_latency_p95_ms", "mean"),
        up=("dso_upstream_aggregation_latency_p95_ms", "mean"),
    ).reset_index()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
    axL.plot(grp["n_nodes"], grp["p50"], "o-", color=PRIMARY, label="P2P p50")
    axL.plot(grp["n_nodes"], grp["p95"], "s-", color=WARM, label="P2P p95")
    axL.set_xscale("log")
    axL.set_xlabel("Number of nodes")
    axL.set_ylabel("Latency (ms)")
    axL.set_title("P2P consensus latency")
    axL.legend()
    axR.plot(grp["n_nodes"], grp["down"], "o-", color=PRIMARY, label="DSO downstream p95")
    axR.plot(grp["n_nodes"], grp["up"], "s-", color=ACCENT, label="DSO upstream p95")
    axR.set_xscale("log")
    axR.set_xlabel("Number of nodes")
    axR.set_ylabel("Latency (ms)")
    axR.set_title("DSO interface latency")
    axR.legend()
    fig.tight_layout()
    stem = out_dir / "fig2_protocol_and_dso_latency"
    save_both(fig, str(stem))
    plt.close(fig)
    return stem.with_suffix(".png")


def make_fig3_loadbal(
    df_c2: pd.DataFrame, profiles: dict, out_dir: Path, *, headline_pen: int = 50,
) -> Path:
    apply_style()
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(9, 7))

    # Top panel: aggregate load profile for headline penetration
    key_base = (headline_pen, "baseline")
    key_coord = (headline_pen, "coordinated")
    pens_present = sorted(set([p for p, _ in profiles.keys()]))
    if key_base not in profiles or key_coord not in profiles:
        # fall back to first available penetration
        headline_pen = pens_present[0]
        key_base = (headline_pen, "baseline")
        key_coord = (headline_pen, "coordinated")
    base_p = profiles.get(key_base, np.zeros(1))
    coord_p = profiles.get(key_coord, np.zeros(1))
    hours = np.arange(len(base_p)) / 4.0  # buckets to hours
    axT.plot(hours, base_p, color=WARM, label="Baseline", alpha=0.8)
    axT.plot(hours, coord_p, color=PRIMARY, label="Coordinated", alpha=0.9)
    axT.set_xlabel("Hour of week")
    axT.set_ylabel("Aggregate network load (kW)")
    axT.set_title(f"Load profile @ {headline_pen}% EV penetration")
    axT.legend()

    # Bottom panel: overload hours by penetration, grouped by regime
    grp = df_c2.groupby(["penetration_pct", "regime"]).agg(
        overload_mean=("transformer_overload_hours", "mean"),
        overload_std=("transformer_overload_hours", "std"),
    ).reset_index()
    pens = sorted(grp["penetration_pct"].unique())
    width = 0.35
    x = np.arange(len(pens))
    base_vals = [grp[(grp["penetration_pct"] == p) & (grp["regime"] == "baseline")]["overload_mean"].sum() for p in pens]
    coord_vals = [grp[(grp["penetration_pct"] == p) & (grp["regime"] == "coordinated")]["overload_mean"].sum() for p in pens]
    base_std = [grp[(grp["penetration_pct"] == p) & (grp["regime"] == "baseline")]["overload_std"].fillna(0).sum() for p in pens]
    coord_std = [grp[(grp["penetration_pct"] == p) & (grp["regime"] == "coordinated")]["overload_std"].fillna(0).sum() for p in pens]
    axB.bar(x - width/2, base_vals, width, yerr=base_std, color=WARM, label="Baseline", capsize=4)
    axB.bar(x + width/2, coord_vals, width, yerr=coord_std, color=PRIMARY, label="Coordinated", capsize=4)
    axB.set_xticks(x)
    axB.set_xticklabels([f"{p}%" for p in pens])
    axB.set_xlabel("EV penetration")
    axB.set_ylabel("Transformer overload (h/week)")
    axB.set_title("Transformer overload reduction by regime")
    axB.legend()

    fig.tight_layout()
    stem = out_dir / "fig3_load_baseline_vs_coordinated"
    save_both(fig, str(stem))
    plt.close(fig)
    return stem.with_suffix(".png")


def make_fig4_pinn(df_c3: pd.DataFrame, extras: dict, out_dir: Path) -> Path:
    apply_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: bar of MAE by model with PINN twice
    grp = df_c3.groupby(["model", "regime"]).agg(
        mae_mean=("mae_kw", "mean"),
        mae_std=("mae_kw", "std"),
    ).reset_index()
    order = [
        ("pinn", "fixed_budget", "PINN (fixed)"),
        ("pinn", "extended", "PINN (extended)"),
        ("lstm", "fixed_budget", "LSTM"),
        ("xgboost", "fixed_budget", "XGBoost"),
    ]
    means = []
    stds = []
    labels = []
    colors = [PRIMARY, ACCENT, WARM, GREY]
    for model, regime, label in order:
        sub = grp[(grp["model"] == model) & (grp["regime"] == regime)]
        if sub.empty:
            means.append(0); stds.append(0); labels.append(label + " (n/a)")
        else:
            means.append(float(sub["mae_mean"].iloc[0]))
            stds.append(float(sub["mae_std"].fillna(0).iloc[0]))
            labels.append(label)
    x = np.arange(len(order))
    axL.bar(x, means, yerr=stds, color=colors, capsize=4)
    axL.set_xticks(x)
    axL.set_xticklabels(labels, rotation=15, ha="right")
    axL.set_ylabel("Test MAE (kW)")
    axL.set_title("Forecast error by model")

    # Right: predicted vs actual scatter for three models on test set
    y_test = extras.get("y_test", np.array([]))
    preds = extras.get("preds", {})
    if "pinn_extended" in preds:
        axR.scatter(y_test, preds["pinn_extended"], s=6, alpha=0.4,
                    color=PRIMARY, label="PINN ext")
    elif "pinn_fixed" in preds:
        axR.scatter(y_test, preds["pinn_fixed"], s=6, alpha=0.4,
                    color=PRIMARY, label="PINN")
    if "lstm" in preds:
        axR.scatter(y_test, preds["lstm"], s=6, alpha=0.4,
                    color=WARM, label="LSTM")
    if "xgboost" in preds:
        axR.scatter(y_test, preds["xgboost"], s=6, alpha=0.4,
                    color=GREY, label="XGBoost")
    if len(y_test):
        lim = max(y_test.max(), 1) * 1.05
        axR.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        axR.set_xlim(0, lim); axR.set_ylim(0, lim)
    axR.set_xlabel("Actual load (kW)")
    axR.set_ylabel("Predicted load (kW)")
    axR.set_title("Predicted vs actual (test week)")
    axR.legend(markerscale=2)

    fig.tight_layout()
    stem = out_dir / "fig4_pinn_vs_baselines"
    save_both(fig, str(stem))
    plt.close(fig)
    return stem.with_suffix(".png")
