# Copyright 2026 Negentra Yazılım ve Oyun Teknolojileri A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""VeeShare PowerBoost Audit Simulation — single entry point.

Usage:
    python run.py --mode {smoke|audit|full}

Modes per spec §1.1:
    smoke  — pipeline sanity test, < 60 sec
    audit  — primary audit simulation run, target ≤ 25 min
    full   — academic-grade extended run, target ≤ 60 min
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

# --- Windows-safe stdout: force UTF-8 if the console can't handle it.
# (Greek letters / math symbols appear in our log messages.)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    # Older Python or non-standard streams; safest fallback is to do nothing.
    pass

# --- Windows console QuickEdit / Pause protection ---------------------------
# In Windows cmd.exe and PowerShell, clicking the window or selecting text
# puts the console into "QuickEdit" / "Mark" mode. While in this mode, any
# process writing to stdout BLOCKS until the user hits a key. This makes
# Python appear frozen for hours during long runs (e.g., during PINN
# training) even though it is doing no real work.
#
# Two-layer protection:
#   (a) Disable QuickEdit on the current console at startup.
#   (b) Force unbuffered stdout so even if QuickEdit somehow re-engages, the
#       impact is one-line, not many-megabytes-of-buffer.
def _disable_windows_quickedit():
    """Best-effort: turn off QuickEdit on Windows so the console can't pause
    the process. Silent no-op on non-Windows."""
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        STD_INPUT_HANDLE = -10
        ENABLE_EXTENDED_FLAGS = 0x0080
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_INSERT_MODE = 0x0020
        h_in = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        if h_in is None or h_in == -1:
            return
        mode = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(h_in, ctypes.byref(mode)):
            return
        # Drop QuickEdit + Insert; keep extended flags so the change sticks.
        new_mode = (mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE & ~ENABLE_INSERT_MODE
        kernel32.SetConsoleMode(h_in, new_mode)
    except Exception:
        pass


_disable_windows_quickedit()

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.aggregate import build_headline, build_traceability
from src.figures import (
    make_fig1_throughput, make_fig2_latency, make_fig3_loadbal, make_fig4_pinn,
)
from src.report import build_report_html
from src.sim_counterfactual import run_counterfactual
from src.sim_loadbalance import run_cluster2
from src.sim_pinn import run_cluster3
from src.sim_protocol import run_cluster1
from src.summary import build_summary_md, write_summary_md
from src.utils.manifest import build_manifest, write_manifest
from src.utils.seed import set_global_seeds
from src.validate import run_validation


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------
def load_configs(mode: str) -> tuple[int, dict, Path, Path, Path, Path]:
    seed_path = ROOT / "config" / "seed.yaml"
    priors_path = ROOT / "config" / "priors.yaml"
    scen_path = ROOT / "config" / "scenarios.yaml"
    ctx_path = ROOT / "config" / "context.yaml"

    with open(seed_path, encoding="utf-8") as f:
        seed_root = int(yaml.safe_load(f)["seed_root"])
    with open(scen_path, encoding="utf-8") as f:
        scen = yaml.safe_load(f)

    # Apply mode overrides
    overrides = scen.get("mode_overrides", {}).get(mode, {})
    cfg_c1 = copy.deepcopy(scen["cluster1_protocol"])
    cfg_c2 = copy.deepcopy(scen["cluster2_loadbalance"])
    cfg_c3 = copy.deepcopy(scen["cluster3_pinn"])
    cfg = {
        "cluster1": cfg_c1,
        "cluster2": cfg_c2,
        "cluster3": cfg_c3,
        "p5_stretch": copy.deepcopy(scen.get("p5_stretch", {})),
    }
    if "cluster1" in overrides:
        cfg["cluster1"].update(overrides["cluster1"])
    if "cluster2" in overrides:
        cfg["cluster2"].update(overrides["cluster2"])
    if "cluster3" in overrides:
        cfg["cluster3"].update(overrides["cluster3"])
    cfg["p5_enabled"] = overrides.get("p5_enabled", False)

    return seed_root, cfg, seed_path, priors_path, scen_path, ctx_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="VeeShare PowerBoost simulation")
    parser.add_argument(
        "--mode", choices=["smoke", "audit", "full"], default="audit",
        help="Run mode (default: audit)",
    )
    parser.add_argument(
        "--out-dir", default="outputs",
        help="Output directory (default: outputs). Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the root seed from config/seed.yaml for this run "
             "(e.g. --seed 100). When omitted, seed_root from config/seed.yaml is used. "
             "Setting this is equivalent to editing seed_root in config/seed.yaml but "
             "leaves the file untouched; the effective seed is recorded in run_manifest.json.",
    )
    args = parser.parse_args()
    mode = args.mode

    out_arg = Path(args.out_dir)
    out_dir = out_arg if out_arg.is_absolute() else ROOT / out_arg
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    logs_dir = out_dir / "logs"
    for d in (tables_dir, figures_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Logging — explicit UTF-8 to avoid Windows locale crashes.
    # Custom handler that flushes after EVERY record so the log file is
    # always current even if the terminal hangs (Windows QuickEdit) or the
    # process is killed mid-run.
    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    log_path = logs_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[FlushingFileHandler(log_path, mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("run")
    log.info("=== VeeShare PowerBoost Audit Simulation — mode=%s ===", mode)

    t0 = time.time()
    seed_root, cfg, seed_path, priors_path, scen_path, ctx_path = load_configs(mode)
    if args.seed is not None:
        seed_root = args.seed
    set_global_seeds(seed_root, also_torch=True)
    log.info("Seed root = %d%s", seed_root,
             " (overridden via --seed; config/seed.yaml left unchanged)"
             if args.seed is not None else "")

    # ---- Cluster 1 ----
    log.info("Running Cluster 1 (protocol + DSO + transparency)...")
    t = time.time()
    df_c1 = run_cluster1(seed_root, cfg["cluster1"])
    log.info("Cluster 1 done in %.1fs, %d rows", time.time() - t, len(df_c1))
    df_c1.to_csv(tables_dir / "cluster1_protocol_scaling.csv", index=False, encoding="utf-8")

    # ---- Cluster 2 ----
    log.info("Running Cluster 2 (load balancing)...")
    t = time.time()
    df_c2, profiles = run_cluster2(seed_root, cfg["cluster2"], priors_path)
    log.info("Cluster 2 done in %.1fs, %d rows", time.time() - t, len(df_c2))
    df_c2.to_csv(tables_dir / "cluster2_load_balancing.csv", index=False, encoding="utf-8")

    # ---- Cluster 3 ----
    log.info("Running Cluster 3 (PINN benchmark)...")
    t = time.time()
    extended_seed_count = int(
        cfg["cluster3"].get(
            "pinn_extended_seed_count",
            3 if mode == "full" else 1,
        )
    )
    df_c3, c3_extras = run_cluster3(
        seed_root, cfg["cluster3"], cfg["cluster2"], priors_path,
        extended_seed_count=extended_seed_count,
    )
    log.info("Cluster 3 done in %.1fs, %d rows", time.time() - t, len(df_c3))
    df_c3.to_csv(tables_dir / "cluster3_pinn_benchmark.csv", index=False, encoding="utf-8")

    # ---- P5 stretch (counterfactual TR vs liberalised) — defensive ----
    # Runs only if explicitly enabled. NEVER blocks the audit deliverables:
    # any failure here is caught and logged, the audit still completes.
    if cfg.get("p5_enabled"):
        log.info("Running P5 counterfactual (TR vs liberalised regulatory regimes)...")
        t = time.time()
        try:
            df_p5 = run_counterfactual(
                seed_root, cfg["cluster2"], cfg["p5_stretch"], priors_path,
            )
            df_p5.to_csv(
                tables_dir / "counterfactual_TR_vs_liberalised.csv",
                index=False, encoding="utf-8",
            )
            log.info("P5 counterfactual done in %.1fs, %d rows",
                     time.time() - t, len(df_p5))
        except Exception as exc:
            log.error("P5 counterfactual FAILED (audit deliverables unaffected): %s", exc)
    else:
        log.info("P5 counterfactual skipped (p5_enabled=False)")

    # ---- Headline + traceability (CRITICAL — must always succeed) ----
    log.info("Building headline_numbers.csv and audit_traceability.csv...")
    headline_n = 1000 if 1000 in df_c1["n_nodes"].unique() else int(df_c1["n_nodes"].iloc[0])
    headline_pen = cfg["cluster2"].get("headline_penetration", 50)
    if headline_pen not in df_c2["penetration_pct"].unique():
        headline_pen = int(df_c2["penetration_pct"].iloc[0])
    test_n = len(c3_extras.get("y_test", []))
    headline_df = build_headline(df_c1, df_c2, df_c3,
                                 headline_n=headline_n, headline_pen=headline_pen,
                                 test_sample_count=test_n)
    headline_df.to_csv(tables_dir / "headline_numbers.csv", index=False, encoding="utf-8")
    trace_df = build_traceability()
    trace_df.to_csv(tables_dir / "audit_traceability.csv", index=False, encoding="utf-8")

    # ---- Figures (CRITICAL) ----
    log.info("Generating figures...")
    fig1 = make_fig1_throughput(df_c1, figures_dir)
    fig2 = make_fig2_latency(df_c1, figures_dir)
    fig3 = make_fig3_loadbal(df_c2, profiles, figures_dir, headline_pen=headline_pen)
    fig4 = make_fig4_pinn(df_c3, c3_extras, figures_dir)
    figure_paths = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4}

    # ---- SUMMARY.md (defensive — failure here must not kill the audit deliverable) ----
    log.info("Writing SUMMARY.md...")
    summary_text = ""
    try:
        summary_text = build_summary_md(
            headline_df=headline_df, c1_df=df_c1, c2_df=df_c2, c3_df=df_c3, mode=mode,
        )
        write_summary_md(summary_text, out_dir / "SUMMARY.md")
    except Exception as exc:
        log.error("SUMMARY.md generation/write FAILED: %s", exc)
        log.error("Headlines and traceability are safe. Proceeding to manifest + REPORT.")
        summary_text = (
            f"# SUMMARY.md generation failed\n\n"
            f"Error: {type(exc).__name__}: {exc}\n\n"
            "Headline numbers and traceability CSVs are valid; "
            "re-run summary generation manually if needed.\n"
        )
        # Best-effort write of the stub
        try:
            (out_dir / "SUMMARY.md").write_text(summary_text, encoding="utf-8")
        except Exception:
            pass

    # ---- Validation ----
    log.info("Running validation gates...")
    wall_time_seconds = time.time() - t0
    all_critical_passed, val_report = run_validation(
        headline_df=headline_df, c1_df=df_c1, c2_df=df_c2, c3_df=df_c3,
        traceability_df=trace_df, figure_paths=list(figure_paths.values()),
        wall_time_seconds=wall_time_seconds, mode=mode,
    )
    try:
        with open(logs_dir / "validation_report.json", "w", encoding="utf-8") as f:
            json.dump(val_report, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        log.error("validation_report.json write FAILED: %s", exc)

    if not all_critical_passed:
        log.error("Critical validation checks FAILED:")
        for c in val_report["checks"]:
            if c["severity"] == "critical" and c["status"] == "FAIL":
                log.error("  - %s: %s | observed=%s", c["id"], c["description"], c["observed"])

    # ---- Manifest (defensive) ----
    config_paths = {
        "seed.yaml": seed_path, "priors.yaml": priors_path,
        "scenarios.yaml": scen_path, "context.yaml": ctx_path,
    }
    output_paths = {
        "headline_numbers.csv": tables_dir / "headline_numbers.csv",
        "audit_traceability.csv": tables_dir / "audit_traceability.csv",
        "cluster1_protocol_scaling.csv": tables_dir / "cluster1_protocol_scaling.csv",
        "cluster2_load_balancing.csv": tables_dir / "cluster2_load_balancing.csv",
        "cluster3_pinn_benchmark.csv": tables_dir / "cluster3_pinn_benchmark.csv",
        "fig1_protocol_throughput.png": fig1,
        "fig2_protocol_and_dso_latency.png": fig2,
        "fig3_load_baseline_vs_coordinated.png": fig3,
        "fig4_pinn_vs_baselines.png": fig4,
        "SUMMARY.md": out_dir / "SUMMARY.md",
    }
    try:
        manifest = build_manifest(
            run_mode=mode, seed_root=seed_root,
            config_paths=config_paths, output_paths=output_paths,
            wall_time_seconds=wall_time_seconds,
            validation_passed=all_critical_passed,
        )
        write_manifest(manifest, logs_dir / "run_manifest.json")
    except Exception as exc:
        log.error("Manifest write FAILED: %s", exc)
        manifest = {"run_id": "unknown", "timestamp_utc": "unknown"}

    # ---- REPORT.html (defensive) ----
    log.info("Compiling REPORT.html...")
    try:
        badge = "pass" if (
            all_critical_passed and val_report["advisory_failed_count"] == 0
        ) else ("warn" if all_critical_passed else "fail")
        build_report_html(
            out_path=out_dir / "REPORT.html",
            run_id=manifest["run_id"], mode=mode,
            timestamp=manifest["timestamp_utc"],
            seed=seed_root,
            wall_time=f"{wall_time_seconds:.1f} s",
            headline_df=headline_df, traceability_df=trace_df,
            figure_paths=figure_paths, summary_text=summary_text,
            validation_status=badge,
        )
    except Exception as exc:
        log.error("REPORT.html generation FAILED: %s", exc)
        log.error("Audit deliverables (CSVs + figures + manifest) are still safe.")

    # ---- Final delivery message ----
    print("\n" + "=" * 67)
    print("VeeShare PowerBoost Audit Simulation — Run Complete")
    print("=" * 67)
    print(f"Run ID:        {manifest['run_id']}")
    print(f"Mode:          {mode}")
    print(f"Seed (root):   {seed_root}")
    print(f"Wall time:     {wall_time_seconds:.1f} s")
    print(f"Validation:    {'PASS' if all_critical_passed else 'FAIL'}"
          f" (advisory warnings: {val_report['advisory_failed_count']})")
    print()
    print("Outputs:")
    def _display_path(p: Path) -> Path:
        try:
            return p.relative_to(ROOT)
        except ValueError:
            return p

    for label, p in output_paths.items():
        size = p.stat().st_size if p.exists() else 0
        print(f"  {_display_path(p)}  ({size:,} bytes)")
    print(f"  {_display_path(out_dir / 'REPORT.html')}")
    print(f"  {_display_path(logs_dir / 'run_manifest.json')}")
    print(f"  {_display_path(logs_dir / 'validation_report.json')}")
    print(f"  {_display_path(logs_dir / 'run.log')}")
    print()
    val_head = headline_df.set_index("metric_id")["value"]
    print("Headline numbers (P1):")
    for mid in [f"M{i:02d}" for i in range(1, 21)]:
        v = val_head.get(mid, float("nan"))
        unit = headline_df.set_index("metric_id").loc[mid, "unit"]
        name = headline_df.set_index("metric_id").loc[mid, "metric_name"]
        if v != v:
            v_str = "n/a"
        elif abs(v) >= 1000:
            v_str = f"{v:,.1f}"
        else:
            v_str = f"{v:.3f}"
        print(f"  {mid}  {name:48s}  {v_str:>12s}  {unit}")
    print()
    print("Next step: hand outputs to the report writer.")
    print("=" * 67)

    return 0 if all_critical_passed else 1


if __name__ == "__main__":
    sys.exit(main())
