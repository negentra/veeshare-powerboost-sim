# Output Files

Each run writes a self-contained evidence package under the directory passed
with `--out-dir`.

Example:

```bash
python run.py --mode audit --out-dir outputs_audit
```

Creates:

```text
outputs_audit/
|-- tables/
|-- figures/
|-- logs/
|-- SUMMARY.md
`-- REPORT.html
```

## tables/headline_numbers.csv

The 20 headline metrics used by the audit narrative.

Important columns:

- `metric_id`: stable ID, for example `M06`.
- `cluster`: simulation cluster number.
- `metric_name`: machine-readable metric name.
- `value`: numeric value.
- `unit`: unit of measure.
- `note`: short interpretation note.

## tables/audit_traceability.csv

Maps each metric to the audit phrase, gap, requirement, and hurdle it supports.
This is the main traceability table.

## tables/cluster1_protocol_scaling.csv

Protocol and DSO-interface scaling results by node count and repetition.

Key columns:

- `n_nodes`
- `throughput_mean`
- `latency_p95_ms`
- `dso_downstream_latency_p95_ms`
- `dso_upstream_aggregation_latency_p95_ms`
- `independent_verifiability_pct`

## tables/cluster2_load_balancing.csv

Coordinated versus baseline charging results by EV penetration and repetition.

Key columns:

- `penetration_pct`
- `regime`
- `peak_load_kw`
- `transformer_overload_hours`
- `total_energy_delivered_kwh`
- `unmet_demand_kwh`
- `avg_charging_cost_eur_per_kwh`

## tables/cluster3_pinn_benchmark.csv

Forecasting benchmark results for PINN, LSTM, and XGBoost.

Key columns:

- `model`
- `regime`
- `seed`
- `mae_kw`
- `rmse_kw`
- `mape_pct`
- `inference_latency_ms_per_sample`
- `physics_constraint_violations`
- `converged`

## tables/counterfactual_TR_vs_liberalised.csv

Regulatory counterfactual results.

Key columns:

- `regime`
- `active_hosts`
- `total_energy_delivered_kwh`
- `unmet_demand_kwh`
- `total_welfare_proxy_eur_per_week`
- `welfare_delta_vs_TR`

## figures/

Four figures are generated as both PNG and SVG:

- `fig1_protocol_throughput`
- `fig2_protocol_and_dso_latency`
- `fig3_load_baseline_vs_coordinated`
- `fig4_pinn_vs_baselines`

## logs/run.log

Step-by-step execution log. Use this first when diagnosing long runs.

## logs/run_manifest.json

Reproducibility manifest.

Includes:

- run ID
- run mode
- timestamp
- root seed
- Python version
- package versions
- input config hashes
- output file hashes
- wall time

## logs/validation_report.json

Validation gates. The most important fields are:

- `critical_failed_count`
- `advisory_failed_count`
- `all_critical_passed`
- `checks`

Critical failures invalidate the run. Advisory warnings should be explained in
the report but do not automatically invalidate the run.

## SUMMARY.md

Human-readable Markdown summary.

## REPORT.html

Self-contained HTML report with tables, figures, summary text, and validation
badge.
