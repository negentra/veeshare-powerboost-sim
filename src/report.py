"""Auto-generate REPORT.html from outputs."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
from jinja2 import Template


HTML_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>VeeShare Grid Flexibility Audit — Simulation Report</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 920px;
         margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.5; }
  h1, h2, h3 { color: #1B3A5C; }
  h1 { border-bottom: 3px solid #2E75B6; padding-bottom: 0.3em; }
  .badge { display: inline-block; padding: 4px 10px; border-radius: 4px;
           font-weight: 600; color: white; }
  .badge.pass { background: #2ecc71; }
  .badge.warn { background: #f39c12; }
  .badge.fail { background: #e74c3c; }
  table { width: 100%; border-collapse: collapse; margin: 1em 0;
          font-size: 0.92em; }
  th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
  th { background: #1B3A5C; color: white; }
  tr:nth-child(even) { background: #f5f7fa; }
  img { max-width: 100%; height: auto; margin: 1em 0;
        border: 1px solid #e0e0e0; padding: 6px; background: #fff; }
  .meta { background: #f5f7fa; border-left: 4px solid #2E75B6;
          padding: 0.8em 1em; margin: 1em 0; font-family: monospace;
          font-size: 0.9em; }
  .disclaimer { background: #fdf6e3; border: 1px solid #f4d03f;
                padding: 1em 1.2em; border-radius: 4px; margin: 1.5em 0; }
  pre { background: #f5f7fa; padding: 0.8em; overflow-x: auto;
        border-left: 3px solid #1B3A5C; font-size: 0.85em; }
  @media print { body { max-width: none; } img { max-width: 100%; } }
</style>
</head>
<body>

<h1>VeeShare: Decentralised P2P EV Charging Network &amp; Grid Flexibility Audit — Simulation Report</h1>

<div class="meta">
Run ID: {{ run_id }}<br>
Mode: <strong>{{ mode }}</strong><br>
Timestamp: {{ timestamp }}<br>
Seed (root): {{ seed }}<br>
Wall time: {{ wall_time }}<br>
Validation: <span class="badge {{ badge_class }}">{{ badge_text }}</span>
</div>

<h2>1. Headline Metrics (P1 Deliverable)</h2>
<p>The 20 headline metrics anchoring this audit. All values are
   simulation-derived; see disclaimer at the foot of this report.</p>
{{ headline_table | safe }}

<h2>2. Figures</h2>
{% for fig in figures %}
<h3>{{ fig.title }}</h3>
<img src="data:image/png;base64,{{ fig.b64 }}" alt="{{ fig.title }}">
{% endfor %}

<h2>3. Audit Traceability</h2>
<p>Each metric anchored to a specific phrase in the PowerBoost
   application scope, with linked gaps / requirements / hurdles
   from <code>SUMMARY.md</code>.</p>
{{ traceability_table | safe }}

<h2>4. Summary (from SUMMARY.md)</h2>
<pre>{{ summary_text }}</pre>

<div class="disclaimer">
<strong>Methodology disclaimer.</strong> All numerical outputs of this
report are <strong>simulation-derived</strong>. The Lisbon topology used
as the reference network is synthetic and represents plausible structure
rather than measured field data. Behavioural input distributions are
calibrated against the order of magnitude reported in ElaadNL public
materials. No VeeShare field data was used because none exists at the
time of this audit. All conclusions drawn from these outputs must be
qualified accordingly, and the outputs are <strong>not</strong> suitable
for investment decisions, regulatory submissions, or grid-operation
planning without further validation.
</div>

</body>
</html>
"""
)


def _img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_report_html(
    *,
    out_path: Path,
    run_id: str,
    mode: str,
    timestamp: str,
    seed: int,
    wall_time: str,
    headline_df: pd.DataFrame,
    traceability_df: pd.DataFrame,
    figure_paths: dict,
    summary_text: str,
    validation_status: str,
) -> None:
    """Write REPORT.html. validation_status in {pass, warn, fail}."""
    badge_class = {"pass": "pass", "warn": "warn", "fail": "fail"}.get(
        validation_status, "warn"
    )
    badge_text = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}.get(
        validation_status, "WARN"
    )
    headline_html = headline_df.to_html(index=False, classes="headline",
                                        float_format="{:.3f}".format)
    trace_html = traceability_df.to_html(index=False, classes="trace")

    figures = []
    titles = {
        "fig1": "Figure 1 — DePIN protocol throughput vs node count",
        "fig2": "Figure 2 — P2P consensus & DSO interface latency",
        "fig3": "Figure 3 — Baseline vs coordinated load profile",
        "fig4": "Figure 4 — PINN vs LSTM vs XGBoost prediction quality",
    }
    for key, p in figure_paths.items():
        figures.append({"title": titles.get(key, key), "b64": _img_b64(p)})

    html = HTML_TEMPLATE.render(
        run_id=run_id, mode=mode, timestamp=timestamp,
        seed=seed, wall_time=wall_time,
        headline_table=headline_html,
        traceability_table=trace_html,
        figures=figures, summary_text=summary_text,
        badge_class=badge_class, badge_text=badge_text,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit UTF-8 — HTML may contain Greek letters from PINN section.
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
