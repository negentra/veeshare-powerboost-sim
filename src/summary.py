"""Generate SUMMARY.md from cluster outputs and headline metrics.

Structure follows spec §8: seven mandatory sections, factual tone,
no promotional language.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt(v, decimals=2):
    if v != v:  # NaN
        return "n/a"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{decimals}f}"


def build_summary_md(
    *, headline_df: pd.DataFrame, c1_df: pd.DataFrame, c2_df: pd.DataFrame,
    c3_df: pd.DataFrame, mode: str,
) -> str:
    val = headline_df.set_index("metric_id")["value"]
    m = {mid: float(val.get(mid, float("nan"))) for mid in
         [f"M{i:02d}" for i in range(1, 21)]}

    if m["M02"] >= m["M01"]:
        throughput_sentence = (
            f"Throughput scales from\n"
            f"  {_fmt(m['M01'])} tx/sec at 1k nodes to {_fmt(m['M02'])} tx/sec\n"
            f"  at 5k nodes"
        )
    else:
        throughput_sentence = (
            f"Throughput drops from\n"
            f"  {_fmt(m['M01'])} tx/sec at 1k nodes to {_fmt(m['M02'])} tx/sec\n"
            f"  at 5k nodes"
        )

    if m["M20"] <= m["M10"]:
        pinn_budget_sentence = (
            f"Fixed 60-second\n"
            f"  training yields MAE {_fmt(m['M10'])} kW; extended training\n"
            f"  reduces this to {_fmt(m['M20'])} kW, indicating that the\n"
            f"  fixed-budget regime materially underestimates PINN capability"
        )
    else:
        pinn_budget_sentence = (
            f"Fixed 60-second\n"
            f"  training yields MAE {_fmt(m['M10'])} kW; extended training\n"
            f"  worsens this to {_fmt(m['M20'])} kW, indicating that the\n"
            f"  current PINN setup is training-budget sensitive but not yet stable"
        )

    parts = []

    parts.append(
        "# VeeShare: Decentralised P2P EV Charging Network & Grid Flexibility Audit — Simulation Summary\n"
    )
    parts.append(f"*Run mode: `{mode}`. All figures simulation-derived.*\n")

    # ---- Section 1: Methodology & Data Architecture ----
    parts.append("## 1. Methodology & Data Architecture\n")
    parts.append(
        "All numbers in this report are simulation-derived. The reference\n"
        "topology is a synthetic-but-plausible 50-station, 6-substation\n"
        "Lisbon proxy. Behavioural input distributions are calibrated\n"
        "against the order of magnitude reported in published ElaadNL\n"
        "open-dashboard materials and are used as priors, not as fits to\n"
        "any field dataset. The seed in `config/seed.yaml` and the\n"
        "manifest in `outputs/logs/run_manifest.json` together enable\n"
        "bit-exact reproduction of every headline metric.\n"
    )
    parts.append("\n```\n"
        "LAYER 1 — Network topology (Lisbon proxy)\n"
        "  - 50 charging stations across 6 substations\n"
        "  - Transformer capacities: 250/400/630 kVA mix\n"
        "  - 1 DSO/aggregator node connected to all substations\n\n"
        "LAYER 2 — Behavioural priors (ElaadNL-calibrated, synthetic)\n"
        "  - Arrival time bimodal distribution\n"
        "  - Session duration log-normal\n"
        "  - Energy demand log-normal\n"
        "  - Day-of-week multipliers\n\n"
        "LAYER 3 — Market & contextual signals (synthetic)\n"
        "  - Intra-day MIBEL-style price profile\n"
        "  - Weekly seasonality\n"
        "  - Portugal AFIR EV penetration trajectory (qualitative anchor)\n\n"
        "LAYER 4 — VeeShare DePIN simulation (audit core)\n"
        "  - P2P gossip protocol model\n"
        "  - DSO interface model (downstream / upstream)\n"
        "  - Transparency / verifiability tracking\n"
        "  - Raissi-style PINN with inverse parameter recovery\n"
        "    (Cluster 3); governing ODE in docs/PINN_PHYSICS.md\n"
        "```\n"
    )

    # ---- Section 2: Innovation Gaps Identified ----
    parts.append("## 2. Innovation Gaps Identified\n")
    parts.append(
        f"- **Gap-1: Protocol scalability ceiling.** {throughput_sentence};\n"
        f"  max stable node count (p95 latency < 2 s) is\n"
        f"  {int(m['M05']) if m['M05']==m['M05'] else 'n/a'}. Architecture\n"
        f"  review needed before pilot deployment beyond this scale (M01, M02, M05).\n"
    )
    parts.append(
        f"- **Gap-2: DSO interface latency under load.** At 1k nodes,\n"
        f"  downstream broadcast p95 is {_fmt(m['M17'])} ms and upstream\n"
        f"  aggregation p95 is {_fmt(m['M18'])} ms; these will need to be\n"
        f"  characterised against actual DSO acceptance thresholds during\n"
        f"  Solution Box (M17, M18).\n"
    )
    parts.append(
        f"- **Gap-3: PINN training-budget sensitivity.** {pinn_budget_sentence}.\n"
        f"  This points to optimisation instability under the ODE residual\n"
        f"  loss (M10 vs M20).\n"
    )
    parts.append(
        f"- **Gap-4: Coordinated charging unmet-demand trade-off.**\n"
        f"  Coordinated regime introduces {_fmt(m['M08'])} kWh of additional\n"
        f"  unmet demand at 50% penetration; trade-off boundary needs\n"
        f"  user-experience validation (M08).\n"
    )
    parts.append(
        f"- **Gap-5: Soft physics residual.** PINN under extended\n"
        f"  training still produces a {_fmt(m['M14'])}% boundary-violation\n"
        f"  rate (predicted load outside [0, capacity]); hard-\n"
        f"  constraint reformulation (projection / KKT-PINN) should be\n"
        f"  evaluated for safety-critical deployment (M14).\n"
    )

    # ---- Section 3: Infrastructure Requirements ----
    parts.append("\n## 3. Infrastructure Requirements\n")
    parts.append(
        "- **Req-1: Hierarchical-aggregation overlay** for the P2P gossip\n"
        "  layer to sustain throughput beyond the observed ceiling\n"
        "  (addresses Gap-1; M01, M02, M05).\n"
    )
    parts.append(
        "- **Req-2: DSO middleware** with deterministic broadcast and\n"
        "  upstream-aggregation guarantees at sub-2-second p95 latency\n"
        "  (addresses Gap-2; M17, M18).\n"
    )
    parts.append(
        "- **Req-3: Adaptive coordination policy** balancing peak shaving\n"
        "  with user-experience SLAs (addresses Gap-4; M06, M07, M08, M09).\n"
    )
    parts.append(
        "- **Req-4: Continuous-learning PINN deployment pipeline** allowing\n"
        "  on-site retraining within operational latency budgets (addresses\n"
        "  Gap-3 and Gap-5; M13, M14, M20).\n"
    )
    parts.append(
        "- **Req-5: Transparency / audit log** with witness-set ≥ 3\n"
        "  guaranteed by protocol (addresses transparency commitment; M19).\n"
    )

    # ---- Section 4: Technological Hurdles ----
    parts.append("\n## 4. Technological Hurdles in P2P Energy Sharing\n")
    parts.append(
        f"- **Hurdle-1: Protocol scaling.** At 5k nodes, observed throughput\n"
        f"  is {_fmt(m['M02'])} tx/sec with p95 latency of {_fmt(m['M04'])} ms;\n"
        f"  acceptable for current target deployment scale but a hard ceiling\n"
        f"  for community-scale expansion (M02, M04).\n"
    )
    parts.append(
        f"- **Hurdle-2: Residual peak load.** Even with coordinated charging,\n"
        f"  peak load is reduced by {_fmt(m['M06'])}% and transformer\n"
        f"  overload by {_fmt(m['M07'])}%; complete elimination requires\n"
        f"  additional grid-side investment (M06, M07).\n"
    )
    parts.append(
        f"- **Hurdle-3: PINN inference budget.** Extended-training PINN\n"
        f"  delivers MAE {_fmt(m['M20'])} kW with inference latency\n"
        f"  {_fmt(m['M13'], 3)} ms/sample in this benchmark; forecasting\n"
        f"  accuracy, not inference latency, is the binding maturity gap (M13, M20).\n"
    )

    # ---- Section 5: Solution Box Readiness Assessment ----
    parts.append("\n## 5. Solution Box Readiness Assessment\n")
    parts.append(
        f"The simulation places VeeShare at TRL {int(m['M15'])} (analytical\n"
        f"and experimental proof-of-concept). The Solution Box pilot is\n"
        f"expected to advance the platform to TRL {int(m['M16'])} (component\n"
        f"validation in a relevant environment) by addressing, in priority\n"
        f"order: Gap-2 (DSO interface, since real grid-operator interaction\n"
        f"can only be validated in pilot), Gap-1 (scaling beyond simulation\n"
        f"ceiling), and Gap-5 (hard physics-constraint formulation). Hurdles\n"
        f"that require structural grid investment (Hurdle-2) and\n"
        f"community-governance design (transparent transactions at the\n"
        f"socio-economic layer) are explicitly out of Solution Box scope\n"
        f"and deferred to subsequent funding stages.\n"
    )

    # ---- Section 6: Community & Transparency Considerations ----
    parts.append("\n## 6. Community & Transparency Considerations\n")
    parts.append(
        f"Transparency is partially addressed quantitatively in this audit\n"
        f"via M19, the independent-verifiability proxy ({_fmt(m['M19'])}%\n"
        f"of transactions achieve witness-set ≥ 3 at the 1k-node scale).\n"
        f"The broader audit-application objective of 'empowering local\n"
        f"energy communities through transparent, decentralized\n"
        f"transactions' has socio-economic dimensions (governance model,\n"
        f"willingness to participate, distributional outcomes) that cannot\n"
        f"be evaluated through technical simulation alone. These are\n"
        f"explicitly deferred to Solution Box pilot design, where stakeholder\n"
        f"engagement, user surveys, and governance-model selection will be\n"
        f"the appropriate instruments.\n"
    )

    # ---- Section 7: Known Limitations ----
    parts.append("\n## 7. Known Limitations\n")
    parts.append(
        "- The Lisbon topology is synthetic; it is not a substitute for\n"
        "  measured MOBI.E or E-REDES data.\n"
        "- Behavioural priors are calibrated against ElaadNL Dutch data;\n"
        "  Turkish and Portuguese arrival patterns and session lengths\n"
        "  may differ in ways not captured here.\n"
        "- The DSO interface is modelled with a single aggregator; real\n"
        "  systems may have hierarchical or geographic aggregator layers.\n"
        "- The PINN physics constraint is implemented as a soft penalty;\n"
        "  hard-constraint formulations may yield different trade-offs.\n"
        "- Market prices are synthetic intra-day shapes, not OMIE/MIBEL\n"
        "  field data; absolute cost numbers are illustrative.\n"
        "- No multi-DSO coordination or cross-border flows are modelled.\n"
        "- Bidirectional vehicle-to-grid (V2G) flows are out of scope.\n"
        "- Five protocol repetitions and the chosen 95% CI do not rule out\n"
        "  rare-event behaviour at the simulated scales.\n"
        "\n"
        "**These results are not suitable for investment decisions,\n"
        "regulatory submissions, or grid-operation planning without\n"
        "further validation.**\n"
    )

    return "".join(parts)


def write_summary_md(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit UTF-8 to avoid Windows cp1254 / cp1252 crashes on Greek
    # letters and math symbols (κ, λ, τ, ≥, etc.) in PINN/physics text.
    path.write_text(text, encoding="utf-8")
