# Limitations

Auditor-facing honest caveats. Read before drawing any conclusions
from this simulation's outputs.

## What this simulation does NOT model

- **Multi-DSO coordination.** A single aggregator is modelled. Real
  systems often have hierarchical or geographic aggregator layers.
- **Cross-border energy flows.** Iberian interconnection effects on
  pricing or load are not modelled.
- **Congestion pricing or dynamic ToU.** Prices are intra-day shapes;
  no real-time scarcity pricing is modelled.
- **Regulatory dynamics.** EPDK / Portuguese ERSE rule changes,
  permitting timelines, lawsuit risk — all out of scope.
- **User willingness-to-pay or behavioural response.** Arrival times
  and energy demands are sampled from priors; no price-elasticity
  feedback.
- **Battery degradation.** Charging-cycle wear on either vehicle or
  stationary storage is ignored.
- **Vehicle-to-grid (V2G) bidirectional flows.** All flows are unidir-
  ectional charging only.
- **Physical layer.** Cable losses, transformer thermal dynamics,
  reactive power, harmonic distortion — none modelled.

## What the synthetic Lisbon topology cannot tell us about real Lisbon

- Spatial clustering of stations is jittered around 6 substation
  centroids. Real MOBI.E density patterns differ.
- Transformer assignments are simplified (one substation per station,
  no redundancy).
- Real Lisbon also has rooftop PV, residential charging, district
  heating, EV-bus depots — all excluded.

## What 5 repetitions and 95% CI cannot rule out

- Rare-event behaviour (extreme weather, holiday traffic patterns,
  protocol attack scenarios).
- Black-swan loads (e.g., simultaneous EV-bus depot recharge events).
- Long-tail latency in the gossip protocol (we cap at 30 rounds).

## Where the audit findings should and should NOT be generalised

- **Should:** Generalised to other similar urban EU contexts at the
  qualitative level — "DePIN protocols scale sub-linearly", "coordinated
  charging reduces transformer overload", etc.
- **Should NOT:** Generalised to absolute kWh / kW / EUR estimates for
  any specific city. The numbers are illustrative, not predictive.

## What the PINN does and does not represent

- The Cluster 3 PINN solves a first-order continuity (flow-balance)
  ODE for aggregate substation load:
  `dP/dt = κ · λ(t) · P_session − P/τ`, with κ recovered as a learnable
  physical parameter (inverse-PINN setup per Raissi 2019). See
  `docs/PINN_PHYSICS.md` for the full derivation.
- This is **not** a PDE-PINN; it is an ODE-PINN — a strict subset of
  the Raissi family.
- The PINN does **not** model per-vehicle SOC dynamics, bidirectional
  V2G flows, or battery degradation — only the aggregate load
  continuity equation.
- The PINN physics constraint on boundary (0 ≤ P ≤ capacity) is a
  soft penalty; a hard-constraint reformulation (projection layer or
  KKT-PINN) may yield different trade-offs and should be evaluated
  before safety-critical deployment.
- The PINN has been trained only on synthetic data; field deployment
  would require retraining and re-identification of κ.

## Explicit non-use statement

**These results are not suitable for investment decisions, regulatory
submissions, grid-operation planning, or capacity-procurement decisions
without further validation through field measurement and independent
expert review.**
