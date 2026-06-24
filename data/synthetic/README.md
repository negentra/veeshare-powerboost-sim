# Synthetic data

This repository does not ship real charger, user, or grid operational data.

Simulation inputs are defined as **synthetic configuration** under `config/`:

| File | Purpose |
|------|---------|
| `config/seed.yaml` | Root random seed (`seed_root`) for deterministic runs |
| `config/priors.yaml` | Behavioural and physical priors (arrival rates, power mix, etc.) |
| `config/scenarios.yaml` | Run modes, cluster settings, and sweep parameters |
| `config/context.yaml` | Metric-to-audit traceability phrases |

The synthetic Lisbon proxy topology and market shapes are generated in code
(`src/topology.py`, `src/market.py`) from these priors. See `ASSUMPTIONS.md`
for every parameter rationale.