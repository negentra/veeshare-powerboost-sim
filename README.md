# VeeShare: Decentralised P2P EV Charging Network & Grid Flexibility Audit

Reproducible simulation and evidence-generation code for
**VeeShare: Decentralised P2P EV Charging Network & Grid Flexibility Audit**,
developed by **Negentra Yazılım ve Oyun Teknolojileri A.Ş.**

> **Repository boundary.** This repository contains simulation and reproducibility
> code for *VeeShare: Decentralised P2P EV Charging Network & Grid Flexibility
> Audit*, carried out under the PowerBoost Open Call / GreenGrid Eurocluster
> innovation audit. It does **not** contain production platform code, commercial
> deployment logic, real user data, payment infrastructure, DSO integrations, OCPP
> production integrations, token/crypto logic, or operational charger-control
> software.

## Purpose

This project lets an independent reviewer install the code, run deterministic
simulation experiments, and inspect synthetic audit evidence (CSV tables, charts,
validation gates, and provenance manifests) without prior knowledge of the
codebase.

All numerical results are **simulation-derived**. They are not field measurements
and must not be used for investment, regulatory, grid-operation, or commercial
deployment decisions without pilot validation and expert review.

## Audit context

This repository supports **VeeShare: Decentralised P2P EV Charging Network &
Grid Flexibility Audit**, submitted under the **PowerBoost Open Call** of the
**GreenGrid Eurocluster** cascade-funding programme (European Union Single
Market Programme, **SMP-COSME**).

The repository is scoped narrowly to simulation, benchmarking, reproducibility,
synthetic-data generation, chart generation, and audit-evidence packaging for
that audit.

## Funding acknowledgement

This activity has received support from the GreenGrid Eurocluster project under
the European Union Single Market Programme (SMP-COSME).

## What this repository contains

- Python simulation pipeline (`run.py`, `src/`)
- Synthetic configuration (`config/`)
- Unit and smoke tests (`tests/`)
- Multi-seed runner script (`scripts/run_multiseed.ps1`)
- Documentation for outputs, reproducibility, and physics (`docs/`)
- Modelling assumptions and limitations (`ASSUMPTIONS.md`, `LIMITATIONS.md`)

## What this repository does not contain

- Production EV-charging or DePIN platform code
- Real user, charger, or DSO operational data
- Credentials, API keys, wallets, or payment infrastructure
- OCPP production integrations or live charger-control software
- Commercial deployment logic or proprietary business rules
- Token, cryptocurrency, or blockchain operational code

Synthetic parameters live in `config/`; see `data/synthetic/README.md`.

## Repository layout

```text
.
|-- README.md
|-- LICENSE                 Apache License, Version 2.0 (full text)
|-- NOTICE                  Attribution and funding notice
|-- run.py                  Main entry point
|-- requirements.txt
|-- pyproject.toml
|-- config/                 Synthetic seeds, priors, scenarios
|-- data/synthetic/         Pointer docs (no real data files)
|-- scripts/                Multi-seed runner
|-- src/                    Simulation and report code
|-- tests/
|-- docs/                   RUNBOOK, OUTPUTS, REPRODUCIBILITY, PINN_PHYSICS
|-- outputs/sample/         How to generate local evidence (not committed)
|-- ASSUMPTIONS.md
|-- LIMITATIONS.md
`-- CHANGELOG.md
```

Generated run folders (`outputs_*`, `outputs_multiseed_*`, etc.) are gitignored.

## Required environment

- **Python** 3.11 or newer (3.12 tested locally)
- **OS:** Windows, macOS, or Linux
- **Disk:** ~5 GB free for dependencies and outputs
- **Time:** smoke ~1 min; audit ~20–25 min; full multi-seed ~hours (PINN training)

### Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Verify installation

```bash
python -m pytest tests/ -q
```

Expected: `19 passed`.

## How to reproduce the audit simulation

### Single-seed audit run

```bash
python run.py --mode audit --seed 100 --out-dir outputs_audit_seed100
```

### Full simulation (extended PINN + counterfactual)

```bash
python run.py --mode full --seed 100 --out-dir outputs_full_seed100
```

### Multi-seed pipeline (recommended for variability evidence)

Windows PowerShell:

```powershell
.\scripts\run_multiseed.ps1 -Seeds 100,200,300,400 -RunLabel final
```

Cross-platform equivalent (no PowerShell required):

```bash
python run.py --mode full --seed 100 --out-dir outputs_multiseed_final/seed100
python run.py --mode full --seed 200 --out-dir outputs_multiseed_final/seed200
python run.py --mode full --seed 300 --out-dir outputs_multiseed_final/seed300
python run.py --mode full --seed 400 --out-dir outputs_multiseed_final/seed400
```

Report the mean and seed range for metrics that vary materially across seeds.

## Seed reproducibility

| Concept | Location | Notes |
|---------|----------|-------|
| `seed_root` | `config/seed.yaml` | Default root seed when `--seed` is omitted |
| `--seed N` | `run.py` CLI | Overrides `seed_root` for one run; file unchanged |
| Effective seed | `logs/run_manifest.json` | Recorded per run |
| Sub-seeds | `src/utils/seed.py` | Derived deterministically from root seed per module |

Example — reproduce seed 400 without editing any file:

```bash
python run.py --mode full --seed 400 --out-dir outputs_seed400
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for manifest hashes and
multi-seed comparison guidance.

## Expected output files

Each `--out-dir` receives:

```text
tables/headline_numbers.csv          # 20 headline metrics (M01–M20)
tables/audit_traceability.csv        # Metric-to-audit mapping
tables/cluster1_protocol_scaling.csv
tables/cluster2_load_balancing.csv
tables/cluster3_pinn_benchmark.csv
tables/counterfactual_TR_vs_liberalised.csv   # audit/full modes
figures/fig1_protocol_throughput.png|.svg
figures/fig2_protocol_and_dso_latency.png|.svg
figures/fig3_load_baseline_vs_coordinated.png|.svg
figures/fig4_pinn_vs_baselines.png|.svg
logs/run.log
logs/run_manifest.json               # SHA-256 output hashes + provenance
logs/validation_report.json          # Critical/advisory validation gates
SUMMARY.md
REPORT.html
```

Charts are generated under `<out-dir>/figures/` by `src/figures.py` during the
run. Details: [docs/OUTPUTS.md](docs/OUTPUTS.md).

### SHA-256 manifest

`logs/run_manifest.json` records:

- run mode and effective root seed
- Python version and package versions
- SHA-256 hashes of input configs and output artefacts
- wall time and validation status

Use this file to verify that a result package matches a specific code + config
state.

### Validation

Open `logs/validation_report.json`. A run is acceptable for audit evidence when:

```json
"critical_failed_count": 0
```

Expected advisory warnings in full mode (see [docs/RUNBOOK.md](docs/RUNBOOK.md)):

- `V09`: coordinated unmet demand above 5% advisory threshold
- `V12`: extended PINN MAE worse than fixed-budget PINN
- `V16`: headline CSV lacks confidence-interval columns

## Quick smoke test

```bash
python run.py --mode smoke --out-dir outputs_smoke
```

Smoke output is for pipeline sanity only, not audit evidence.

## Further reading

- [docs/RUNBOOK.md](docs/RUNBOOK.md) — step-by-step for non-developers
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) — seeds, manifests, multi-seed
- [docs/OUTPUTS.md](docs/OUTPUTS.md) — column definitions
- [ASSUMPTIONS.md](ASSUMPTIONS.md) — every modelling parameter
- [LIMITATIONS.md](LIMITATIONS.md) — what the simulation cannot claim

## Disclaimer

The repository is provided by Negentra for audit transparency and
reproducibility purposes. All results are simulation-derived and require pilot
validation.

THE SOFTWARE AND ALL SIMULATION OUTPUTS ARE PROVIDED "AS IS", WITHOUT WARRANTY
OF ANY KIND, EXPRESS OR IMPLIED. THIS REPOSITORY IS FOR **SIMULATION AND AUDIT
REPRODUCIBILITY ONLY**. DO NOT USE OUTPUTS FOR PRODUCTION DEPLOYMENT, LIVE
GRID OPERATION, REGULATORY FILINGS, OR INVESTMENT DECISIONS WITHOUT INDEPENDENT
FIELD VALIDATION.

Publication of this repository does not imply that the European Union, GreenGrid
Eurocluster, Sunrise Tech Park, or the PowerBoost Open Call endorses the
software, validates the simulation results, or certifies the repository.

## Licence

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).

```text
SPDX-License-Identifier: Apache-2.0
```