# VeeShare PowerBoost Simulation

This repository contains a reproducible simulation pipeline for the
VeeShare PowerBoost Innovation Audit. It is designed so that a reviewer can
install it, run a quick test, run the full audit simulation, and inspect the
generated evidence without needing prior knowledge of the codebase.

All numerical results are simulation-derived. They are not field
measurements and must not be used as investment, regulatory, or grid
operation evidence without pilot validation.

## What The Simulation Does

The pipeline has four workstreams:

1. Protocol and DSO interface scaling
   - P2P gossip throughput and latency from 100 to 5,000 nodes.
   - DSO downstream broadcast latency and upstream aggregation latency.
   - Independent transaction verifiability proxy.

2. Coordinated EV charging load balancing
   - Baseline versus coordinated charging.
   - 30%, 50%, and 70% EV penetration scenarios.
   - 50 charging stations and 6 substations in a synthetic Lisbon proxy grid.

3. Forecasting and PINN maturity benchmark
   - Raissi-style Physics-Informed Neural Network.
   - LSTM baseline.
   - XGBoost baseline.
   - Next-hour substation-load forecasting.

4. Regulatory counterfactual
   - Restrictive Turkish-style licensing assumptions versus a liberalised
     reference regime.
   - Weekly welfare proxy comparison.

## Repository Contents

```text
.
|-- run.py                    Main entry point
|-- requirements.txt          Python dependency list
|-- pyproject.toml            Project metadata and pytest config
|-- config/
|   |-- seed.yaml             Root random seed
|   |-- priors.yaml           Behavioural assumptions
|   |-- scenarios.yaml        Run modes and simulation settings
|   `-- context.yaml          Metric-to-audit traceability phrases
|-- src/                      Simulation and report-generation code
|-- tests/                    Unit and smoke tests
|-- docs/
|   |-- RUNBOOK.md            Step-by-step instructions for non-developers
|   |-- OUTPUTS.md            Explanation of generated files
|   |-- REPRODUCIBILITY.md    How deterministic runs work
|   `-- PINN_PHYSICS.md       PINN governing-equation notes
|-- scripts/
|   `-- run_multiseed.ps1     Optional Windows multi-seed runner
|-- ASSUMPTIONS.md            Modelling assumptions
|-- LIMITATIONS.md            Known limitations
|-- CHANGELOG.md              Version history
`-- LICENSE
```

Generated folders such as `outputs/`, `outputs_seed*`, `backups/`,
`sonuclar/`, and virtual environments are intentionally not tracked.

## Quick Start

Use Python 3.11 or newer. Python 3.12 was used for the latest local runs.

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the tests

```bash
python -m pytest tests/ -q
```

Expected result:

```text
19 passed
```

### 3. Run a quick smoke simulation

This verifies that the pipeline works end to end. It is intentionally small.

```bash
python run.py --mode smoke --out-dir outputs_smoke
```

Open:

```text
outputs_smoke/REPORT.html
outputs_smoke/SUMMARY.md
outputs_smoke/logs/validation_report.json
```

### 4. Run the audit simulation

This is the practical single-seed audit run.

```bash
python run.py --mode audit --out-dir outputs_audit
```

### 5. Run the full simulation

This is slower and includes the full extended PINN setup.

```bash
python run.py --mode full --out-dir outputs_full
```

On a normal laptop, full mode can take tens of minutes per seed. The PINN
training stage is the slow part.

## Optional Multi-Seed Run

For stronger evidence, run several seeds and compare averages.

Windows PowerShell:

```powershell
.\scripts\run_multiseed.ps1 -Seeds 100,200,300,400 -RunLabel final
```

This creates:

```text
outputs_multiseed_final/
|-- seed100/
|-- seed200/
|-- seed300/
|-- seed400/
`-- _logs/
```

Each seed folder has the same output structure as a normal `run.py` run.

Alternatively, run the seeds directly with `--seed` (any platform, no
PowerShell required):

```bash
python run.py --mode full --seed 100 --out-dir outputs_multiseed_final/seed100
python run.py --mode full --seed 200 --out-dir outputs_multiseed_final/seed200
python run.py --mode full --seed 300 --out-dir outputs_multiseed_final/seed300
python run.py --mode full --seed 400 --out-dir outputs_multiseed_final/seed400
```

Then report the mean and seed range for metrics that vary materially.

## Important Validation Notes

The latest corrected validation logic treats these as advisory warnings:

- `V09`: coordinated unmet demand is above the 5% advisory threshold in the
  full scenario.
- `V12`: extended PINN training currently has worse MAE than fixed-budget PINN.
- `V16`: headline CSV files do not yet include confidence-interval columns.

Critical validation checks should pass. If a critical validation check fails,
the run should not be used as audit evidence.

## Main Output Files

Every run writes these files under the selected `--out-dir`:

```text
tables/headline_numbers.csv
tables/audit_traceability.csv
tables/cluster1_protocol_scaling.csv
tables/cluster2_load_balancing.csv
tables/cluster3_pinn_benchmark.csv
tables/counterfactual_TR_vs_liberalised.csv
figures/*.png
figures/*.svg
logs/run.log
logs/run_manifest.json
logs/validation_report.json
SUMMARY.md
REPORT.html
```

See [docs/OUTPUTS.md](docs/OUTPUTS.md) for details.

## How To Reproduce A Result

1. Choose a root seed: either set `seed_root` in `config/seed.yaml`, or pass
   `--seed N` on the command line. `--seed` overrides the file for that run,
   leaves `config/seed.yaml` untouched, and records the effective seed in
   `logs/run_manifest.json`.
2. Run the desired mode with an explicit output directory.
3. Check `logs/run_manifest.json`.
4. Check `logs/validation_report.json`.

Example (reproduce seed 400 without editing any file):

```bash
python run.py --mode full --seed 400 --out-dir outputs_seed400
```

For more detail, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Known Limitations

The grid topology, behavioural sessions, prices, and regulatory scenarios are
synthetic. The model is intended to identify technical risks and pilot
questions, not to replace measured DSO or field data.

See [LIMITATIONS.md](LIMITATIONS.md).

## Funding

This activity has received support from the GreenGrid Eurocluster project
under the European Union Single Market Programme (SMP-COSME).

## License

MIT. See [LICENSE](LICENSE).
