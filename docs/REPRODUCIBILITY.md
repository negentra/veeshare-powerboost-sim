# Reproducibility Guide

This project is deterministic for a fixed configuration, seed, Python
environment, and dependency set.

## Root Seed

The root seed is stored in:

```text
config/seed.yaml
```

Example:

```yaml
seed_root: 400
```

Each simulation module derives its own deterministic sub-seed from this root
seed. Changing the root seed creates a new independent simulation realisation.

You can override the root seed for a single run without editing the file, using
the `--seed` flag (recommended for reviewers reproducing one seed):

```bash
python run.py --mode full --seed 400 --out-dir outputs_seed400
```

`--seed` is equivalent to setting `seed_root` in `config/seed.yaml`; the
effective seed is recorded in `logs/run_manifest.json`.

## Recommended Run Pattern

Use explicit output directories so one run does not overwrite another:

```bash
python run.py --mode full --seed 400 --out-dir outputs_seed400
```

Avoid relying on the default `outputs/` directory for important evidence.

## Manifest

Every run writes:

```text
<out-dir>/logs/run_manifest.json
```

This file records:

- run mode
- root seed
- timestamp
- Python version
- package versions
- input config hashes
- output hashes
- wall time
- validation status

Keep this file with every result package.

## Validation

Every run writes:

```text
<out-dir>/logs/validation_report.json
```

Use the run only if:

```json
"critical_failed_count": 0
```

Advisory warnings should be reviewed and explained. In the latest full runs,
these advisory warnings are expected:

- `V09`: coordinated unmet demand is above the 5% advisory threshold in the
  full scenario.
- `V12`: extended PINN training currently has worse MAE than fixed-budget PINN.
- `V16`: headline CSV files do not yet include confidence-interval columns.

## Multi-Seed Evidence

Single-seed results are reproducible, but not enough to show variability. For
reporting, prefer several seeds:

Windows PowerShell:

```powershell
.\scripts\run_multiseed.ps1 -Seeds 100,200,300,400 -RunLabel final
```

Alternatively, on any platform:

```bash
python run.py --mode full --seed 100 --out-dir outputs_multiseed_final/seed100
python run.py --mode full --seed 200 --out-dir outputs_multiseed_final/seed200
python run.py --mode full --seed 300 --out-dir outputs_multiseed_final/seed300
python run.py --mode full --seed 400 --out-dir outputs_multiseed_final/seed400
```

Then compare:

```text
outputs_multiseed_final/seed100/tables/headline_numbers.csv
outputs_multiseed_final/seed200/tables/headline_numbers.csv
outputs_multiseed_final/seed300/tables/headline_numbers.csv
outputs_multiseed_final/seed400/tables/headline_numbers.csv
```

Report the mean and seed range for metrics that vary materially.

## Generated Files Are Not Versioned

Generated outputs are ignored by Git. This keeps the repository small and
prevents accidental mixing of old and new evidence.

To share evidence, send the output folder separately.
