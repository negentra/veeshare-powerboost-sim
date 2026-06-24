# Runbook For Non-Technical Reviewers

This runbook explains how to run and test the VeeShare PowerBoost simulation
from a clean checkout.

## What You Need

- A computer with Windows, macOS, or Linux.
- Python 3.11 or newer.
- Internet access for installing Python packages.
- About 5 GB free disk space.
- Patience for full runs. PINN training is intentionally slow.

## One-Time Setup

Open a terminal in the repository folder.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS Or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## First Check: Run Tests

```bash
python -m pytest tests/ -q
```

Expected result:

```text
19 passed
```

If tests fail, do not use the simulation outputs until the failure is
understood.

## Fast Pipeline Check

Run:

```bash
python run.py --mode smoke --out-dir outputs_smoke
```

Then open:

```text
outputs_smoke/REPORT.html
```

This smoke run is not audit evidence. It only proves that the code can run on
the machine.

## Audit Run

Run:

```bash
python run.py --mode audit --out-dir outputs_audit
```

After it finishes, check:

```text
outputs_audit/logs/validation_report.json
outputs_audit/tables/headline_numbers.csv
outputs_audit/REPORT.html
```

Critical validation failures mean the run should not be used.

## Full Run

Run:

```bash
python run.py --mode full --out-dir outputs_full
```

Full mode is slower. It runs the extended PINN setup and the regulatory
counterfactual.

## Multi-Seed Run On Windows

For a more robust result, run several independent random seeds:

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

Each seed is passed to `run.py` via `--seed`; `config/seed.yaml` is left
unchanged. To reproduce a single seed directly:

```bash
python run.py --mode full --seed 100 --out-dir outputs_seed100
```

## How To Know The Run Worked

Open `logs/validation_report.json` inside the output directory.

Good sign:

```json
"critical_failed_count": 0
```

Expected advisory warnings in full mode:

- `V09`: coordinated unmet demand is above the 5% advisory threshold.
- `V12`: extended PINN is currently worse than fixed-budget PINN.
- `V16`: headline confidence interval columns are not implemented yet.

## What To Send To Someone Else

For one run, send the whole output folder, for example:

```text
outputs_full/
```

For multi-seed evidence, send:

```text
outputs_multiseed_final/
```

Do not edit CSV files manually. If a value looks wrong, rerun the simulation or
inspect the source code and manifest.
