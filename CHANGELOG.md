# Changelog

All notable changes to the VeeShare PowerBoost audit simulation.

## [v3.1.5] — 2026-06 (reviewer ergonomics)

- **Added `--seed` CLI override to `run.py`.** A reviewer can now reproduce
  any single seed directly, e.g. `python run.py --mode full --seed 100`,
  without editing `config/seed.yaml`. When omitted, behaviour is unchanged
  (the `seed_root` from `config/seed.yaml` is used). The effective seed is
  recorded in `run_manifest.json`.
- **Non-breaking.** It does not change any default-run numbers and is
  mathematically equivalent to editing `seed_root`: both feed the same root
  seed into `set_global_seeds`, which derives all per-module sub-seeds.
- **`scripts/run_multiseed.ps1` simplified** to pass `--seed` per run instead
  of rewriting and restoring `config/seed.yaml`; outputs and numbers are
  identical to the previous method.
- **Docs updated** (`docs/REPRODUCIBILITY.md`, `docs/RUNBOOK.md`) with the
  single-command per-seed pattern.
- No simulation-code logic changes vs v3.1.4; all 20 headline metrics, the P5
  counterfactual, the 4 figures, and the validation gates are unchanged.

## [v3.1.4] — 2026-05 (PowerBoost submission release)

- **PowerBoost Innovation Audit submission-ready release.** This is the
  version corresponding to the `Negentra_PowerBoost_ChallengeDescription_2026`
  Annex II report (3-page deliverable) submitted to the Sunrise Tech Park
  coordinator under GreenGrid Eurocluster Sub-Grant Agreement (GA No.
  101236890).
- **No simulation-code logic changes vs v3.1.3.** Test suite (`pytest`)
  expanded with `slow` marker registration in `conftest.py` to remove
  pytest warnings; `test_validation.py` updated to reflect the V16
  uncertainty-column advisory gate added in v3.1.2.
- **Reproducibility statement.** Running `python run.py --mode audit` on
  a stock Python 3.10-3.12 environment reproduces all 20 headline metrics,
  4 figures, the P5 counterfactual table, the validation report, and the
  audit traceability matrix that back the Annex II Report.

## [v3.1.3] — 2026-05

- **P5 counterfactual analysis IMPLEMENTED.** Previously the spec
  declared a P5 stretch (`counterfactual_TR_vs_liberalised.csv`) but no
  code produced it; in `full` mode the run silently skipped it. Now
  `src/sim_counterfactual.py` implements it properly:
  - Holds the session realisation constant across both regimes
    (same arrival/duration/energy patterns); varies only which subset
    of stations is regulatorily 'active' and the pricing band.
  - TR regime: 12/50 stations (licensed-CPO model), ±20% price band,
    deterministic CPO selection by power capacity.
  - Liberalised regime: 50/50 stations, no price intervention.
  - Welfare proxy: consumer surplus − unmet-demand penalty,
    EUR/week.
  - Welfare delta is now reported as a headline number for the
    academic paper.
- **P5 is now enabled in both `audit` and `full` modes** so the audit
  deliverable is complete; the only `full` vs `audit` difference now
  is the extended-PINN seed count (3 vs 1).
- **LSTM training now also emits 30s heartbeats to the log file**
  (previously only PINN did). Both deep-learning training loops are
  now silent-failure-proof.

## [v3.1.2] — 2026-05

- **Windows QuickEdit / terminal-pause hardening.** The previous run took
  7 hours wall time not because of computation but because the Windows
  console paused stdout writes when the user clicked the window. Fix:
  - `_disable_windows_quickedit()` at startup turns off QuickEdit and
    Insert modes on the current console (ctypes call to kernel32).
  - `sys.stdout.reconfigure(line_buffering=True)` so per-line flushes
    happen even without explicit `flush=True`.
  - Custom `FlushingFileHandler` in `run.py` flushes the log file after
    EVERY record. Even if the terminal hangs or the process is killed,
    the log captures progress up to the last instant.
  - PINN training loop now emits a heartbeat to the log file every 30
    seconds with epoch, elapsed time, val loss, and current learned κ.
    User can `tail -f outputs/logs/run.log` (or check Notepad) to
    confirm progress without depending on the terminal.

## [v3.1.1] — 2026-05

- **Windows encoding hardening.** All file I/O explicitly uses
  `encoding="utf-8"`. Fixes `UnicodeEncodeError` on Windows TR/EU
  locales (cp1254/cp1252) when SUMMARY.md / REPORT.html / manifests
  contain Greek letters (κ, λ, τ) or math symbols (≥, ≤).
  Affected: `src/summary.py`, `src/report.py`, `src/utils/manifest.py`,
  `src/priors.py`, `src/sim_pinn.py`, `run.py` (logging FileHandler +
  validation_report JSON + all CSV writes + YAML reads + stdout
  reconfigure), `tests/test_pinn_physics.py`.
- **Defensive write protection.** SUMMARY.md, REPORT.html, and
  manifest writes wrapped in try/except so a single artefact failure
  cannot lose the audit deliverables (headline + traceability CSVs +
  figures are written first and protected).
- **PINN extended training budget bumped 5 → 10 minutes.** At
  ~1.4 sec/epoch on CPU this raises convergence from ~200 to ~400
  epochs (~85% → >95% converged for this ODE). Audit-mode wall time
  rises from ~13 to ~18-20 minutes, still well inside the 25-minute
  target and the 37-minute validation hard limit.

## [v3.1.0] — 2026-05

- **Real Raissi-style PINN.** Cluster 3 now implements a true
  physics-informed neural network per Raissi-Perdikaris-Karniadakis
  (2019), replacing the v3.0 physics-constrained bounded MLP.
  - Governing equation: first-order ODE for aggregate substation
    load — `dP/dt = κ · λ(t) · P_session − P/τ`.
  - Physics residual computed via `torch.autograd.grad` with chain
    rule through the cyclic hour-of-day features.
  - Activation switched from ReLU to **tanh** for smooth derivatives.
  - **Inverse PINN**: κ (daily session count) is a learnable physical
    parameter, reported per training run as
    `learned_kappa_sessions_per_day` in
    `cluster3_pinn_benchmark.csv`.
  - New `docs/PINN_PHYSICS.md` documenting derivation and protocol.
- TÜBİTAK 1507 and GreenGrid Eurocluster cleared for using the V2G
  aggregate-flow ODE formulation; no double-funding concern.

## [v3.0.0] — 2026-05

- First production implementation of spec v3.
- Three run modes (smoke / audit / full).
- 20 headline metrics, 18-row Cluster 2 sweep, 10-row Cluster 3 benchmark.
- DSO interface model and transparency tracking integrated into Cluster 1.
- PINN trained in two regimes (fixed_budget, extended).
- 15 validation gates with critical / advisory classification.
- Auto-generated `REPORT.html`.
- Full traceability matrix (`audit_traceability.csv`) anchoring every
  metric to a phrase in the PowerBoost application scope.
