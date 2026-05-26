# Assumptions

Every magic number, modelling choice, and parameter setting in the
simulation, with rationale. Auditor-facing.

---

### A-01: Lisbon bounding box (synthetic)
- **Value:** lat 38.70–38.78 N, lon −9.20 to −9.10 W
- **Used in:** `src/topology.py`
- **Rationale:** Synthetic plausible spatial extent matching central
  Lisbon order of magnitude. NOT a measured station registry.

### A-02: Transformer capacities
- **Value:** mix of 250, 400, 400, 630, 630, 400 kVA
- **Used in:** `src/topology.py`
- **Rationale:** Typical urban MV/LV distribution-transformer sizes.
  Order-of-magnitude representative.

### A-03: Station mix (50 stations)
- **Value:** 40 AC (7–22 kW), 8 DC fast (50–150 kW), 2 ultra-fast (>150 kW)
- **Used in:** `src/topology.py`
- **Rationale:** Approximates AFIR-conformant urban deployment ratio.

### A-04: Iberian hourly price anchor profile
- **Value:** 24-hour vector EUR/MWh, ranging 25–140
- **Used in:** `src/market.py`
- **Rationale:** Synthetic shape capturing morning shoulder + midday
  solar dip + evening peak. Mean ~70, peak ~140, trough ~25. Order of
  magnitude only; NOT fetched from OMIE/MIBEL.

### A-05: Gossip round duration
- **Value:** 80 ms per simulated round
- **Used in:** `src/sim_protocol.py`
- **Rationale:** Approximates inter-node WAN+LAN hop latency budget for
  a DePIN overlay; conservative middle estimate.

### A-06: Message size
- **Value:** 256 bytes per gossip message
- **Used in:** `src/sim_protocol.py`
- **Rationale:** Sender ID + tx ID + payload + signature stub typical
  overhead.

### A-07: Max finality rounds cap
- **Value:** 30
- **Used in:** `src/sim_protocol.py`
- **Rationale:** Protective ceiling to prevent pathological simulations.
  log2(5000)/log2(9) ≈ 4 → 30 is generous safety margin.

### A-08: Default gossip parameters
- **Value:** k=8 fan-out, finality quorum 0.667 (2/3), tx rate 2/min/node
- **Used in:** `config/scenarios.yaml`
- **Rationale:** Typical gossip-protocol BFT parameters; configurable.

### A-09: Daily charging probability per vehicle
- **Value:** 0.30
- **Used in:** `src/sim_loadbalance.py`
- **Rationale:** Approximates average passenger-EV daily charging
  frequency reported in published Dutch/German driving studies.

### A-10: Substation overload threshold
- **Value:** 0.95 × transformer kVA
- **Used in:** `src/sim_loadbalance.py`
- **Rationale:** Standard 5% safety margin used by EU DSOs.

### A-11: Sampling buckets per hour
- **Value:** 4 (i.e., 15-minute resolution)
- **Used in:** `src/sim_loadbalance.py`
- **Rationale:** Industry-standard interval for distribution-grid load
  aggregation.

### A-12: PINN architecture
- **Value:** 2 hidden layers × 64 units, **tanh** activation; physics
  residual loss weight λ_phys = 0.1, boundary penalty weight
  λ_bd = 0.01.
- **Used in:** `src/sim_pinn.py`
- **Rationale:** Compact model fitting CPU training budget. `tanh`
  chosen over ReLU for smooth derivatives required by the PINN physics
  residual (`torch.autograd`). λ values chosen as literature-typical
  starting points (Raissi et al. 2019).

### A-12a: PINN governing equation (Raissi-style ODE residual)
- **Equation:** dP/dt = κ · λ(t) · P_session − P/τ
- **Used in:** `src/sim_pinn.py::_physics_residual_loss`
- **Rationale:** First-order continuity (flow-balance) ODE for
  aggregate substation load. κ = daily session count, λ(t) = arrival-
  intensity PDF from priors, P_session = mean charging power from
  priors, τ = mean session duration from priors. Equivalent to
  continuous-service M/M/∞ queue. See `docs/PINN_PHYSICS.md` for the
  full derivation.

### A-12b: PINN learnable physical parameter (inverse-PINN)
- **Value:** κ (daily session count) is a learnable parameter,
  initialised at log(30); reported in
  `cluster3_pinn_benchmark.csv::learned_kappa_sessions_per_day`.
- **Used in:** `src/sim_pinn.py::PINNRegressor`
- **Rationale:** Inverse-PINN setup (Raissi 2019, §3.2) — the network
  simultaneously solves the forecasting task and identifies κ from
  the data + physics constraint. Provides a physically interpretable
  output beyond MAE.

### A-13: PINN training regimes
- **Value:** fixed_budget = 60 sec wall time, extended = 600 sec
  (10 min) wall time
- **Used in:** `src/sim_pinn.py`
- **Rationale:** Fixed budget allows fair comparison with LSTM/XGBoost
  at identical compute. Extended regime (10 min) characterises PINN
  near convergence: at ~1.4 sec/epoch on CPU this yields ~400 epochs,
  sufficient for >95% convergence of the data MSE and stabilisation
  of the learned κ for this smooth first-order ODE problem. Smaller
  budgets leave κ drifting; larger budgets show diminishing returns.

### A-14: Extended-regime seed count in `audit` mode
- **Value:** 1 (vs 3 in full mode)
- **Used in:** `config/scenarios.yaml`, `run.py`
- **Rationale:** Wall-time budget constraint. Single-seed extended
  result is reported with the caveat that no CI is available; full mode
  produces 3 seeds for CI.

### A-15: Vehicle population
- **Value:** 10,000 (1,000 in smoke)
- **Used in:** `config/scenarios.yaml`
- **Rationale:** Reasonable urban-district eligible fleet size for a
  50-station network slice.

### A-16: Penetration sweep
- **Value:** 30 / 50 / 70 %
- **Used in:** `config/scenarios.yaml`
- **Rationale:** Brackets Portugal AFIR trajectory (mid-2030s to stress
  test); see `config/context.yaml`.

### A-17: DSO message loss rates
- **Value:** 1% default, 5% resilience test
- **Used in:** `src/sim_protocol.py`
- **Rationale:** 1% reflects typical wide-area network packet loss
  baseline; 5% captures degraded-network resilience scenario.

### A-18: Witness threshold for verifiability
- **Value:** 3
- **Used in:** `src/sim_protocol.py`
- **Rationale:** Common minimum-quorum heuristic for distributed-ledger
  independent verification.

### A-19: DSO broadcast cadence
- **Value:** 60 simulated seconds
- **Used in:** `src/sim_protocol.py`
- **Rationale:** Typical demand-response signalling cadence in
  distribution-grid pilots.

### A-20: Portugal AFIR trajectory anchors
- **Value:** 2025=3%, 2027=8%, 2030=20%, 2035=50%
- **Used in:** `config/context.yaml` (narrative anchor only)
- **Rationale:** Order-of-magnitude reading of EU AFIR Portugal
  obligations. NEVER fed into simulation; only quoted in `SUMMARY.md`.
