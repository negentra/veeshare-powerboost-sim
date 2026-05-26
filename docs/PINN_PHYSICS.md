# PINN Physics — Governing Equation & Inverse Formulation

This document specifies the physics-informed neural network (PINN)
formulation used in `src/sim_pinn.py`, following the
Raissi–Perdikaris–Karniadakis (2019) framework.

## 1. Governing equation

The aggregate charging-load at a single substation P(t) [kW] is
modelled by the **first-order continuity (flow-balance) ODE**:

    dP/dt = κ · λ(t) · P_session − P(t) / τ

**Physical interpretation:**

- **κ ∈ ℝ⁺**: total daily session count attached to the substation.
  **Learnable** parameter (inverse-PINN setup), see §4.
- **λ(t)**: normalised arrival-intensity PDF on [0, 24) hours,
  reflecting bimodal commuter behaviour. Derived from `priors.yaml`
  (mixture of two truncated normals + uniform background);
  integrates to 1 over the day.
- **P_session**: mean charging power per session [kW], computed
  analytically from the `charging_power_mix` in `priors.yaml`.
- **τ**: mean session duration [h], the mean of the log-normal in
  `priors.yaml`:  τ = exp(μ_log + σ_log² / 2).
- **First term** (source): energy flowing into the substation as new
  sessions begin.
- **Second term** (first-order relaxation): sessions terminate at
  characteristic timescale τ, equivalent to the continuous-service
  approximation of an M/M/∞ queue.

At equilibrium (dP/dt = 0, daily averaged):
    P_eq = κ · P_session · τ / 24

## 2. PINN architecture

A fully-connected MLP with `tanh` activations (smooth derivatives,
required for clean autograd):

    P_θ(x) = MLP(x; θ)

where x ∈ ℝ²⁹ contains 24 h of lagged load, two sin/cos hour-of-day
features, two sin/cos day-of-week features, and a temperature proxy.

## 3. Composite loss

    L = L_data + λ_phys · L_phys + λ_bd · L_bd

where:

    L_data = (1/N) Σ ‖P_θ(x_n) - P_true_n‖²

    L_phys = (1/N_c) Σ ‖ ∂P_θ/∂t |_c - κ_θ · λ(t_c) · P_session + P_θ(x_c)/τ ‖²

    L_bd   = (1/N) Σ [ReLU(-P_θ)]² + [ReLU(P_θ - C_n)]²

Weights: λ_phys = 0.1, λ_bd = 0.01.

## 4. Computing ∂P/∂t via autograd (Raissi protocol)

The time variable enters the network indirectly through cyclic
encoding: s_h = sin(2π·t/24), c_h = cos(2π·t/24). The temporal
derivative is recovered by **automatic differentiation + chain rule**:

    ∂P_θ/∂t = (∂P_θ/∂s_h)·(ds_h/dt) + (∂P_θ/∂c_h)·(dc_h/dt)

with:
    ds_h/dt = (2π/24)·c_h
    dc_h/dt = -(2π/24)·s_h

The gradients ∂P_θ/∂s_h and ∂P_θ/∂c_h are computed via
`torch.autograd.grad(p_pred, x, create_graph=True)`.

## 5. Inverse-PINN: learnable physical parameter

Following Raissi et al. (2019, §3.2), the daily session count κ is
**not fixed**; it is a learnable parameter:

    κ_θ = exp(θ_log_κ),   θ_log_κ ∈ ℝ

The exponential parametrisation ensures positivity. Initial value:
θ_log_κ,0 = log(30) — a prior of 30 sessions/day/substation.

During training, κ is updated jointly with the network weights. **The
learned value is reported in the output CSV** as
`learned_kappa_sessions_per_day` — a direct PINN-recovered physical
estimate from the simulated load data.

This is the **inverse problem** formulation: the network simultaneously
solves the forecasting task and identifies a key physical parameter
of the governing ODE.

## 6. Collocation points

In this implementation, collocation points coincide with the training
batch — every training sample contributes both a data-loss and a
physics-residual term. This is a standard simplification when the data
is densely sampled across the time domain (hourly samples across
multiple weeks).

A future extension can add separately-sampled collocation points
covering hours/days under-represented in the training set, letting
the physics residual regularise the network in extrapolation regions.

## 7. What this PINN is NOT

To prevent overclaim:

- **Not a PDE-PINN.** This is a first-order ODE PINN. The original
  Raissi paper covers both PDEs and ODEs; ODE-PINNs are a strict
  subset of the family.
- **Not solving the full V2G dynamics.** A complete V2G model would
  include per-vehicle SOC trajectories, bidirectional flows, and
  battery degradation. This PINN models only the aggregate load
  continuity equation, which is sufficient for substation-level
  forecasting but not for individual vehicle dispatch decisions.
- **Not validated against measured field data.** The training data is
  synthetic (produced by `sim_loadbalance.py`). Real-world deployment
  would require retraining and re-identification of κ.

## 8. Reference

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
*Physics-informed neural networks: A deep learning framework for
solving forward and inverse problems involving nonlinear partial
differential equations.* Journal of Computational Physics, 378, 686–707.
