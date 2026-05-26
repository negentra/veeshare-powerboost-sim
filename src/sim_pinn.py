"""Cluster 3 — PINN digital maturity benchmark (Raissi-style real PINN).

Compares three models on next-hour substation-load forecasting:
  1. PINN  — neural net with physics-residual loss from a first-order
             ODE governing aggregate charging-load dynamics (continuity
             equation). Residual computed via torch.autograd. Includes
             one learnable physical parameter (κ, daily arrival rate) —
             the inverse-PINN setup from Raissi-Perdikaris-Karniadakis (2019).
  2. LSTM  — 2-layer LSTM baseline.
  3. XGBoost — gradient-boosted trees baseline.

Governing equation:
    dP/dt = κ · λ(t) · P_avg_session  -  P / τ
where:
    κ        = daily aggregate arrival count (learnable parameter)
    λ(t)     = bimodal arrival-intensity PDF over [0, 24] (from priors)
    P_avg    = mean charging power per session (from priors)
    τ        = mean session duration hours (from priors)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from src.priors import Priors
from src.sim_loadbalance import (
    BUCKETS_PER_HOUR,
    _generate_sessions,
    _simulate_regime,
)
from src.topology import build_topology
from src.utils.seed import derive_seed


# ---------------------------------------------------------------------------
# Physical parameters derived from priors
# ---------------------------------------------------------------------------
def compute_p_session_avg_kw(priors_cfg: dict) -> float:
    """Mean charging power per session, kW, from the mixture in priors."""
    m = priors_cfg["charging_power_mix"]
    o, t, d = m["one_phase"], m["three_phase"], m["dc_fast"]
    total = o["weight"] + t["weight"] + d["weight"]
    avg = (
        o["weight"] * (o["min_kw"] + o["max_kw"]) / 2
        + t["weight"] * (t["min_kw"] + t["max_kw"]) / 2
        + d["weight"] * (d["min_kw"] + d["max_kw"]) / 2
    ) / total
    return float(avg)


def compute_tau_hours(priors_cfg: dict) -> float:
    """Mean session duration in hours (mean of log-normal)."""
    d = priors_cfg["session_duration_hours"]
    return float(np.exp(d["mu_log"] + d["sigma_log"] ** 2 / 2))


def arrival_intensity_pdf_torch(hour, priors_cfg):
    """λ(t): bimodal arrival-intensity PDF on [0, 24], as a torch tensor.

    Built from priors.yaml: mixture of two truncated normals (morning,
    evening) + uniform off-peak component.
    """
    import torch

    a = priors_cfg["arrival_time"]
    m = a["morning_peak"]
    e = a["evening_peak"]
    uniform_weight = a["off_peak_uniform_weight"]

    inv_sqrt_2pi = 1.0 / float(np.sqrt(2.0 * np.pi))

    def _norm(x, mu, sigma):
        return inv_sqrt_2pi / sigma * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    pdf = (
        m["weight"] * _norm(hour, m["mean_hour"], m["std_hour"])
        + e["weight"] * _norm(hour, e["mean_hour"], e["std_hour"])
        + uniform_weight / 24.0
    )
    return pdf


# ---------------------------------------------------------------------------
# Synthetic dataset construction
# ---------------------------------------------------------------------------
def _build_dataset(
    seed_root: int,
    cfg_pinn: dict,
    cfg_lb_template: dict,
    priors_path: Path,
) -> Tuple:
    """Build train/val/test arrays from synthetic load weeks."""
    priors = Priors.load(priors_path)
    week_hours = 168
    weeks_total = cfg_pinn["weeks_total"]
    total_hours = week_hours * weeks_total

    cfg_lb = dict(cfg_lb_template)
    cfg_lb["week_hours"] = total_hours
    cfg_lb["repetitions"] = 1
    cfg_lb["penetration_pcts"] = [50]
    cfg_lb["regimes"] = ["baseline"]

    stations, substations, _ = build_topology(
        seed_root,
        station_count=cfg_lb["station_count"],
        substation_count=cfg_lb["substation_count"],
    )
    sessions = _generate_sessions(
        seed_root, rep=1, penetration_pct=50, cfg=cfg_lb,
        priors=priors, stations=stations,
    )

    week_buckets = total_hours * BUCKETS_PER_HOUR
    sub_idx = {s.substation_id: i for i, s in enumerate(substations)}
    load = np.zeros((len(substations), week_buckets))
    bucket_hours = 1.0 / BUCKETS_PER_HOUR
    for s in sessions:
        duration = s.departure_bucket - s.arrival_bucket
        if duration <= 0:
            continue
        ideal_buckets = int(
            np.ceil(s.requested_kwh / (s.requested_power_kw * bucket_hours))
        )
        usable = min(ideal_buckets, duration)
        si = sub_idx[s.substation_id]
        load[si, s.arrival_bucket : s.arrival_bucket + usable] += s.requested_power_kw

    hourly = load.reshape(len(substations), total_hours, BUCKETS_PER_HOUR).mean(axis=2)

    look = cfg_pinn["lookback_hours"]
    horizon = cfg_pinn["forecast_horizon_hours"]
    n_sub = len(substations)
    X_list, y_list, cap_list = [], [], []

    rng_temp = np.random.default_rng(derive_seed(seed_root, "sim_pinn_temp"))
    t_proxy = 15 + 8 * np.sin(2 * np.pi * np.arange(total_hours) / (24 * 30)) + rng_temp.normal(
        0, 1, size=total_hours
    )

    for s_i in range(n_sub):
        cap_kw = substations[s_i].capacity_kva * 0.95
        for t in range(look, total_hours - horizon):
            hist = hourly[s_i, t - look : t]
            hour = t % 24
            dow = (t // 24) % 7
            feats = np.concatenate(
                [
                    hist,
                    [
                        np.sin(2 * np.pi * hour / 24),
                        np.cos(2 * np.pi * hour / 24),
                        np.sin(2 * np.pi * dow / 7),
                        np.cos(2 * np.pi * dow / 7),
                        t_proxy[t],
                    ],
                ]
            )
            X_list.append(feats)
            y_list.append(hourly[s_i, t + horizon - 1])
            cap_list.append(cap_kw)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    caps = np.array(cap_list, dtype=np.float32)

    n_per_sub = total_hours - look - horizon
    train_hours = cfg_pinn["weeks_train"] * week_hours
    val_hours = cfg_pinn["weeks_val"] * week_hours

    def _split(seg_start, seg_end):
        idx = []
        for s_i in range(n_sub):
            base = s_i * n_per_sub
            lo = max(0, seg_start - look)
            hi = max(0, seg_end - look)
            idx.extend(range(base + lo, min(base + hi, base + n_per_sub)))
        return np.array(idx)

    train_idx = _split(0, train_hours)
    val_idx = _split(train_hours, train_hours + val_hours)
    test_idx = _split(train_hours + val_hours, total_hours - horizon)

    if val_idx.size == 0:
        val_idx = train_idx[-max(1, len(train_idx) // 10) :]
    if test_idx.size == 0:
        test_idx = train_idx[-1:]

    return (
        (X[train_idx], y[train_idx], caps[train_idx]),
        (X[val_idx], y[val_idx], caps[val_idx]),
        (X[test_idx], y[test_idx], caps[test_idx]),
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _make_pinn_model(in_dim: int, cfg_pinn: dict, lookback: int,
                     p_session_avg: float, tau_hours: float):
    """Build a Raissi-style PINN with smooth (tanh) activations and a
    learnable physical parameter κ.
    """
    import torch
    import torch.nn as nn

    class PINNRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lookback = lookback
            self.p_session_avg = p_session_avg
            self.tau_hours = tau_hours
            H = cfg_pinn["pinn_hidden_units"]
            L = cfg_pinn["pinn_hidden_layers"]
            seq = []
            prev = in_dim
            for _ in range(L):
                seq.append(nn.Linear(prev, H))
                seq.append(nn.Tanh())  # smooth derivatives for autograd
                prev = H
            seq.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*seq)
            # Inverse PINN: learnable daily arrival count, init = 30
            self.log_kappa = nn.Parameter(torch.tensor(float(np.log(30.0))))

        @property
        def kappa(self):
            return torch.exp(self.log_kappa)

        def forward(self, x):
            return self.net(x)

    return PINNRegressor()


def _physics_residual_loss(model, x_batch, priors_cfg):
    """Compute the PDE residual loss via autograd.

    Residual: dP/dt - [κ·λ(t)·P_avg − P/τ]
    Chain rule recovers dP/dt from dP/dhour_sin and dP/dhour_cos.
    """
    import torch

    L = model.lookback
    x = x_batch.detach().clone().requires_grad_(True)
    p_pred = model(x)  # [B, 1]

    # First-order gradient of p_pred w.r.t. inputs
    grads = torch.autograd.grad(
        outputs=p_pred.sum(),
        inputs=x,
        create_graph=True,
    )[0]  # [B, in_dim]

    dP_dhsin = grads[:, L]
    dP_dhcos = grads[:, L + 1]

    hour_sin = x[:, L]
    hour_cos = x[:, L + 1]
    # Reconstruct hour ∈ [0, 24) from sin/cos
    hour = torch.atan2(hour_sin, hour_cos) * (24.0 / (2.0 * np.pi))
    hour = torch.where(hour < 0, hour + 24.0, hour)

    # Chain rule: d(hour_sin)/d(hour) = (2π/24)·cos(2π·hour/24) = (2π/24)·hour_cos
    #             d(hour_cos)/d(hour) = -(2π/24)·sin(2π·hour/24) = -(2π/24)·hour_sin
    chain = 2.0 * np.pi / 24.0
    dP_dt = dP_dhsin * chain * hour_cos + dP_dhcos * (-chain) * hour_sin  # [B]

    lambda_t = arrival_intensity_pdf_torch(hour, priors_cfg)  # [B]
    p = p_pred.squeeze(-1)  # [B]
    rhs = model.kappa * lambda_t * model.p_session_avg - p / model.tau_hours

    residual = dP_dt - rhs
    return residual.pow(2).mean()


def _train_pinn(train, val, cfg_pinn, *, budget_sec, seed, lookback,
                priors_cfg):
    """Train the Raissi-style PINN: data MSE + physics residual + soft bounds."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    X_tr, y_tr, c_tr = train
    X_v, y_v, c_v = val

    Xt = torch.tensor(X_tr)
    yt = torch.tensor(y_tr).unsqueeze(1)
    ct = torch.tensor(c_tr).unsqueeze(1)
    Xv = torch.tensor(X_v)
    yv = torch.tensor(y_v).unsqueeze(1)

    in_dim = X_tr.shape[1]
    p_session_avg = compute_p_session_avg_kw(priors_cfg)
    tau_hours = compute_tau_hours(priors_cfg)

    model = _make_pinn_model(in_dim, cfg_pinn, lookback, p_session_avg, tau_hours)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    lam_phys = cfg_pinn["pinn_lambda_physics"]
    lam_bd = 0.01

    batch_size = 64
    n = X_tr.shape[0]
    start = time.time()
    val_losses = []
    epoch = 0
    last_heartbeat = start
    # Heartbeat to file via logging — survives terminal pause.
    pinn_log = logging.getLogger("pinn")
    pinn_log.info("PINN training start: budget=%.0fs, seed=%d, batches/epoch=%d",
                  budget_sec, seed, max(1, n // batch_size))
    while time.time() - start < budget_sec:
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = Xt[idx]
            yb = yt[idx]
            cb = ct[idx]
            opt.zero_grad()
            pred = model(xb)
            data_loss = mse(pred, yb)
            phys_loss = _physics_residual_loss(model, xb, priors_cfg)
            neg_pen = torch.relu(-pred).pow(2).mean()
            over_pen = torch.relu(pred - cb).pow(2).mean()
            bound_loss = neg_pen + over_pen
            loss = data_loss + lam_phys * phys_loss + lam_bd * bound_loss
            loss.backward()
            opt.step()
            if time.time() - start >= budget_sec:
                break
        epoch += 1
        with torch.no_grad():
            pred_v = model(Xv)
            val_losses.append(mse(pred_v, yv).item())
        # Heartbeat every 30 sec to the log file (NOT to stdout) so terminal
        # pauses cannot block us.
        now = time.time()
        if now - last_heartbeat >= 30.0:
            pinn_log.info(
                "  ...PINN seed=%d epoch=%d elapsed=%.0fs/%.0fs val_loss=%.4f kappa=%.2f",
                seed, epoch, now - start, budget_sec,
                val_losses[-1], float(model.kappa.detach().item()),
            )
            last_heartbeat = now
        if time.time() - start >= budget_sec:
            break
    train_time = time.time() - start
    pinn_log.info("PINN training done: seed=%d epochs=%d wall=%.1fs final_val=%.4f",
                  seed, epoch, train_time, val_losses[-1] if val_losses else float("nan"))

    if len(val_losses) >= 5:
        tail = val_losses[-max(1, len(val_losses) // 5) :]
        head = val_losses[: max(1, len(val_losses) // 5)]
        converged = (np.mean(tail) <= np.mean(head)) and (
            np.std(tail) < 0.2 * np.mean(tail) + 1e-6
        )
    else:
        converged = False

    param_count = sum(p.numel() for p in model.parameters())
    learned_kappa = float(model.kappa.detach().item())
    return model, train_time, converged, param_count, learned_kappa


def _eval_model_torch(model, X_test, y_test, caps_test, is_pinn: bool):
    import torch

    Xt = torch.tensor(X_test)
    with torch.no_grad():
        start = time.time()
        pred = model(Xt).squeeze(1).numpy()
        infer_total = time.time() - start
    n = X_test.shape[0]
    inference_ms = 1000.0 * infer_total / max(1, n)
    err = pred - y_test
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = _bounded_mape_pct(err, y_test)
    if is_pinn:
        violations = int(((pred < 0) | (pred > caps_test)).sum())
    else:
        violations = 0
    return {
        "mae_kw": mae,
        "rmse_kw": rmse,
        "mape_pct": mape,
        "inference_latency_ms_per_sample": inference_ms,
        "physics_constraint_violations": violations,
        "predictions": pred,
    }


def _train_lstm(train, val, cfg_pinn, *, budget_sec, seed, lookback):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    X_tr, y_tr, _ = train
    X_v, y_v, _ = val

    extra = X_tr.shape[1] - lookback

    class LSTMReg(nn.Module):
        def __init__(self, hidden, layers, extra_dim):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True)
            self.fc = nn.Linear(hidden + extra_dim, 1)

        def forward(self, x):
            seq = x[:, :lookback].unsqueeze(-1)
            ext = x[:, lookback:]
            out, _ = self.lstm(seq)
            last = out[:, -1, :]
            return self.fc(torch.cat([last, ext], dim=1))

    model = LSTMReg(cfg_pinn["lstm_hidden_units"], cfg_pinn["lstm_layers"], extra)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    Xt = torch.tensor(X_tr)
    yt = torch.tensor(y_tr).unsqueeze(1)
    batch_size = 64
    n = X_tr.shape[0]
    start = time.time()
    last_heartbeat = start
    lstm_log = logging.getLogger("lstm")
    lstm_log.info("LSTM training start: budget=%.0fs, seed=%d", budget_sec, seed)
    epoch = 0
    while time.time() - start < budget_sec:
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = mse(pred, yt[idx])
            loss.backward()
            opt.step()
            if time.time() - start >= budget_sec:
                break
        epoch += 1
        now = time.time()
        if now - last_heartbeat >= 30.0:
            lstm_log.info("  ...LSTM seed=%d epoch=%d elapsed=%.0fs/%.0fs",
                          seed, epoch, now - start, budget_sec)
            last_heartbeat = now
    train_time = time.time() - start
    param_count = sum(p.numel() for p in model.parameters())
    lstm_log.info("LSTM training done: seed=%d epochs=%d wall=%.1fs",
                  seed, epoch, train_time)
    return model, train_time, True, param_count


def _train_xgb(train, val, cfg_pinn, *, budget_sec, seed):
    import xgboost as xgb

    X_tr, y_tr, _ = train
    X_v, y_v, _ = val
    n_est = cfg_pinn["xgb_n_estimators"]
    start = time.time()
    model = xgb.XGBRegressor(
        n_estimators=n_est,
        max_depth=5,
        learning_rate=0.08,
        random_state=seed,
        n_jobs=2,
        early_stopping_rounds=20,
        verbosity=0,
    )
    if X_v.shape[0] > 0:
        model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    else:
        model.fit(X_tr, y_tr, verbose=False)
    train_time = time.time() - start
    param_count = -1
    return model, train_time, True, param_count


def _bounded_mape_pct(err: np.ndarray, y_true: np.ndarray) -> float:
    """MAPE with a 1 kW denominator floor to avoid zero-load explosions."""
    denom = np.maximum(1.0, np.abs(y_true))
    return float(np.mean(np.abs(err) / denom) * 100.0)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def run_cluster3(
    seed_root: int,
    cfg_pinn: dict,
    cfg_lb_template: dict,
    priors_path: Path,
    *,
    extended_seed_count: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Run the full Cluster 3 benchmark with Raissi-style PINN."""
    train, val, test = _build_dataset(seed_root, cfg_pinn, cfg_lb_template, priors_path)
    X_test, y_test, c_test = test

    # Load priors for the physics residual
    with open(priors_path, encoding="utf-8") as f:
        priors_cfg = yaml.safe_load(f)

    rows = []
    extras = {"y_test": y_test, "preds": {}, "learned_kappa": {}}
    seeds_full = list(cfg_pinn["model_init_seeds"]) or [0]

    # PINN fixed_budget × all seeds
    for s in seeds_full:
        model, t_train, conv, pc, kappa = _train_pinn(
            train, val, cfg_pinn,
            budget_sec=cfg_pinn["pinn_fixed_budget_seconds"],
            seed=s, lookback=cfg_pinn["lookback_hours"],
            priors_cfg=priors_cfg,
        )
        ev = _eval_model_torch(model, X_test, y_test, c_test, is_pinn=True)
        rows.append({
            "model": "pinn", "regime": "fixed_budget", "seed": s,
            "mae_kw": ev["mae_kw"], "rmse_kw": ev["rmse_kw"],
            "mape_pct": ev["mape_pct"],
            "inference_latency_ms_per_sample": ev["inference_latency_ms_per_sample"],
            "train_time_sec": t_train, "parameter_count": pc,
            "physics_constraint_violations": ev["physics_constraint_violations"],
            "learned_kappa_sessions_per_day": kappa,
            "converged": conv,
        })
        if s == seeds_full[0]:
            extras["preds"]["pinn_fixed"] = ev["predictions"]
        extras["learned_kappa"][f"fixed_seed{s}"] = kappa

    # PINN extended × extended_seed_count seeds
    ext_seeds = seeds_full[:extended_seed_count]
    for s in ext_seeds:
        model, t_train, conv, pc, kappa = _train_pinn(
            train, val, cfg_pinn,
            budget_sec=cfg_pinn["pinn_extended_seconds"],
            seed=s, lookback=cfg_pinn["lookback_hours"],
            priors_cfg=priors_cfg,
        )
        ev = _eval_model_torch(model, X_test, y_test, c_test, is_pinn=True)
        rows.append({
            "model": "pinn", "regime": "extended", "seed": s,
            "mae_kw": ev["mae_kw"], "rmse_kw": ev["rmse_kw"],
            "mape_pct": ev["mape_pct"],
            "inference_latency_ms_per_sample": ev["inference_latency_ms_per_sample"],
            "train_time_sec": t_train, "parameter_count": pc,
            "physics_constraint_violations": ev["physics_constraint_violations"],
            "learned_kappa_sessions_per_day": kappa,
            "converged": conv,
        })
        if s == ext_seeds[0]:
            extras["preds"]["pinn_extended"] = ev["predictions"]
        extras["learned_kappa"][f"extended_seed{s}"] = kappa

    # LSTM
    for s in seeds_full:
        model, t_train, conv, pc = _train_lstm(
            train, val, cfg_pinn,
            budget_sec=cfg_pinn["lstm_budget_seconds"],
            seed=s, lookback=cfg_pinn["lookback_hours"],
        )
        ev = _eval_model_torch(model, X_test, y_test, c_test, is_pinn=False)
        rows.append({
            "model": "lstm", "regime": "fixed_budget", "seed": s,
            "mae_kw": ev["mae_kw"], "rmse_kw": ev["rmse_kw"],
            "mape_pct": ev["mape_pct"],
            "inference_latency_ms_per_sample": ev["inference_latency_ms_per_sample"],
            "train_time_sec": t_train, "parameter_count": pc,
            "physics_constraint_violations": 0,
            "learned_kappa_sessions_per_day": float("nan"),
            "converged": conv,
        })
        if s == seeds_full[0]:
            extras["preds"]["lstm"] = ev["predictions"]

    # XGBoost
    for s in seeds_full:
        model, t_train, conv, pc = _train_xgb(
            train, val, cfg_pinn,
            budget_sec=cfg_pinn["xgb_budget_seconds"], seed=s,
        )
        start = time.time()
        pred = model.predict(X_test)
        infer_total = time.time() - start
        err = pred - y_test
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        mape = _bounded_mape_pct(err, y_test)
        rows.append({
            "model": "xgboost", "regime": "fixed_budget", "seed": s,
            "mae_kw": mae, "rmse_kw": rmse, "mape_pct": mape,
            "inference_latency_ms_per_sample": 1000.0 * infer_total / max(1, X_test.shape[0]),
            "train_time_sec": t_train, "parameter_count": pc,
            "physics_constraint_violations": 0,
            "learned_kappa_sessions_per_day": float("nan"),
            "converged": True,
        })
        if s == seeds_full[0]:
            extras["preds"]["xgboost"] = pred

    return pd.DataFrame(rows), extras
