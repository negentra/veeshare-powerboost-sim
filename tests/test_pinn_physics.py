"""Unit tests for the Raissi-style PINN physics: autograd, arrival
intensity PDF, and learnable parameter behaviour."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.sim_pinn import (
    _make_pinn_model,
    _physics_residual_loss,
    arrival_intensity_pdf_torch,
    compute_p_session_avg_kw,
    compute_tau_hours,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def priors_cfg():
    with open(ROOT / "config" / "priors.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def cfg_pinn():
    return {
        "pinn_hidden_units": 32,
        "pinn_hidden_layers": 2,
        "pinn_lambda_physics": 0.1,
    }


def test_p_session_avg_in_reasonable_range(priors_cfg):
    p = compute_p_session_avg_kw(priors_cfg)
    assert 5.0 < p < 20.0, f"Expected mean session power 5-20 kW, got {p}"


def test_tau_in_reasonable_range(priors_cfg):
    tau = compute_tau_hours(priors_cfg)
    assert 2.0 < tau < 7.0, f"Expected mean session 2-7 h, got {tau}"


def test_arrival_intensity_integrates_to_unity(priors_cfg):
    """The PDF over [0, 24) should integrate to ~1 (or close)."""
    hours = torch.linspace(0, 24, 2401)
    pdf = arrival_intensity_pdf_torch(hours, priors_cfg)
    # Trapezoidal rule
    integral = torch.trapz(pdf, hours).item()
    # Tail effects in truncated normals make this slightly less than 1
    assert 0.85 < integral < 1.15, f"PDF integrates to {integral}, expected near 1"


def test_arrival_intensity_has_two_peaks(priors_cfg):
    """PDF should peak near morning (8.5) and evening (18.5)."""
    hours = torch.linspace(0, 24, 241)
    pdf = arrival_intensity_pdf_torch(hours, priors_cfg).numpy()
    # Argmax in [6, 11] and [16, 21]
    morning_window = pdf[60:111]  # hours 6.0 to 11.0
    evening_window = pdf[160:211]  # hours 16.0 to 21.0
    midday_window = pdf[120:140]  # hours 12.0 to 14.0
    assert morning_window.max() > midday_window.max()
    assert evening_window.max() > midday_window.max()


def test_pinn_model_has_learnable_kappa(cfg_pinn, priors_cfg):
    in_dim = 29  # 24 history + 5 features
    model = _make_pinn_model(
        in_dim, cfg_pinn, lookback=24,
        p_session_avg=compute_p_session_avg_kw(priors_cfg),
        tau_hours=compute_tau_hours(priors_cfg),
    )
    # Check kappa parameter exists
    assert hasattr(model, "log_kappa")
    assert hasattr(model, "kappa")
    kappa_val = float(model.kappa.detach().item())
    assert 25.0 < kappa_val < 35.0, f"Initial kappa should be ~30, got {kappa_val}"


def test_physics_residual_uses_autograd(cfg_pinn, priors_cfg):
    """The physics-residual loss must be differentiable through autograd."""
    in_dim = 29
    model = _make_pinn_model(
        in_dim, cfg_pinn, lookback=24,
        p_session_avg=compute_p_session_avg_kw(priors_cfg),
        tau_hours=compute_tau_hours(priors_cfg),
    )
    torch.manual_seed(0)
    x = torch.randn(8, in_dim)
    # Replace hour features with valid sin/cos values for hour=10
    x[:, 24] = float(np.sin(2 * np.pi * 10 / 24))  # hour_sin
    x[:, 25] = float(np.cos(2 * np.pi * 10 / 24))  # hour_cos

    loss = _physics_residual_loss(model, x, priors_cfg)
    assert loss.requires_grad, "Physics loss must be differentiable"
    # Backward through it
    loss.backward()
    # Kappa gradient should be non-zero (physics loss depends on kappa)
    assert model.log_kappa.grad is not None
    assert abs(model.log_kappa.grad.item()) > 0, "Kappa should receive gradient signal"


def test_chain_rule_recovers_hour_derivative(priors_cfg, cfg_pinn):
    """Numerical check: chain-rule autograd dP/dhour should ~ match finite diff."""
    in_dim = 29
    torch.manual_seed(0)
    model = _make_pinn_model(
        in_dim, cfg_pinn, lookback=24,
        p_session_avg=compute_p_session_avg_kw(priors_cfg),
        tau_hours=compute_tau_hours(priors_cfg),
    )

    def features_at_hour(h):
        x = torch.zeros(in_dim)
        x[24] = float(np.sin(2 * np.pi * h / 24))
        x[25] = float(np.cos(2 * np.pi * h / 24))
        return x

    # Finite-difference reference
    h = 10.0
    eps = 0.01
    with torch.no_grad():
        f_plus = model(features_at_hour(h + eps).unsqueeze(0)).item()
        f_minus = model(features_at_hour(h - eps).unsqueeze(0)).item()
        fd_dP_dh = (f_plus - f_minus) / (2 * eps)

    # Autograd via chain rule (replicate what _physics_residual_loss does)
    x = features_at_hour(h).unsqueeze(0).requires_grad_(True)
    p = model(x).sum()
    grads = torch.autograd.grad(p, x, create_graph=False)[0][0]
    chain = 2 * np.pi / 24.0
    autograd_dP_dh = (
        grads[24] * chain * np.cos(2 * np.pi * h / 24)
        + grads[25] * (-chain) * np.sin(2 * np.pi * h / 24)
    ).item()

    # Within 1% relative for a single point
    assert abs(autograd_dP_dh - fd_dP_dh) < 0.02 * (abs(fd_dP_dh) + 1e-4), \
        f"Chain-rule autograd ({autograd_dP_dh}) ≠ finite diff ({fd_dP_dh})"
