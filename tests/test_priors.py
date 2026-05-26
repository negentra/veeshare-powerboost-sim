"""Test prior sampling."""

from pathlib import Path

import numpy as np
import pytest

from src.priors import Priors

ROOT = Path(__file__).resolve().parent.parent


def test_priors_load():
    priors = Priors.load(ROOT / "config" / "priors.yaml")
    rng = np.random.default_rng(42)
    arrivals = priors.sample_arrival_hour(rng, 1000)
    assert arrivals.shape == (1000,)
    assert (arrivals >= 0).all() and (arrivals < 24).all()

    durations = priors.sample_duration_hours(rng, 1000)
    assert (durations > 0).all() and (durations <= 14.0).all()

    energies = priors.sample_energy_kwh(rng, 1000)
    assert (energies > 0).all() and (energies <= 80.0).all()

    powers = priors.sample_charging_power_kw(rng, 1000)
    assert (powers >= 3.7).all() and (powers <= 150.0).all()


def test_day_of_week():
    priors = Priors.load(ROOT / "config" / "priors.yaml")
    assert priors.day_of_week_multiplier(0) == 1.0  # Monday
    assert priors.day_of_week_multiplier(5) == 0.7  # Saturday
    assert priors.day_of_week_multiplier(6) == 0.5  # Sunday
