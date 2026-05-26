"""Layer 2 — Behavioural priors sampling.

Reads config/priors.yaml and provides functions to sample arrival times,
session durations, energy demands, and charging powers. Outputs are
simulation-derived approximations of ElaadNL published-literature shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml


@dataclass
class Priors:
    raw: dict

    @classmethod
    def load(cls, path: Path) -> "Priors":
        with open(path, "r", encoding="utf-8") as f:
            return cls(raw=yaml.safe_load(f))

    # --- arrival time of day (hours, 0-24) -------------------------------------
    def sample_arrival_hour(self, rng: np.random.Generator, n: int) -> np.ndarray:
        a = self.raw["arrival_time"]
        m = a["morning_peak"]
        e = a["evening_peak"]
        u_w = a["off_peak_uniform_weight"]
        w_m, w_e = m["weight"], e["weight"]
        choices = rng.choice(
            ["m", "e", "u"], size=n, p=[w_m, w_e, u_w]
        )
        out = np.empty(n)
        morn_mask = choices == "m"
        eve_mask = choices == "e"
        uni_mask = choices == "u"
        out[morn_mask] = rng.normal(m["mean_hour"], m["std_hour"], size=morn_mask.sum())
        out[eve_mask] = rng.normal(e["mean_hour"], e["std_hour"], size=eve_mask.sum())
        out[uni_mask] = rng.uniform(0, 24, size=uni_mask.sum())
        # wrap into [0,24)
        return np.mod(out, 24.0)

    # --- session duration in hours ---------------------------------------------
    def sample_duration_hours(self, rng: np.random.Generator, n: int) -> np.ndarray:
        d = self.raw["session_duration_hours"]
        x = rng.lognormal(mean=d["mu_log"], sigma=d["sigma_log"], size=n)
        return np.clip(x, d["min_clip"], d["max_clip"])

    # --- energy demand per session in kWh --------------------------------------
    def sample_energy_kwh(self, rng: np.random.Generator, n: int) -> np.ndarray:
        e = self.raw["energy_demand_kwh"]
        x = rng.lognormal(mean=e["mu_log"], sigma=e["sigma_log"], size=n)
        return np.clip(x, e["min_clip"], e["max_clip"])

    # --- charging power kW (mixture) -------------------------------------------
    def sample_charging_power_kw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        m = self.raw["charging_power_mix"]
        o = m["one_phase"]
        t = m["three_phase"]
        d = m["dc_fast"]
        total = o["weight"] + t["weight"] + d["weight"]
        ps = [o["weight"] / total, t["weight"] / total, d["weight"] / total]
        choice = rng.choice(["1p", "3p", "dc"], size=n, p=ps)
        out = np.empty(n)
        for label, cfg in [("1p", o), ("3p", t), ("dc", d)]:
            mask = choice == label
            out[mask] = rng.uniform(cfg["min_kw"], cfg["max_kw"], size=mask.sum())
        return out

    # --- day-of-week multiplier (Mon=0 ... Sun=6) -----------------------------
    def day_of_week_multiplier(self, dow_index: int) -> float:
        names = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        return float(self.raw["day_of_week_multiplier"][names[dow_index % 7]])
