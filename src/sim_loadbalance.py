"""Cluster 2 — Real-time load balancing.

Simulates a representative week of charging sessions across the topology,
under two regimes:
  - baseline: uncoordinated, sessions start immediately at full power
  - coordinated: greedy peak-shaving heuristic that shifts charging within
    each session's [arrival, departure] window to flatten transformer load

For each (penetration_pct, regime, repetition) cell, returns weekly
aggregate metrics: peak load, peak-to-avg ratio, transformer overload
hours, energy delivered, unmet demand, average charging cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.market import generate_weekly_prices
from src.priors import Priors
from src.topology import build_topology
from src.utils.seed import derive_seed


# A-09: Average daily charging probability per vehicle in the eligible
# population, used to determine number of sessions per week.
DAILY_CHARGE_PROB = 0.30

# A-10: Substation overload threshold = transformer kVA * 0.95 (5% buffer).
OVERLOAD_THRESHOLD = 0.95

# A-11: 15-min buckets per week (168h * 4 = 672 buckets)
BUCKETS_PER_HOUR = 4


@dataclass
class Session:
    station_id: str
    substation_id: str
    max_station_kw: float
    arrival_bucket: int
    departure_bucket: int
    requested_kwh: float
    requested_power_kw: float


def _generate_sessions(
    seed_root: int,
    rep: int,
    penetration_pct: int,
    cfg: dict,
    priors: Priors,
    stations,
) -> List[Session]:
    """Generate a week of charging sessions for the eligible population."""
    rng = np.random.default_rng(
        derive_seed(seed_root, f"loadbal::rep{rep}::pen{penetration_pct}")
    )
    eligible = int(cfg["vehicle_population"] * penetration_pct / 100.0)
    n_sessions_expected = int(eligible * 7 * DAILY_CHARGE_PROB)
    if n_sessions_expected == 0:
        return []

    week_hours = cfg["week_hours"]
    week_buckets = week_hours * BUCKETS_PER_HOUR

    # Sample arrivals across week
    days = rng.integers(0, max(1, week_hours // 24), size=n_sessions_expected)
    hours = priors.sample_arrival_hour(rng, n_sessions_expected)
    # DoW multiplier: rejection sampling to weight by day_of_week_multiplier
    dow_mult = np.array([priors.day_of_week_multiplier(int(d % 7)) for d in days])
    keep = rng.random(n_sessions_expected) < dow_mult
    days = days[keep]
    hours = hours[keep]
    n = days.size

    arrival_hour_total = days * 24 + hours
    arrival_buckets = (arrival_hour_total * BUCKETS_PER_HOUR).astype(int)
    arrival_buckets = np.clip(arrival_buckets, 0, week_buckets - 1)

    durations_h = priors.sample_duration_hours(rng, n)
    duration_buckets = np.maximum(
        1, (durations_h * BUCKETS_PER_HOUR).astype(int)
    )
    departure_buckets = np.minimum(
        week_buckets - 1, arrival_buckets + duration_buckets
    )

    energies = priors.sample_energy_kwh(rng, n)
    requested_powers = priors.sample_charging_power_kw(rng, n)

    # Assign each session to a random station (uniform among 50)
    station_idx = rng.integers(0, len(stations), size=n)

    sessions: List[Session] = []
    for i in range(n):
        st = stations[station_idx[i]]
        sessions.append(
            Session(
                station_id=st.station_id,
                substation_id=st.substation_id,
                max_station_kw=st.max_power_kw,
                arrival_bucket=int(arrival_buckets[i]),
                departure_bucket=int(departure_buckets[i]),
                requested_kwh=float(energies[i]),
                requested_power_kw=float(min(requested_powers[i], st.max_power_kw)),
            )
        )
    return sessions


def _simulate_regime(
    sessions: List[Session],
    substations,
    week_buckets: int,
    regime: str,
) -> Dict[str, float]:
    """Run a given regime, return aggregate metrics for one cell.

    Returns kW per bucket per substation, then derives peak_load_kw etc.
    """
    bucket_hours = 1.0 / BUCKETS_PER_HOUR
    sub_idx = {s.substation_id: i for i, s in enumerate(substations)}
    sub_capacities_kw = np.array([s.capacity_kva * 0.95 for s in substations])
    load = np.zeros((len(substations), week_buckets))
    energy_delivered = 0.0
    unmet = 0.0

    if regime == "baseline":
        for s in sessions:
            duration = s.departure_bucket - s.arrival_bucket
            if duration <= 0:
                unmet += s.requested_kwh
                continue
            # Time to deliver requested energy at requested power
            ideal_buckets = int(
                np.ceil(s.requested_kwh / (s.requested_power_kw * bucket_hours))
            )
            usable_buckets = min(ideal_buckets, duration)
            kwh_delivered = (
                usable_buckets * s.requested_power_kw * bucket_hours
            )
            kwh_delivered = min(kwh_delivered, s.requested_kwh)
            unmet += max(0.0, s.requested_kwh - kwh_delivered)
            energy_delivered += kwh_delivered
            si = sub_idx[s.substation_id]
            load[si, s.arrival_bucket : s.arrival_bucket + usable_buckets] += (
                s.requested_power_kw
            )

    elif regime == "coordinated":
        # Greedy peak-shaving per substation: process sessions in arrival order,
        # for each session distribute its energy across its window, preferring
        # buckets where substation load is currently lowest, up to per-session
        # max power.
        # Sort sessions by arrival for stability.
        sessions_sorted = sorted(sessions, key=lambda s: s.arrival_bucket)
        for s in sessions_sorted:
            duration = s.departure_bucket - s.arrival_bucket
            if duration <= 0:
                unmet += s.requested_kwh
                continue
            si = sub_idx[s.substation_id]
            window = slice(s.arrival_bucket, s.departure_bucket)
            remaining_kwh = s.requested_kwh
            max_per_bucket_kwh = s.requested_power_kw * bucket_hours
            # Sort window buckets by current substation load (ascending)
            window_buckets = np.arange(s.arrival_bucket, s.departure_bucket)
            order = window_buckets[np.argsort(load[si, window])]
            for b in order:
                if remaining_kwh <= 0:
                    break
                headroom_kw = sub_capacities_kw[si] - load[si, b]
                allowed_kw = min(s.requested_power_kw, max(0.0, headroom_kw))
                delivered_kwh = min(
                    remaining_kwh,
                    allowed_kw * bucket_hours,
                    max_per_bucket_kwh,
                )
                if delivered_kwh <= 0:
                    continue
                load[si, b] += delivered_kwh / bucket_hours
                remaining_kwh -= delivered_kwh
            delivered = s.requested_kwh - remaining_kwh
            energy_delivered += delivered
            unmet += remaining_kwh
    else:
        raise ValueError(f"Unknown regime: {regime}")

    # Aggregate metrics
    aggregate_load = load.sum(axis=0)  # total kW per bucket
    peak_load_kw = float(aggregate_load.max())
    avg_load_kw = float(aggregate_load.mean())
    peak_to_avg = peak_load_kw / max(1e-6, avg_load_kw)

    overload_buckets = (load > sub_capacities_kw[:, None]).sum()
    overload_hours = float(overload_buckets * bucket_hours)

    return {
        "peak_load_kw": peak_load_kw,
        "peak_to_avg_ratio": peak_to_avg,
        "transformer_overload_hours": overload_hours,
        "total_energy_delivered_kwh": float(energy_delivered),
        "unmet_demand_kwh": float(unmet),
        # Average charging cost left to caller (needs market prices).
        "aggregate_load_kw": aggregate_load,  # for figure 3
    }


def run_cluster2(
    seed_root: int,
    cfg: dict,
    priors_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Run the Cluster 2 sweep.

    Returns:
        df: DataFrame with the per-row CSV metrics.
        extras: dict with aggregate_load_profile for the headline 50%
            penetration case, used for fig3.
    """
    priors = Priors.load(priors_path)
    stations, substations, _G = build_topology(seed_root, station_count=cfg["station_count"], substation_count=cfg["substation_count"])
    week_buckets = cfg["week_hours"] * BUCKETS_PER_HOUR
    prices = generate_weekly_prices(seed_root, cfg["week_hours"])
    # Repeat prices per bucket so we have per-bucket EUR/MWh
    prices_per_bucket = np.repeat(prices, BUCKETS_PER_HOUR)

    rows = []
    profiles = {}  # (penetration, regime) -> aggregate_load_kw

    for pen in cfg["penetration_pcts"]:
        for rep in range(1, cfg["repetitions"] + 1):
            sessions = _generate_sessions(
                seed_root, rep, pen, cfg, priors, stations
            )
            for regime in cfg["regimes"]:
                # Re-seed deterministically: same sessions for both regimes
                # so we measure the regime effect cleanly.
                res = _simulate_regime(
                    sessions, substations, week_buckets, regime
                )
                # Compute weighted-average cost: cost = sum(load_kwh * price)
                load_kwh_per_bucket = res["aggregate_load_kw"] / BUCKETS_PER_HOUR
                total_energy = max(1e-6, load_kwh_per_bucket.sum())
                weighted_price_eur_per_mwh = (
                    (load_kwh_per_bucket * prices_per_bucket[:week_buckets]).sum()
                    / total_energy
                )
                avg_cost_eur_per_kwh = weighted_price_eur_per_mwh / 1000.0

                rows.append(
                    {
                        "penetration_pct": pen,
                        "regime": regime,
                        "repetition": rep,
                        "peak_load_kw": res["peak_load_kw"],
                        "peak_to_avg_ratio": res["peak_to_avg_ratio"],
                        "transformer_overload_hours": res[
                            "transformer_overload_hours"
                        ],
                        "total_energy_delivered_kwh": res[
                            "total_energy_delivered_kwh"
                        ],
                        "unmet_demand_kwh": res["unmet_demand_kwh"],
                        "avg_charging_cost_eur_per_kwh": float(
                            avg_cost_eur_per_kwh
                        ),
                    }
                )
                # Keep first rep's profile per (pen, regime) for figure 3
                if rep == 1:
                    profiles[(pen, regime)] = res["aggregate_load_kw"]

    df = pd.DataFrame(rows)
    return df, profiles
