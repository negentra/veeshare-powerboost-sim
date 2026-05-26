"""P5 stretch — Counterfactual TR vs liberalised regulatory regimes.

Re-runs Cluster 2's load-balancing simulation under two contrasting
regulatory regimes derived from Turkish Şarj Hizmeti Yönetmeliği (2/4/2022)
versus a liberalised reference scenario.

This output feeds the academic paper (Energy Policy submission with
Münür Sacit Herdem), not the audit deliverable. It demonstrates the
welfare cost of restrictive regulation in P2P EV charging.

Regulatory parameters modelled:
  regime_TR:
    - Only licensed CPOs operate stations (12 of 50 active hosts)
    - Floor/ceiling price intervention (±20% around mean)
    - Equal-treatment obligation: no preferential pool pricing
  regime_liberalised:
    - All stations available as P2P hosts (50 of 50 active)
    - No price intervention
    - Hosts can offer preferential pricing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.market import generate_weekly_prices
from src.priors import Priors
from src.sim_loadbalance import (
    BUCKETS_PER_HOUR,
    _generate_sessions,
    _simulate_regime,
)
from src.topology import build_topology
from src.utils.seed import derive_seed


log = logging.getLogger("counterfactual")


def _restrict_active_hosts(
    stations,
    active_count: int,
    seed_root: int,
) -> list:
    """Return a subset of stations that are active under a regulatory regime.

    Selection is deterministic for a given seed: under a licensed-CPO model,
    the most powerful stations (DC fast + ultra) are kept first, then AC
    in deterministic order.
    """
    rng = np.random.default_rng(derive_seed(seed_root, f"cf_hosts_{active_count}"))
    # Sort: keep highest-power stations preferentially (typical CPO portfolio)
    sorted_by_power = sorted(stations, key=lambda s: -s.max_power_kw)
    # Then a deterministic shuffle within ties via the rng
    indices = list(range(len(sorted_by_power)))
    rng.shuffle(indices)
    # Stable: take the first `active_count` sorted by power, breaking ties
    # with the shuffled order
    chosen = sorted(
        sorted_by_power,
        key=lambda s: (-s.max_power_kw, indices[sorted_by_power.index(s)]),
    )[:active_count]
    return chosen


def _filter_sessions_to_active_hosts(sessions, active_station_ids):
    """Drop sessions that landed on now-inactive stations.

    Under the TR regime, P2P hosts (most stations) are not licensed CPOs,
    so sessions allocated to them are 'unserved' from the regulated system's
    perspective. Modelling choice: those sessions become unmet demand.
    """
    active_set = set(active_station_ids)
    return [s for s in sessions if s.station_id in active_set]


def _compute_welfare_proxy(
    energy_delivered_kwh: float,
    unmet_demand_kwh: float,
    avg_cost_eur_per_kwh: float,
    *,
    consumer_willingness_pay_eur_per_kwh: float = 0.30,
    unmet_penalty_eur_per_kwh: float = 0.50,
) -> float:
    """Welfare proxy = consumer surplus − unmet-demand penalty.

    Consumer surplus per kWh = WTP − price.
    Unmet demand modelled as a per-kWh welfare loss (penalty).

    Returns EUR/week.
    """
    consumer_surplus = energy_delivered_kwh * max(
        0.0, consumer_willingness_pay_eur_per_kwh - avg_cost_eur_per_kwh
    )
    unmet_loss = unmet_demand_kwh * unmet_penalty_eur_per_kwh
    return consumer_surplus - unmet_loss


def run_counterfactual(
    seed_root: int,
    cfg_cluster2: dict,
    cfg_p5: dict,
    priors_path: Path,
) -> pd.DataFrame:
    """Run the P5 counterfactual analysis.

    Holds session arrivals constant (same seed/realisation); varies only
    which subset of stations is 'regulated as active' and the pricing
    regime. The penetration is fixed at the headline value (50%) to
    isolate the regulatory effect.
    """
    log.info("P5 counterfactual: building shared session realisation...")
    priors = Priors.load(priors_path)
    stations, substations, _ = build_topology(
        seed_root,
        station_count=cfg_cluster2["station_count"],
        substation_count=cfg_cluster2["substation_count"],
    )

    # Use the headline penetration to keep apples-to-apples with audit results
    headline_pen = cfg_cluster2.get("headline_penetration", 50)
    cfg_lb = dict(cfg_cluster2)
    cfg_lb["penetration_pcts"] = [headline_pen]

    # Generate one realisation of weekly sessions on the full topology
    sessions_full = _generate_sessions(
        seed_root, rep=1, penetration_pct=headline_pen,
        cfg=cfg_lb, priors=priors, stations=stations,
    )
    week_buckets = cfg_lb["week_hours"] * BUCKETS_PER_HOUR
    log.info("  generated %d sessions across %d stations",
             len(sessions_full), len(stations))

    prices = generate_weekly_prices(seed_root, cfg_lb["week_hours"])
    prices_per_bucket = np.repeat(prices, BUCKETS_PER_HOUR)

    rows: List[Dict] = []

    # ---- TR regime ---------------------------------------------------------
    tr_cfg = cfg_p5["regime_TR"]
    tr_active = _restrict_active_hosts(stations, tr_cfg["active_hosts"], seed_root)
    tr_active_ids = {s.station_id for s in tr_active}
    tr_sessions = _filter_sessions_to_active_hosts(sessions_full, tr_active_ids)
    # Sessions that did not find a CPO station become unmet demand
    tr_dropped_kwh = sum(
        s.requested_kwh for s in sessions_full if s.station_id not in tr_active_ids
    )
    log.info("  TR regime: %d/%d active stations, %d sessions served, %.0f kWh dropped",
             len(tr_active), len(stations), len(tr_sessions), tr_dropped_kwh)
    res_tr = _simulate_regime(tr_sessions, substations, week_buckets, "coordinated")
    res_tr["unmet_demand_kwh"] += tr_dropped_kwh  # add regulatory exclusion

    energy_tr = float(res_tr["total_energy_delivered_kwh"])
    unmet_tr = float(res_tr["unmet_demand_kwh"])
    load_kwh_per_bucket_tr = res_tr["aggregate_load_kw"] / BUCKETS_PER_HOUR
    total_energy_tr = max(1e-6, load_kwh_per_bucket_tr.sum())
    weighted_price_tr_mwh = (
        load_kwh_per_bucket_tr * prices_per_bucket[:week_buckets]
    ).sum() / total_energy_tr
    avg_cost_tr = float(weighted_price_tr_mwh) / 1000.0
    # Apply price ceiling/floor effect: TR regime imposes ±20% band; for
    # welfare proxy, this caps the price the operator can charge.
    floor_mult = 1.0 + tr_cfg["price_floor_pct"] / 100.0
    ceil_mult = 1.0 + tr_cfg["price_ceiling_pct"] / 100.0
    mean_price = float(prices.mean()) / 1000.0
    avg_cost_tr_capped = float(np.clip(
        avg_cost_tr, mean_price * floor_mult, mean_price * ceil_mult
    ))
    welfare_tr = _compute_welfare_proxy(
        energy_tr, unmet_tr, avg_cost_tr_capped,
    )

    rows.append({
        "regime": "TR",
        "active_hosts": len(tr_active),
        "peak_load_kw": float(res_tr["peak_load_kw"]),
        "peak_load_reduction_pct": float("nan"),  # filled after both regimes
        "transformer_overload_hours": float(res_tr["transformer_overload_hours"]),
        "total_energy_delivered_kwh": energy_tr,
        "unmet_demand_kwh": unmet_tr,
        "avg_charging_cost_eur_per_kwh": avg_cost_tr_capped,
        "total_welfare_proxy_eur_per_week": welfare_tr,
    })

    # ---- Liberalised regime -------------------------------------------------
    lib_cfg = cfg_p5["regime_liberalised"]
    lib_active = _restrict_active_hosts(stations, lib_cfg["active_hosts"], seed_root)
    lib_active_ids = {s.station_id for s in lib_active}
    lib_sessions = _filter_sessions_to_active_hosts(sessions_full, lib_active_ids)
    log.info("  Liberalised regime: %d/%d active stations, %d sessions served",
             len(lib_active), len(stations), len(lib_sessions))
    res_lib = _simulate_regime(lib_sessions, substations, week_buckets, "coordinated")

    energy_lib = float(res_lib["total_energy_delivered_kwh"])
    unmet_lib = float(res_lib["unmet_demand_kwh"])
    load_kwh_per_bucket_lib = res_lib["aggregate_load_kw"] / BUCKETS_PER_HOUR
    total_energy_lib = max(1e-6, load_kwh_per_bucket_lib.sum())
    weighted_price_lib_mwh = (
        load_kwh_per_bucket_lib * prices_per_bucket[:week_buckets]
    ).sum() / total_energy_lib
    avg_cost_lib = float(weighted_price_lib_mwh) / 1000.0
    welfare_lib = _compute_welfare_proxy(
        energy_lib, unmet_lib, avg_cost_lib,
    )

    rows.append({
        "regime": "liberalised",
        "active_hosts": len(lib_active),
        "peak_load_kw": float(res_lib["peak_load_kw"]),
        "peak_load_reduction_pct": float("nan"),
        "transformer_overload_hours": float(res_lib["transformer_overload_hours"]),
        "total_energy_delivered_kwh": energy_lib,
        "unmet_demand_kwh": unmet_lib,
        "avg_charging_cost_eur_per_kwh": avg_cost_lib,
        "total_welfare_proxy_eur_per_week": welfare_lib,
    })

    df = pd.DataFrame(rows)
    # Fill peak_load_reduction_pct relative to a hypothetical no-coordination baseline
    # — but here we compare regimes against each other.
    df["welfare_delta_vs_TR"] = df["total_welfare_proxy_eur_per_week"] - welfare_tr

    log.info("P5 counterfactual: TR welfare=%.0f EUR/week, Liberalised=%.0f EUR/week, delta=%.0f",
             welfare_tr, welfare_lib, welfare_lib - welfare_tr)

    return df
