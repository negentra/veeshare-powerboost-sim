"""Layer 1 — Synthetic Lisbon network topology.

Generates a synthetic-but-plausible subset of the Lisbon MOBI.E network:
50 charging stations across 6 substations, plus one DSO/aggregator node.

NOTE: All coordinates and transformer capacities are SYNTHETIC. They are
NOT fetched from MOBI.E, E-REDES, or any real registry. Auditor-visible
provenance: this module produces order-of-magnitude plausible structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import networkx as nx
import numpy as np

from src.utils.seed import module_rng

# --- A-01: Lisbon bounding box (synthetic; lat/lon order of magnitude only)
LAT_MIN, LAT_MAX = 38.70, 38.78
LON_MIN, LON_MAX = -9.20, -9.10

# --- A-02: Transformer capacities (kVA) drawn from typical MV/LV distribution
TRANSFORMER_CAPACITIES_KVA = [250, 400, 400, 630, 630, 400]

# --- A-03: Station mix (50 stations total)
STATION_MIX = [
    # (count, type, min_kw, max_kw)
    (40, "AC", 7.0, 22.0),
    (8, "DC_FAST", 50.0, 150.0),
    (2, "DC_ULTRA", 150.0, 350.0),
]

DSO_NODE_ID = "DSO"


@dataclass
class Station:
    station_id: str
    substation_id: str
    lat: float
    lon: float
    type: str
    max_power_kw: float


@dataclass
class Substation:
    substation_id: str
    capacity_kva: float
    lat: float
    lon: float


def build_topology(seed_root: int, station_count: int = 50, substation_count: int = 6):
    """Construct stations, substations, and a networkx graph including DSO."""
    rng = module_rng(seed_root, "topology")

    # --- substations
    sub_lats = rng.uniform(LAT_MIN, LAT_MAX, size=substation_count)
    sub_lons = rng.uniform(LON_MIN, LON_MAX, size=substation_count)
    substations: List[Substation] = []
    for i in range(substation_count):
        cap = TRANSFORMER_CAPACITIES_KVA[i % len(TRANSFORMER_CAPACITIES_KVA)]
        substations.append(
            Substation(
                substation_id=f"SS{i+1:02d}",
                capacity_kva=float(cap),
                lat=float(sub_lats[i]),
                lon=float(sub_lons[i]),
            )
        )

    # --- stations: build flat list per STATION_MIX
    expected_total = sum(c for c, *_ in STATION_MIX)
    if expected_total != station_count:
        # If user requests a non-default count, scale mix proportionally.
        scale = station_count / expected_total
        scaled_mix = [
            (max(1, int(round(c * scale))), t, lo, hi) for c, t, lo, hi in STATION_MIX
        ]
        # Adjust last entry to make totals match exactly.
        diff = station_count - sum(c for c, *_ in scaled_mix)
        c0, t0, lo0, hi0 = scaled_mix[0]
        scaled_mix[0] = (c0 + diff, t0, lo0, hi0)
        mix = scaled_mix
    else:
        mix = STATION_MIX

    stations: List[Station] = []
    s_idx = 0
    for count, stype, lo_kw, hi_kw in mix:
        for _ in range(count):
            sub = substations[s_idx % substation_count]
            jitter_lat = rng.normal(0, 0.003)
            jitter_lon = rng.normal(0, 0.003)
            stations.append(
                Station(
                    station_id=f"ST{s_idx+1:03d}",
                    substation_id=sub.substation_id,
                    lat=float(sub.lat + jitter_lat),
                    lon=float(sub.lon + jitter_lon),
                    type=stype,
                    max_power_kw=float(rng.uniform(lo_kw, hi_kw)),
                )
            )
            s_idx += 1

    # --- graph: stations <-> their substation, substations <-> DSO
    G = nx.Graph()
    G.add_node(DSO_NODE_ID, role="dso")
    for sub in substations:
        G.add_node(sub.substation_id, role="substation", capacity_kva=sub.capacity_kva)
        G.add_edge(DSO_NODE_ID, sub.substation_id)
    for st in stations:
        G.add_node(
            st.station_id,
            role="station",
            type=st.type,
            max_power_kw=st.max_power_kw,
        )
        G.add_edge(st.station_id, st.substation_id)

    return stations, substations, G
