"""Test topology construction."""

from src.topology import build_topology, DSO_NODE_ID


def test_topology_basic():
    stations, substations, G = build_topology(42)
    assert len(stations) == 50
    assert len(substations) == 6
    assert G.number_of_nodes() == 57  # 50 + 6 + 1 DSO
    assert DSO_NODE_ID in G.nodes
    # DSO connected to all substations
    assert G.degree(DSO_NODE_ID) == 6
    # Each station has exactly one substation neighbour
    for st in stations:
        neighbours = list(G.neighbors(st.station_id))
        assert len(neighbours) == 1


def test_topology_deterministic():
    s1, _, _ = build_topology(42)
    s2, _, _ = build_topology(42)
    for a, b in zip(s1, s2):
        assert a.station_id == b.station_id
        assert a.max_power_kw == b.max_power_kw
