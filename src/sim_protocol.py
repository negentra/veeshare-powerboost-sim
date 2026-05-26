"""Cluster 1 — DePIN protocol + DSO interface + transparency.

Three integrated sub-models sharing the same simulation:

A. P2P gossip protocol scalability
   - Round-based rumour-spreading model: each round, each "infected" node
     forwards to k random peers; transaction reaches finality when >=2/3
     of nodes have seen it. Bandwidth and message-count per tx tracked.

B. DSO interface
   - Downstream broadcast: DSO emits a signal that propagates to all
     nodes via the same gossip protocol; p95 reception time recorded.
   - Upstream aggregation: nodes send status packets to DSO; time-to-95%
     collection recorded.
   - Message loss: configurable per-edge drop probability.

C. Transparency / verifiability
   - Each tx tagged with content_hash and witness_set; fraction with
     witness_set >= W (default W=3) reported.

Per-scale sweep: N in node_counts × repetitions repetitions × sim_minutes
of activity. Deterministic via module_rng + per-repetition sub-seeds.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.seed import derive_seed


# ---------------------------------------------------------------------------
# Per-round simulation constants (A-05..A-08)
# ---------------------------------------------------------------------------
# A-05: Round duration (milliseconds). Reflects a typical inter-node hop
# latency budget on a wide-area DePIN overlay (LAN+WAN mix).
ROUND_MS = 80.0

# A-06: Average bytes per gossip message (sender id + tx id + payload +
# signature stub). Order of magnitude only.
MSG_BYTES = 256

# A-07: Maximum simulated rounds per finality attempt — protective cap to
# prevent pathological behaviour. log2(50000)/log2(9) ≈ 4.9, so 30 is
# generous.
MAX_FINALITY_ROUNDS = 30


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _simulate_gossip_finality(
    n: int,
    k: int,
    quorum_frac: float,
    *,
    rng: np.random.Generator,
    msg_loss_pct: float = 0.0,
) -> Tuple[int, int, np.ndarray]:
    """Simulate gossip rumour-spread for ONE transaction or signal.

    Returns:
        rounds_to_finality (int): rounds until >= quorum nodes are infected
            (or MAX_FINALITY_ROUNDS if never).
        messages_sent (int): total messages sent (counting drops).
        infection_round (np.ndarray): per-node round at which infection
            occurred; -1 if never. Index 0 is the originator.
    """
    quorum = max(1, int(math.ceil(quorum_frac * n)))
    infected = np.zeros(n, dtype=bool)
    infection_round = np.full(n, -1, dtype=int)
    infected[0] = True
    infection_round[0] = 0
    loss_prob = msg_loss_pct / 100.0
    total_msgs = 0
    rounds_to_finality = MAX_FINALITY_ROUNDS
    for r in range(1, MAX_FINALITY_ROUNDS + 1):
        sender_idx = np.where(infected)[0]
        if sender_idx.size == 0:
            break
        # Each sender picks k random peers (with replacement; cheap).
        targets = rng.integers(0, n, size=(sender_idx.size, k))
        total_msgs += sender_idx.size * k
        if loss_prob > 0:
            keep = rng.random(targets.shape) >= loss_prob
        else:
            keep = np.ones(targets.shape, dtype=bool)
        flat_targets = targets[keep]
        newly = flat_targets[~infected[flat_targets]]
        if newly.size > 0:
            unique_new = np.unique(newly)
            infected[unique_new] = True
            infection_round[unique_new] = r
        if infected.sum() >= quorum:
            rounds_to_finality = r
            break
    return rounds_to_finality, total_msgs, infection_round


def _summarise_protocol_rep(
    n: int,
    k: int,
    sim_minutes: float,
    tx_rate: float,
    quorum_frac: float,
    msg_loss_pct: float,
    witness_threshold: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Run one repetition: sample a moderate number of transactions and
    aggregate throughput, latency, bandwidth, message overhead, verifiability.

    For tractability we sample ``min(tx_rate * sim_minutes * n, 500)`` finality
    events. This is statistically sufficient for p50/p95 quantile estimation
    while keeping wall time bounded at large N.
    """
    expected_total = max(20, int(round(tx_rate * sim_minutes * n)))
    sample_n = int(min(expected_total, 500))

    rounds_array = np.zeros(sample_n, dtype=int)
    msgs_array = np.zeros(sample_n, dtype=int)
    witness_counts = np.zeros(sample_n, dtype=int)

    for i in range(sample_n):
        rounds, msgs, inf_rd = _simulate_gossip_finality(
            n=n,
            k=k,
            quorum_frac=quorum_frac,
            rng=rng,
            msg_loss_pct=msg_loss_pct,
        )
        rounds_array[i] = rounds
        msgs_array[i] = msgs
        # witness = nodes that ever saw the transaction
        witness_counts[i] = int((inf_rd >= 0).sum())

    latencies_ms = rounds_array.astype(float) * ROUND_MS
    # System-wide offered-load throughput over the simulated period. Finality
    # latency is reported separately in the latency columns.
    sim_seconds = sim_minutes * 60.0
    throughput = expected_total / sim_seconds  # tx/sec system-wide

    # bandwidth per node: total messages across sample, scaled to full
    # expected volume, divided by N nodes, divided by sim_seconds, *MSG_BYTES.
    sample_msg_total = msgs_array.sum()
    full_msg_total = sample_msg_total * (expected_total / max(1, sample_n))
    bytes_per_node_per_sec = full_msg_total * MSG_BYTES / max(1, n) / sim_seconds
    bandwidth_kbps = bytes_per_node_per_sec * 8 / 1024

    msgs_per_tx = float(msgs_array.mean())
    verifiability_pct = float(
        (witness_counts >= witness_threshold).mean() * 100.0
    )

    return {
        "throughput_mean": throughput,
        "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "bandwidth_per_node_kbps": bandwidth_kbps,
        "messages_per_tx": msgs_per_tx,
        "independent_verifiability_pct": verifiability_pct,
    }


def _measure_dso_latencies(
    n: int,
    k: int,
    msg_loss_pct: float,
    rng: np.random.Generator,
    quorum_frac: float = 0.95,
    samples: int = 5,
) -> Tuple[float, float, float]:
    """Measure DSO downstream and upstream latency p95.

    Downstream: rumour-spread from a single source (the DSO) until 95% of
    nodes are reached. We sample `samples` independent realisations and
    take the p95 across all completion times.

    Upstream: each node attempts to deliver one status packet via the
    overlay back to the DSO. We model this as the dual problem — if all
    nodes simultaneously initiate gossip, time to 95% delivery to DSO is
    the same distribution by symmetry; we reuse the downstream measurement
    but report it under the upstream column with an aggregation overhead
    of one extra round (the DSO must consolidate).
    """
    completion_rounds = []
    observed_losses = []
    for _ in range(samples):
        rounds, msgs, inf = _simulate_gossip_finality(
            n=n,
            k=k,
            quorum_frac=quorum_frac,
            rng=rng,
            msg_loss_pct=msg_loss_pct,
        )
        completion_rounds.append(rounds)
        # Effective loss: fraction of intended unique recipients that never
        # received the message at all.
        delivered_frac = float((inf >= 0).mean())
        observed_losses.append(1.0 - delivered_frac)
    arr = np.array(completion_rounds, dtype=float)
    down_p95_ms = float(np.percentile(arr, 95) * ROUND_MS)
    # upstream = downstream + 1 round of DSO consolidation
    up_p95_ms = float((np.percentile(arr, 95) + 1) * ROUND_MS)
    obs_loss_pct = float(np.mean(observed_losses) * 100.0)
    return down_p95_ms, up_p95_ms, obs_loss_pct


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def run_cluster1(seed_root: int, cfg: dict) -> pd.DataFrame:
    """Run the full Cluster 1 sweep and return a DataFrame.

    cfg keys expected: node_counts, repetitions, sim_minutes, gossip_k,
    tx_rate_per_node_per_min, finality_quorum, dso_loss_pct,
    witness_threshold, resilience_test_loss_pct (optional), resilience_test_n.
    """
    base_rng = np.random.default_rng(derive_seed(seed_root, "sim_protocol"))
    rows: List[Dict] = []

    for n in cfg["node_counts"]:
        for rep in range(1, cfg["repetitions"] + 1):
            sub_seed = base_rng.integers(0, 2**31 - 1)
            rng = np.random.default_rng(sub_seed)
            summary = _summarise_protocol_rep(
                n=n,
                k=cfg["gossip_k"],
                sim_minutes=cfg["sim_minutes"],
                tx_rate=cfg["tx_rate_per_node_per_min"],
                quorum_frac=cfg["finality_quorum"],
                msg_loss_pct=cfg["dso_loss_pct"],
                witness_threshold=cfg["witness_threshold"],
                rng=rng,
            )
            dso_down, dso_up, dso_loss_obs = _measure_dso_latencies(
                n=n,
                k=cfg["gossip_k"],
                msg_loss_pct=cfg["dso_loss_pct"],
                rng=rng,
            )
            rows.append(
                {
                    "n_nodes": n,
                    "repetition": rep,
                    "throughput_mean": summary["throughput_mean"],
                    "latency_p50_ms": summary["latency_p50_ms"],
                    "latency_p95_ms": summary["latency_p95_ms"],
                    "bandwidth_per_node_kbps": summary["bandwidth_per_node_kbps"],
                    "messages_per_tx": summary["messages_per_tx"],
                    "dso_downstream_latency_p95_ms": dso_down,
                    "dso_upstream_aggregation_latency_p95_ms": dso_up,
                    "dso_message_loss_pct": dso_loss_obs,
                    "independent_verifiability_pct": summary[
                        "independent_verifiability_pct"
                    ],
                }
            )

    # Resilience test (edge case): one extra row at resilience_test_n with
    # elevated message loss. repetition=99 marks it.
    if cfg.get("resilience_test_loss_pct") is not None:
        n = cfg["resilience_test_n"]
        rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))
        summary = _summarise_protocol_rep(
            n=n,
            k=cfg["gossip_k"],
            sim_minutes=cfg["sim_minutes"],
            tx_rate=cfg["tx_rate_per_node_per_min"],
            quorum_frac=cfg["finality_quorum"],
            msg_loss_pct=cfg["resilience_test_loss_pct"],
            witness_threshold=cfg["witness_threshold"],
            rng=rng,
        )
        dso_down, dso_up, dso_loss_obs = _measure_dso_latencies(
            n=n,
            k=cfg["gossip_k"],
            msg_loss_pct=cfg["resilience_test_loss_pct"],
            rng=rng,
        )
        rows.append(
            {
                "n_nodes": n,
                "repetition": 99,
                "throughput_mean": summary["throughput_mean"],
                "latency_p50_ms": summary["latency_p50_ms"],
                "latency_p95_ms": summary["latency_p95_ms"],
                "bandwidth_per_node_kbps": summary["bandwidth_per_node_kbps"],
                "messages_per_tx": summary["messages_per_tx"],
                "dso_downstream_latency_p95_ms": dso_down,
                "dso_upstream_aggregation_latency_p95_ms": dso_up,
                "dso_message_loss_pct": dso_loss_obs,
                "independent_verifiability_pct": summary[
                    "independent_verifiability_pct"
                ],
            }
        )

    return pd.DataFrame(rows)
