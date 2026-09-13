"""
Topological Graph Metrics (Connectivity Ratio & Topological Accuracy vs OSM Ground Truth).
"""

import random
from typing import Dict, Any, Tuple, Optional
import numpy as np
import networkx as nx

from route_resilience.logging_config import setup_logger

logger = setup_logger("graph.metrics")


def compute_connectivity_ratio(
    raw_graph: nx.Graph,
    healed_graph: nx.Graph,
) -> Dict[str, float]:
    """Computes Connectivity Ratio metric comparing component sizes before vs after healing.

    Args:
        raw_graph: Initial unhealed NetworkX graph.
        healed_graph: Healed NetworkX graph.

    Returns:
        Dict containing:
        - 'largest_component_before': int
        - 'largest_component_after': int
        - 'connectivity_gain_ratio': float (after / max(before, 1))
        - 'connectivity_coverage': float (largest_after / total_nodes)
    """
    total_nodes = max(healed_graph.number_of_nodes(), 1)

    # Largest connected component before healing
    if raw_graph.number_of_nodes() > 0:
        cc_before = max(len(c) for c in nx.connected_components(raw_graph))
    else:
        cc_before = 0

    # Largest connected component after healing
    if healed_graph.number_of_nodes() > 0:
        cc_after = max(len(c) for c in nx.connected_components(healed_graph))
    else:
        cc_after = 0

    ratio = float(cc_after) / float(max(cc_before, 1))
    coverage = float(cc_after) / float(total_nodes)

    return {
        "largest_component_before": cc_before,
        "largest_component_after": cc_after,
        "connectivity_gain_ratio": ratio,
        "connectivity_coverage": coverage,
    }


def compute_topological_accuracy(
    healed_graph: nx.Graph,
    gt_graph: Optional[nx.Graph] = None,
    num_sample_pairs: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """Computes Shortest-Path Topological Accuracy relative to ground truth graph.

    Args:
        healed_graph: Healed Stage A NetworkX graph.
        gt_graph: Ground truth (OSM) NetworkX graph (if None, synthetic baseline is used).
        num_sample_pairs: Number of random origin-destination pairs to evaluate.
        seed: Random seed for sample reproducibility.

    Returns:
        Dict containing:
        - 'mean_shortest_path_error_pct': float
        - 'reachable_pair_pct': float
    """
    if healed_graph.number_of_nodes() < 2:
        return {"mean_shortest_path_error_pct": 0.0, "reachable_pair_pct": 100.0}

    random.seed(seed)
    nodes = list(healed_graph.nodes())

    # Sample random origin-destination pairs
    sample_pairs = []
    attempts = 0
    while len(sample_pairs) < num_sample_pairs and attempts < num_sample_pairs * 5:
        u, v = random.sample(nodes, 2)
        sample_pairs.append((u, v))
        attempts += 1

    errors = []
    reachable_count = 0

    for u, v in sample_pairs:
        if nx.has_path(healed_graph, u, v):
            reachable_count += 1
            sp_healed = nx.shortest_path_length(healed_graph, u, v, weight="weight")

            if gt_graph is not None and gt_graph.has_node(u) and gt_graph.has_node(v) and nx.has_path(gt_graph, u, v):
                sp_gt = nx.shortest_path_length(gt_graph, u, v, weight="weight")
                rel_err = abs(sp_healed - sp_gt) / max(sp_gt, 1e-3)
                errors.append(rel_err)
            else:
                # If ground truth node matching isn't direct, measure Euclidean baseline ratio
                pos_u = np.array(healed_graph.nodes[u]["pos"])
                pos_v = np.array(healed_graph.nodes[v]["pos"])
                direct_dist = np.linalg.norm(pos_u - pos_v) * 10.0
                rel_err = abs(sp_healed - direct_dist) / max(direct_dist, 1e-3)
                errors.append(rel_err)

    reachable_pct = (float(reachable_count) / float(max(len(sample_pairs), 1))) * 100.0
    mean_err_pct = (float(np.mean(errors)) * 100.0) if errors else 0.0

    return {
        "mean_shortest_path_error_pct": float(mean_err_pct),
        "reachable_pair_pct": float(reachable_pct),
    }
