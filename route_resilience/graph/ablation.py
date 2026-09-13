"""
Real-Time Stress Testing & Node-Ablation Resilience Simulation.
Computes Resilience Index, Travel Time Impact %, Affected Segment Length (km), and Disconnection metrics.
"""

import logging
import random
from typing import Dict, Any, List, Tuple
import networkx as nx
import numpy as np

from route_resilience.logging_config import setup_logger

logger = setup_logger("graph.ablation")


def simulate_node_closure(
    graph: nx.Graph,
    closed_node_id: int,
    sample_pairs_count: int = 100,
    disconnected_penalty_factor: float = 5.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Simulates closing a node (junction/road) and quantifies network-wide cascading impact.

    Args:
        graph: Stage A NetworkX graph.
        closed_node_id: ID of node to simulate closure for.
        sample_pairs_count: Number of random OD pairs for evaluation.
        disconnected_penalty_factor: Penalty multiplier for disconnected OD pairs relative to max graph path.
        seed: Random seed for sample reproducibility.

    Returns:
        Dict containing:
        - 'closed_node_id': int
        - 'resilience_index': float in [0.0, 1.0]
        - 'travel_time_impact_pct': float (% increase in travel path length)
        - 'affected_segment_length_km': float (km of road closed)
        - 'disconnected': bool (True if any OD pair became unreachable)
        - 'disconnected_pair_count': int
        - 'total_pairs_evaluated': int
        - 'baseline_avg_length_m': float
        - 'perturbed_avg_length_m': float
    """
    if closed_node_id not in graph:
        raise ValueError(f"Node ID {closed_node_id} does not exist in graph.")

    # Calculate affected segment length (km) for closed node
    affected_meters = sum(
        data.get("weight", 0.0) for _, _, data in graph.edges(closed_node_id, data=True)
    )
    affected_km = affected_meters / 1000.0

    # Sample random OD pairs excluding closed node
    available_nodes = [n for n in graph.nodes() if n != closed_node_id]
    if len(available_nodes) < 2:
        return {
            "closed_node_id": int(closed_node_id),
            "resilience_index": 0.0,
            "travel_time_impact_pct": 0.0,
            "affected_segment_length_km": float(affected_km),
            "disconnected": True,
            "disconnected_pair_count": 0,
            "total_pairs_evaluated": 0,
            "baseline_avg_length_m": 0.0,
            "perturbed_avg_length_m": 0.0,
        }

    random.seed(seed)
    od_pairs: List[Tuple[int, int]] = []
    attempts = 0
    while len(od_pairs) < sample_pairs_count and attempts < sample_pairs_count * 10:
        u, v = random.sample(available_nodes, 2)
        if nx.has_path(graph, u, v):
            od_pairs.append((u, v))
        attempts += 1

    if not od_pairs:
        logger.warning("No connected OD pairs found in baseline graph.")
        return {
            "closed_node_id": int(closed_node_id),
            "resilience_index": 1.0,
            "travel_time_impact_pct": 0.0,
            "affected_segment_length_km": float(affected_km),
            "disconnected": False,
            "disconnected_pair_count": 0,
            "total_pairs_evaluated": 0,
            "baseline_avg_length_m": 0.0,
            "perturbed_avg_length_m": 0.0,
        }

    # 1. Compute Baseline Average Shortest Path Lengths
    baseline_lengths = []
    for u, v in od_pairs:
        l = nx.shortest_path_length(graph, u, v, weight="weight")
        baseline_lengths.append(l)

    baseline_avg = float(np.mean(baseline_lengths))

    # 2. Perturbed Graph (Remove Closed Node)
    perturbed_G = graph.copy()
    perturbed_G.remove_node(closed_node_id)

    # Max path length in baseline graph for penalty calibration
    max_baseline_path = max(baseline_lengths) if baseline_lengths else 1000.0
    penalty_distance = max_baseline_path * disconnected_penalty_factor

    # 3. Compute Perturbed Path Lengths
    perturbed_lengths = []
    disconnected_count = 0

    for u, v in od_pairs:
        if nx.has_path(perturbed_G, u, v):
            l = nx.shortest_path_length(perturbed_G, u, v, weight="weight")
            perturbed_lengths.append(l)
        else:
            disconnected_count += 1
            perturbed_lengths.append(penalty_distance)

    perturbed_avg = float(np.mean(perturbed_lengths))

    # 4. Resilience Index & Impact Metrics
    if disconnected_count == len(od_pairs):
        # Total network collapse
        resilience_index = 0.0
        travel_time_impact_pct = float("inf")
    else:
        resilience_index = float(baseline_avg) / float(max(perturbed_avg, 1e-6))
        # Clamp Resilience Index to [0.0, 1.0]
        resilience_index = max(0.0, min(1.0, resilience_index))

        travel_time_impact_pct = (
            (perturbed_avg - baseline_avg) / max(baseline_avg, 1e-6)
        ) * 100.0

    is_disconnected = disconnected_count > 0

    result = {
        "closed_node_id": int(closed_node_id),
        "resilience_index": float(resilience_index),
        "travel_time_impact_pct": float(travel_time_impact_pct),
        "affected_segment_length_km": float(affected_km),
        "disconnected": bool(is_disconnected),
        "disconnected_pair_count": int(disconnected_count),
        "total_pairs_evaluated": len(od_pairs),
        "baseline_avg_length_m": float(baseline_avg),
        "perturbed_avg_length_m": float(perturbed_avg),
    }

    logger.info(
        f"Closure Simulation (Node {closed_node_id}): Resilience Index = {resilience_index:.4f}, "
        f"Travel Time Impact = +{travel_time_impact_pct:.1f}%, Disconnected = {is_disconnected}"
    )

    return result
