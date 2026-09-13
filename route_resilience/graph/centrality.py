"""
Structural Intelligence: Weighted Betweenness Centrality & Gatekeeper Node Ranking.
"""

import logging
from typing import Dict, List, Any, Tuple
import networkx as nx
import numpy as np

from route_resilience.logging_config import setup_logger

logger = setup_logger("graph.centrality")


def compute_betweenness_centrality(
    graph: nx.Graph,
    weight_attribute: str = "weight",
    normalized: bool = True,
) -> Dict[int, float]:
    """Computes weighted Betweenness Centrality for all nodes in the graph.

    Args:
        graph: Stage A NetworkX graph.
        weight_attribute: Edge weight attribute name (default 'weight' in meters).
        normalized: Whether to normalize centrality scores to [0, 1].

    Returns:
        Dictionary mapping node_id -> betweenness centrality score float.
    """
    if graph.number_of_nodes() == 0:
        return {}

    # Compute betweenness centrality using segment length as weight
    centrality = nx.betweenness_centrality(
        graph,
        weight=weight_attribute,
        normalized=normalized,
    )
    return centrality


def get_gatekeeper_nodes(
    graph: nx.Graph,
    top_n: int = 10,
    centrality_map: Dict[int, float] = None,
) -> List[Dict[str, Any]]:
    """Ranks and extracts top-N Gatekeeper Nodes (critical junctions).

    Args:
        graph: Stage A NetworkX graph.
        top_n: Number of top gatekeeper nodes to return.
        centrality_map: Optional precomputed centrality dictionary.

    Returns:
        List of dictionaries sorted by descending centrality score:
        [{'node_id': int, 'centrality': float, 'pos': (y, x), 'degree': int}]
    """
    if centrality_map is None:
        centrality_map = compute_betweenness_centrality(graph)

    # Sort nodes by centrality score descending
    sorted_nodes = sorted(centrality_map.items(), key=lambda item: item[1], reverse=True)

    gatekeepers = []
    for node_id, score in sorted_nodes[:top_n]:
        node_data = graph.nodes[node_id]
        pos = node_data.get("pos", (node_data.get("y", 0.0), node_data.get("x", 0.0)))
        deg = graph.degree(node_id)

        gatekeepers.append({
            "node_id": int(node_id),
            "centrality": float(score),
            "pos": (float(pos[0]), float(pos[1])),
            "degree": int(deg),
            "is_endpoint": bool(node_data.get("is_endpoint", False)),
        })

    logger.info(f"Identified top-{len(gatekeepers)} Gatekeeper Nodes.")
    return gatekeepers
