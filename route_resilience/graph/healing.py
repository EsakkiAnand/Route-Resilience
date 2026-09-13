"""
Topological Map Healing Algorithm for Broken Road Segmentations.
Combines Angular Trajectory Alignment, Distance Thresholding, Union-Find & Kruskal's MST Logic.
"""

import logging
import math
from typing import Dict, Tuple, List, Set, Optional, Any
import numpy as np
import networkx as nx

from route_resilience.logging_config import setup_logger

logger = setup_logger("graph.healing")


class UnionFind:
    """Disjoint Set Union (Union-Find) helper for graph component merging."""

    def __init__(self, elements):
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            return True
        return False


def get_endpoint_trajectory_vector(
    graph: nx.Graph,
    node: int,
) -> np.ndarray:
    """Computes the directional vector pointing outward along an endpoint's road segment.

    Args:
        graph: NetworkX graph.
        node: Node index of an endpoint (degree == 1).

    Returns:
        2D unit direction vector numpy array [dy, dx].
    """
    pos_u = np.array(graph.nodes[node]["pos"], dtype=np.float32)
    neighbors = list(graph.neighbors(node))

    if not neighbors:
        return np.array([0.0, 0.0], dtype=np.float32)

    nbr = neighbors[0]
    pos_nbr = np.array(graph.nodes[nbr]["pos"], dtype=np.float32)

    # Vector pointing from neighbor towards endpoint (outward trajectory)
    vec = pos_u - pos_nbr
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)
    return vec / norm


def compute_angular_deviation_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes angle in degrees between two 2D direction vectors."""
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return float(np.degrees(angle_rad))


def heal_broken_graph(
    graph: nx.Graph,
    distance_threshold_px: float = 30.0,
    max_angle_diff_deg: float = 45.0,
    pixel_scale_m: float = 10.0,
) -> Tuple[nx.Graph, Dict[str, Any]]:
    """Heals broken road masks into a single connected graph using Union-Find + MST logic.

    Args:
        graph: Input disconnected Stage A NetworkX road graph.
        distance_threshold_px: Max pixel distance to allow healing bridge insertion.
        max_angle_diff_deg: Max angular trajectory deviation allowed for connection.
        pixel_scale_m: Scale in meters per pixel.

    Returns:
        Tuple of (healed nx.Graph, healing execution summary dictionary).
    """
    healed_G = graph.copy()
    if healed_G.number_of_nodes() == 0:
        return healed_G, {"healed_edges_added": 0, "components_before": 0, "components_after": 0}

    initial_components = list(nx.connected_components(healed_G))
    num_comp_before = len(initial_components)

    if num_comp_before <= 1:
        logger.info("Graph is already fully connected. No healing edges required.")
        return healed_G, {
            "healed_edges_added": 0,
            "components_before": num_comp_before,
            "components_after": num_comp_before,
        }

    uf = UnionFind(healed_G.nodes())

    # Find endpoint nodes (degree == 1)
    endpoints = [n for n, deg in healed_G.degree() if deg == 1]

    candidate_bridges = []

    # Evaluate candidate endpoint pairs across different components
    for i in range(len(endpoints)):
        u = endpoints[i]
        pos_u = np.array(healed_G.nodes[u]["pos"], dtype=np.float32)
        dir_u = get_endpoint_trajectory_vector(healed_G, u)

        for j in range(i + 1, len(endpoints)):
            v = endpoints[j]
            if uf.find(u) == uf.find(v):
                continue  # Already in same component

            pos_v = np.array(healed_G.nodes[v]["pos"], dtype=np.float32)
            dist_px = np.linalg.norm(pos_u - pos_v)

            if dist_px > distance_threshold_px:
                continue

            # Candidate bridge direction vector (u -> v)
            bridge_dir = (pos_v - pos_u) / max(dist_px, 1e-6)
            dir_v = get_endpoint_trajectory_vector(healed_G, v)

            # Check trajectory alignment
            angle_u = compute_angular_deviation_deg(dir_u, bridge_dir)
            angle_v = compute_angular_deviation_deg(dir_v, -bridge_dir)

            if angle_u <= max_angle_diff_deg and angle_v <= max_angle_diff_deg:
                candidate_bridges.append({
                    "u": u,
                    "v": v,
                    "dist_px": dist_px,
                    "angle_u": angle_u,
                    "angle_v": angle_v,
                })

    # Sort candidate bridges by distance (Kruskal's MST logic)
    candidate_bridges.sort(key=lambda x: x["dist_px"])

    healed_edges_added = 0
    for bridge in candidate_bridges:
        u, v = bridge["u"], bridge["v"]
        dist_px = bridge["dist_px"]

        # Only insert bridge if it merges two disjoint components (Union-Find)
        if uf.union(u, v):
            dist_m = float(dist_px * pixel_scale_m)
            healed_G.add_edge(
                u,
                v,
                weight=dist_m,
                length_px=float(dist_px),
                healed=True,
                angle_u=bridge["angle_u"],
                angle_v=bridge["angle_v"],
            )
            healed_edges_added += 1

    num_comp_after = nx.number_connected_components(healed_G)
    logger.info(
        f"Topological Healing Complete: Added {healed_edges_added} healing edges. "
        f"Components reduced from {num_comp_before} -> {num_comp_after}."
    )

    summary = {
        "healed_edges_added": healed_edges_added,
        "components_before": num_comp_before,
        "components_after": num_comp_after,
        "candidate_bridges_evaluated": len(candidate_bridges),
    }

    return healed_G, summary
