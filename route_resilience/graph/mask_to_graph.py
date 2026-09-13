"""
Pixel-Walk Network Graph Extraction from Skeleton Rasters into NetworkX.
"""

import logging
from typing import Dict, Tuple, List, Set, Any
import numpy as np
import networkx as nx

from route_resilience.logging_config import setup_logger

logger = setup_logger("graph.mask_to_graph")


def extract_graph_from_skeleton(
    skeleton: np.ndarray,
    pixel_scale_m: float = 10.0,
) -> nx.Graph:
    """Converts 1-pixel wide skeleton image into a NetworkX Graph.

    Nodes represent intersections (degree > 2) and dead-end endpoints (degree == 1).
    Edges represent road segments connecting nodes, with real-world distance weights.

    Args:
        skeleton: 1-pixel wide binary skeleton array [H, W] (0 or 255).
        pixel_scale_m: Real-world meters per pixel (default 10.0m for Sentinel-2).

    Returns:
        networkx.Graph with node attributes ('pos': (y, x)) and edge attributes ('weight': meters, 'pixel_path': list).
    """
    h, w = skeleton.shape
    skel_pts = set(zip(*np.where(skeleton > 0)))

    if not skel_pts:
        logger.warning("Skeleton array is empty. Returning empty graph.")
        return nx.Graph()

    # 8-neighbor offsets
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def get_neighbors(p: Tuple[int, int]) -> List[Tuple[int, int]]:
        y, x = p
        res = []
        for dy, dx in neighbors_8:
            ny, nx_ = y + dy, x + dx
            if (ny, nx_) in skel_pts:
                res.append((ny, nx_))
        return res

    # Classify nodes (intersections & endpoints)
    junction_nodes: Set[Tuple[int, int]] = set()
    endpoint_nodes: Set[Tuple[int, int]] = set()

    for p in skel_pts:
        deg = len(get_neighbors(p))
        if deg != 2:  # Degree 1 (endpoint) or Degree >= 3 (intersection) or Degree 0 (isolated point)
            if deg == 1 or deg == 0:
                endpoint_nodes.add(p)
            else:
                junction_nodes.add(p)

    all_nodes = junction_nodes.union(endpoint_nodes)

    G = nx.Graph()

    # Add node attributes
    node_id_map: Dict[Tuple[int, int], int] = {}
    for idx, p in enumerate(all_nodes):
        node_id_map[p] = idx
        G.add_node(
            idx,
            pos=p,
            y=p[0],
            x=p[1],
            is_endpoint=(p in endpoint_nodes),
            degree_type="endpoint" if p in endpoint_nodes else "intersection",
        )

    # Pixel-walk to trace edges between nodes
    visited_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

    for start_p in all_nodes:
        start_id = node_id_map[start_p]
        for nxt in get_neighbors(start_p):
            if (start_p, nxt) in visited_edges:
                continue

            path = [start_p, nxt]
            visited_edges.add((start_p, nxt))
            visited_edges.add((nxt, start_p))

            curr = nxt
            prev = start_p

            # Walk along 2-degree interior pixels until hitting another node
            while curr not in all_nodes:
                nbrs = get_neighbors(curr)
                next_candidates = [pt for pt in nbrs if pt != prev]
                if not next_candidates:
                    break
                next_pt = next_candidates[0]
                path.append(next_pt)

                visited_edges.add((curr, next_pt))
                visited_edges.add((next_pt, curr))
                prev = curr
                curr = next_pt

            if curr in all_nodes and curr != start_p:
                end_id = node_id_map[curr]
                # Calculate path length in meters
                length_px = 0.0
                for i in range(len(path) - 1):
                    p1, p2 = path[i], path[i + 1]
                    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                    length_px += dist

                length_m = length_px * pixel_scale_m

                if not G.has_edge(start_id, end_id):
                    G.add_edge(
                        start_id,
                        end_id,
                        weight=float(length_m),
                        length_px=float(length_px),
                        pixel_path=path,
                        healed=False,
                    )

    logger.info(f"Extracted Stage A graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G
