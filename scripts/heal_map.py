"""
Phase 3 Execution Script: Skeletonization & Topological Map Healing (Completes Stage A).
Outputs clean, versioned Stage A NetworkX graph artifact and connectivity metrics.
"""

import sys
import json
from pathlib import Path
import yaml
import cv2
import networkx as nx

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.logging_config import setup_logger
from route_resilience.graph.skeletonize import mask_to_skeleton
from route_resilience.graph.mask_to_graph import extract_graph_from_skeleton
from route_resilience.graph.healing import heal_broken_graph
from route_resilience.graph.metrics import compute_connectivity_ratio, compute_topological_accuracy

logger = setup_logger("scripts.heal_map")


def run_map_healing(config_path: Path):
    """Executes Stage A map healing pipeline from predicted mask to serialized graph artifact.

    Args:
        config_path: Path to YAML configuration file.
    """
    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    healing_cfg = config["healing"]

    processed_dir = PROJECT_ROOT / data_cfg["output_dir"]
    mask_path = processed_dir / "full_osm_mask.png"

    if not mask_path.exists():
        raise FileNotFoundError(f"Binary mask not found at {mask_path}. Run prepare_data.py first.")

    # 1. Load binary mask
    logger.info("Step 1: Reading binary road mask...")
    binary_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Simulate occluded broken mask by introducing breaks into mask to demonstrate healing
    broken_mask = binary_mask.copy()
    h, w = broken_mask.shape
    cv2.circle(broken_mask, (int(w * 0.5), int(h * 0.5)), 15, 0, -1)  # Synthetic gap

    # 2. Skeletonization
    logger.info("Step 2: Performing 1-pixel skeletonization...")
    skeleton = mask_to_skeleton(broken_mask)
    cv2.imwrite(str(processed_dir / "skeleton_centerline.png"), skeleton)

    # 3. Mask to Raw Graph Conversion
    logger.info("Step 3: Extracting raw NetworkX graph from skeleton...")
    raw_graph = extract_graph_from_skeleton(skeleton, pixel_scale_m=10.0)

    # 4. Topological Healing Algorithm
    logger.info("Step 4: Executing topological healing algorithm...")
    healed_graph, heal_summary = heal_broken_graph(
        raw_graph,
        distance_threshold_px=healing_cfg["distance_threshold_px"],
        max_angle_diff_deg=healing_cfg["max_angle_diff_deg"],
        pixel_scale_m=10.0,
    )

    # 5. Compute Stage A Metrics
    logger.info("Step 5: Computing Connectivity Ratio & Topological Accuracy...")
    conn_metrics = compute_connectivity_ratio(raw_graph, healed_graph)
    topo_metrics = compute_topological_accuracy(healed_graph)

    # 6. Serialize Stage A Graph Artifact to Disk
    graph_out_dir = PROJECT_ROOT / healing_cfg["output_graph_dir"]
    graph_out_dir.mkdir(parents=True, exist_ok=True)

    # Clean graph attributes for serializability
    clean_graph = nx.Graph()
    for n, data in healed_graph.nodes(data=True):
        clean_graph.add_node(
            int(n),
            y=float(data["pos"][0]),
            x=float(data["pos"][1]),
            is_endpoint=bool(data.get("is_endpoint", False)),
        )

    for u, v, data in healed_graph.edges(data=True):
        clean_graph.add_edge(
            int(u),
            int(v),
            weight=float(data["weight"]),
            length_px=float(data.get("length_px", 0.0)),
            healed=bool(data.get("healed", False)),
        )

    graphML_path = graph_out_dir / "healed_stage_a_graph.graphml"
    json_path = graph_out_dir / "healed_stage_a_graph.json"

    # Save GraphML format for Stage B networkx consumption
    nx.write_graphml(clean_graph, graphML_path)

    # Save GeoJSON-style node/edge dictionary for dashboard rendering
    graph_json = {
        "metadata": {
            "nodes_count": int(healed_graph.number_of_nodes()),
            "edges_count": int(healed_graph.number_of_edges()),
            "components_before": int(heal_summary["components_before"]),
            "components_after": int(heal_summary["components_after"]),
            "healed_edges_added": int(heal_summary["healed_edges_added"]),
            "connectivity_gain_ratio": float(conn_metrics["connectivity_gain_ratio"]),
        },
        "nodes": [
            {
                "id": int(n),
                "y": float(healed_graph.nodes[n]["pos"][0]),
                "x": float(healed_graph.nodes[n]["pos"][1]),
                "is_endpoint": bool(healed_graph.nodes[n].get("is_endpoint", False)),
            }
            for n in healed_graph.nodes()
        ],
        "edges": [
            {
                "source": int(u),
                "target": int(v),
                "weight": float(healed_graph.edges[u, v]["weight"]),
                "healed": bool(healed_graph.edges[u, v].get("healed", False)),
            }
            for u, v in healed_graph.edges()
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph_json, f, indent=2)

    # Print Stage A Completion Report
    report = f"""
==================================================
        STAGE A MAP HEALING COMPLETION REPORT     
==================================================
Nodes Count         : {healed_graph.number_of_nodes()}
Edges Count         : {healed_graph.number_of_edges()}
Components Before   : {heal_summary['components_before']}
Components After    : {heal_summary['components_after']}
Healing Edges Added : {heal_summary['healed_edges_added']}
--------------------------------------------------
Connectivity Gain Ratio : {conn_metrics['connectivity_gain_ratio']:.2f}x
Connectivity Coverage   : {conn_metrics['connectivity_coverage']*100:.1f}%
Topological Accuracy Error: {topo_metrics['mean_shortest_path_error_pct']:.2f}%
Reachable Pairs Pct     : {topo_metrics['reachable_pair_pct']:.1f}%
--------------------------------------------------
Serialized GraphML Graph: {graphML_path}
Serialized JSON Graph   : {json_path}
==================================================
"""
    logger.info(report)
    print(report)

    return healed_graph, conn_metrics, topo_metrics


if __name__ == "__main__":
    config_file = PROJECT_ROOT / "configs" / "default.yaml"
    run_map_healing(config_file)
