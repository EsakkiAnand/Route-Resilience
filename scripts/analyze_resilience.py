"""
Phase 4 Execution Script: Structural Intelligence & Resilience Stress Testing (Stage B).
CLI Command taking a node ID or top gatekeeper and printing resilience metrics.
"""

import sys
import json
import argparse
from pathlib import Path
import yaml
import networkx as nx

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.logging_config import setup_logger
from route_resilience.graph.centrality import compute_betweenness_centrality, get_gatekeeper_nodes
from route_resilience.graph.ablation import simulate_node_closure

logger = setup_logger("scripts.analyze_resilience")


def run_resilience_analysis(
    node_id: int = None,
    top_gatekeeper: bool = True,
    config_path: Path = None,
) -> dict:
    """Executes Stage B resilience stress testing on Stage A graph.

    Args:
        node_id: Optional specific node ID to simulate closure for.
        top_gatekeeper: If True and node_id is None, automatically targets the top gatekeeper node.
        config_path: Optional path to default YAML config file.

    Returns:
        Dictionary of resilience impact simulation metrics.
    """
    config_path = config_path or (PROJECT_ROOT / "configs" / "default.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    graphML_path = PROJECT_ROOT / config["healing"]["output_graph_dir"] / "healed_stage_a_graph.graphml"
    if not graphML_path.exists():
        raise FileNotFoundError(f"Stage A GraphML file not found at {graphML_path}. Run heal_map.py first.")

    logger.info(f"Loading Stage A healed graph from {graphML_path}...")
    graph = nx.read_graphml(graphML_path)

    # Convert node keys to int if necessary
    mapping = {n: int(n) if str(n).isdigit() else n for n in graph.nodes()}
    graph = nx.relabel_nodes(graph, mapping)

    # 1. Gatekeeper Node Ranking
    logger.info("Computing weighted Betweenness Centrality...")
    centrality_map = compute_betweenness_centrality(graph, weight_attribute="weight")
    gatekeepers = get_gatekeeper_nodes(graph, top_n=config["analysis"]["top_n_gatekeepers"], centrality_map=centrality_map)

    # Target Node Selection
    if node_id is None:
        if top_gatekeeper and gatekeepers:
            target_node = gatekeepers[0]["node_id"]
            logger.info(f"Selected Top Gatekeeper Node: {target_node} (Centrality: {gatekeepers[0]['centrality']:.4f})")
        else:
            target_node = list(graph.nodes())[0]
    else:
        target_node = int(node_id)

    # 2. Node Ablation Simulation
    logger.info(f"Simulating road closure perturbation for Node {target_node}...")
    impact = simulate_node_closure(
        graph,
        closed_node_id=target_node,
        sample_pairs_count=config["analysis"]["resilience_index_sample_pairs"],
    )

    # Output Report
    report = f"""
==================================================
        STAGE B RESILIENCE ANALYSIS REPORT        
==================================================
Target Node ID        : {impact['closed_node_id']}
Resilience Index      : {impact['resilience_index']:.4f}
Travel Time Impact    : +{impact['travel_time_impact_pct']:.2f}%
Affected Road Length  : {impact['affected_segment_length_km']:.3f} km
Full Disconnection    : {impact['disconnected']} (Severed Pairs: {impact['disconnected_pair_count']}/{impact['total_pairs_evaluated']})
--------------------------------------------------
Baseline Avg Distance : {impact['baseline_avg_length_m']:.1f} m
Perturbed Avg Distance: {impact['perturbed_avg_length_m']:.1f} m
==================================================
Top-5 Gatekeeper Junctions:
"""
    for gk in gatekeepers[:5]:
        report += f"  - Node {gk['node_id']:3d} | Centrality: {gk['centrality']:.4f} | Pos: {gk['pos']}\n"
    report += "==================================================\n"

    logger.info(report)
    print(report)

    # Return key deliverable schema dictionary
    deliverable = {
        "closed_node_id": impact["closed_node_id"],
        "resilience_index": round(impact["resilience_index"], 4),
        "travel_time_impact_pct": round(impact["travel_time_impact_pct"], 2),
        "affected_km": round(impact["affected_segment_length_km"], 3),
        "disconnected": impact["disconnected"],
        "disconnected_pair_count": impact["disconnected_pair_count"],
    }
    return deliverable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Resilience Stress Testing CLI")
    parser.add_argument("--node-id", type=int, default=None, help="Node ID to simulate closure for")
    parser.add_argument("--top-gatekeeper", action="store_true", default=True, help="Target top gatekeeper node")

    args = parser.parse_args()
    res = run_resilience_analysis(node_id=args.node_id, top_gatekeeper=args.top_gatekeeper)
    print("JSON Deliverable Output:")
    print(json.dumps(res, indent=2))
