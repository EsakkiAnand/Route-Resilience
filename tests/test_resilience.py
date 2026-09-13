"""
Unit tests for Phase 4 Structural Intelligence & Stress Testing (Stage B).
Verifies Resilience Index, Travel Time Impact, and Disconnection on hand-verifiable synthetic graphs.
"""

import unittest
import sys
from pathlib import Path
import networkx as nx

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.graph.centrality import compute_betweenness_centrality, get_gatekeeper_nodes
from route_resilience.graph.ablation import simulate_node_closure


class TestResilienceIntelligence(unittest.TestCase):
    """Test suite for centrality ranking and node ablation stress testing."""

    def test_betweenness_centrality_and_gatekeeper_ranking(self):
        """Test gatekeeper node ranking on a star/hub topology."""
        G = nx.Graph()
        # Hub node 0 connected to nodes 1, 2, 3, 4
        for i in range(1, 5):
            G.add_edge(0, i, weight=10.0)

        centrality = compute_betweenness_centrality(G)
        gatekeepers = get_gatekeeper_nodes(G, top_n=1, centrality_map=centrality)

        # Hub node 0 must have the highest centrality score
        self.assertEqual(gatekeepers[0]["node_id"], 0)
        self.assertGreater(gatekeepers[0]["centrality"], 0.5)

    def test_hand_verifiable_diamond_graph_resilience(self):
        """Test Resilience Index calculation on a hand-verifiable diamond graph with alternative path.

        Graph topology:
              (1)
             /   \
          (0)     (3)
             \   /
              (2)
        All edge weights = 10.0m.
        Baseline OD pair (0, 3) shortest path = 20.0m (via node 1 or node 2).
        If node 1 closes, alternative path exists via node 2 (0 -> 2 -> 3 = 20.0m).
        Resilience Index = 20.0 / 20.0 = 1.0 (100% resilient, 0% delay, disconnected=False).
        """
        G = nx.Graph()
        G.add_edge(0, 1, weight=10.0)
        G.add_edge(1, 3, weight=10.0)
        G.add_edge(0, 2, weight=10.0)
        G.add_edge(2, 3, weight=10.0)

        res = simulate_node_closure(G, closed_node_id=1, sample_pairs_count=10, seed=42)

        self.assertAlmostEqual(res["resilience_index"], 1.0, places=2)
        self.assertAlmostEqual(res["travel_time_impact_pct"], 0.0, places=1)
        self.assertFalse(res["disconnected"])
        self.assertEqual(res["disconnected_pair_count"], 0)

    def test_hand_verifiable_line_graph_disconnection_edge_case(self):
        """Test node ablation on a 3-node path line graph (0 <-> 1 <-> 2) causing full disconnection.

        If middle node 1 is closed, node 0 and node 2 become completely disconnected!
        Verify disconnected=True and severe resilience drop.
        """
        G = nx.Graph()
        G.add_edge(0, 1, weight=15.0)
        G.add_edge(1, 2, weight=15.0)

        res = simulate_node_closure(G, closed_node_id=1, sample_pairs_count=5, seed=42)

        self.assertTrue(res["disconnected"])
        self.assertEqual(res["disconnected_pair_count"], res["total_pairs_evaluated"])
        self.assertEqual(res["resilience_index"], 0.0)


if __name__ == "__main__":
    unittest.main()
