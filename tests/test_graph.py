"""
Unit tests for Phase 3 Skeletonization & Topological Map Healing (Stage A).
Tests synthetic broken graphs to verify valid connections are made and implausible bridges are rejected.
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import networkx as nx

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.graph.skeletonize import mask_to_skeleton
from route_resilience.graph.mask_to_graph import extract_graph_from_skeleton
from route_resilience.graph.healing import heal_broken_graph, compute_angular_deviation_deg
from route_resilience.graph.metrics import compute_connectivity_ratio, compute_topological_accuracy


class TestGraphHealing(unittest.TestCase):
    """Test suite for graph extraction and topological healing logic."""

    def test_skeletonize_binary_mask(self):
        """Test skeletonization reduces thick road mask to 1-pixel centerline."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, :] = 255  # 10-pixel thick horizontal road line

        skeleton = mask_to_skeleton(mask)
        self.assertEqual(skeleton.shape, (100, 100))
        self.assertGreater(np.sum(skeleton > 0), 0)
        # Skeleton should be dramatically thinner than original mask (1000 pixels -> <250 pixels)
        self.assertLess(np.sum(skeleton > 0), 250)

    def test_angular_deviation_degrees(self):
        """Test vector angular deviation calculation."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([1.0, 0.0])
        self.assertAlmostEqual(compute_angular_deviation_deg(v1, v2), 0.0)

        v3 = np.array([0.0, 1.0])
        self.assertAlmostEqual(compute_angular_deviation_deg(v1, v3), 90.0)

    def test_synthetic_broken_graph_healing(self):
        """Test healing algorithm reconnects valid aligned endpoints and rejects implausible bridges."""
        G = nx.Graph()

        # Component 1: Horizontal line segment from (10, 10) to (10, 45) -> Endpoint at (10, 45) pointing right
        G.add_node(0, pos=(10, 10), degree_type="endpoint")
        G.add_node(1, pos=(10, 45), degree_type="endpoint")
        G.add_edge(0, 1, weight=350.0)

        # Component 2: Aligned horizontal line segment from (10, 55) to (10, 90) -> Endpoint at (10, 55) pointing left
        # Distance = 10px, Angle = 0 deg -> SHOULD BE HEALED!
        G.add_node(2, pos=(10, 55), degree_type="endpoint")
        G.add_node(3, pos=(10, 90), degree_type="endpoint")
        G.add_edge(2, 3, weight=350.0)

        # Component 3: Implausible far away component at (80, 80) to (80, 90)
        # Distance = ~75px (> 30px threshold) -> SHOULD NOT BE HEALED!
        G.add_node(4, pos=(80, 80), degree_type="endpoint")
        G.add_node(5, pos=(80, 90), degree_type="endpoint")
        G.add_edge(4, 5, weight=100.0)

        # Component 4: Sharply angled segment at (10, 48) to (50, 48)
        # Pointing perpendicular (90 deg) -> SHOULD NOT BE HEALED due to angle!
        G.add_node(6, pos=(20, 50), degree_type="endpoint")
        G.add_node(7, pos=(50, 50), degree_type="endpoint")
        G.add_edge(6, 7, weight=300.0)

        # Verify initial components = 4
        self.assertEqual(nx.number_connected_components(G), 4)

        # Execute healing
        healed_G, summary = heal_broken_graph(
            G,
            distance_threshold_px=30.0,
            max_angle_diff_deg=45.0,
            pixel_scale_m=10.0,
        )

        # Assert valid edge (1 <-> 2) was added
        self.assertTrue(healed_G.has_edge(1, 2))
        self.assertTrue(healed_G.edges[1, 2].get("healed", False))

        # Assert implausible bridges were NOT added
        self.assertFalse(healed_G.has_edge(1, 4))
        self.assertFalse(healed_G.has_edge(1, 6))

        # Component count should drop by 1 (from 4 -> 3)
        self.assertEqual(nx.number_connected_components(healed_G), 3)
        self.assertEqual(summary["healed_edges_added"], 1)

    def test_connectivity_metrics(self):
        """Test Connectivity Ratio calculation."""
        g_before = nx.Graph()
        g_before.add_edges_from([(0, 1), (2, 3)])  # 2 components of size 2

        g_after = g_before.copy()
        g_after.add_edge(1, 2)  # 1 component of size 4

        metrics = compute_connectivity_ratio(g_before, g_after)
        self.assertEqual(metrics["largest_component_before"], 2)
        self.assertEqual(metrics["largest_component_after"], 4)
        self.assertEqual(metrics["connectivity_gain_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
