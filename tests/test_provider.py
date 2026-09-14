import unittest

import networkx as nx

from route_resilience.graph.provider import graph_summary, node_location, validate_location


class TestGraphProvider(unittest.TestCase):
    def test_location_validation_rejects_invalid_radius(self):
        with self.assertRaises(ValueError):
            validate_location(12.9716, 77.5946, 50)

    def test_summary_and_node_location(self):
        graph = nx.Graph()
        graph.add_node(1, y=12.97, x=77.59)
        graph.add_node(2, y=12.98, x=77.60)
        graph.add_edge(1, 2, weight=100.0)

        self.assertEqual(graph_summary(graph), {"nodes": 2, "edges": 1, "components": 1})
        self.assertEqual(node_location(graph, 1), (12.97, 77.59))


if __name__ == "__main__":
    unittest.main()
