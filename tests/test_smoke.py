"""
Smoke test suite for Route Resilience project scaffolding (unittest compatible).
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import route_resilience
from route_resilience.logging_config import setup_logger


class TestPhase0Scaffolding(unittest.TestCase):
    """Smoke test cases for Phase 0."""

    def test_package_import(self):
        """Verify route_resilience package is importable and has version."""
        self.assertTrue(hasattr(route_resilience, "__version__"))
        self.assertEqual(route_resilience.__version__, "0.1.0")

    def test_logger_setup(self):
        """Verify logger setup utility operates without error."""
        logger = setup_logger(name="test_logger", log_level="DEBUG")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_logger")

    def test_default_config_loading(self):
        """Verify configs/default.yaml exists and is valid YAML."""
        config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        self.assertTrue(config_path.exists(), f"Config file not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.assertIn("project", config)
        self.assertIn("data", config)
        self.assertIn("model", config)
        self.assertIn("healing", config)
        self.assertIn("analysis", config)
        self.assertIn("dashboard", config)
        self.assertEqual(config["data"]["tile_size"], 256)


if __name__ == "__main__":
    unittest.main()
