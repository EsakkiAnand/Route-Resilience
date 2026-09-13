"""
Unit tests for Phase 1 Data Pipeline (Sentinel download, OSM ground truth, Tiling, Augmentation).
"""

import unittest
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.data.sentinel_download import download_sentinel_tile, generate_synthetic_sentinel_tile
from route_resilience.data.osm_ground_truth import fetch_and_rasterize_osm, generate_synthetic_road_mask
from route_resilience.data.tiling import extract_patches, stitch_patches
from route_resilience.data.augmentation import apply_synthetic_occlusions, get_training_augmentation


class TestDataPipeline(unittest.TestCase):
    """Test suite for Data Pipeline functionality."""

    def test_synthetic_sentinel_tile_generation(self):
        """Test synthetic satellite imagery generator output format."""
        tile, source = generate_synthetic_sentinel_tile(shape=(256, 256), seed=123)
        self.assertEqual(source, "synthetic")
        self.assertEqual(tile.shape, (256, 256, 3))
        self.assertEqual(tile.dtype, np.uint8)

    def test_synthetic_road_mask_generation(self):
        """Test synthetic ground truth road mask generator output format."""
        mask, source = generate_synthetic_road_mask(shape=(256, 256), seed=123)
        self.assertEqual(source, "synthetic")
        self.assertEqual(mask.shape, (256, 256))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater(np.sum(mask > 0), 0)

    def test_tiling_patch_extraction_and_stitching(self):
        """Test splitting full tile into patches and stitching back."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        mask = np.ones((512, 512), dtype=np.uint8) * 255

        patches = extract_patches(image, mask, patch_size=256, overlap=32)
        # For 512x512 with patch=256, overlap=32, step=224:
        # y starts at 0, 224, 256 (3 positions) x 3 positions = 9 patches
        self.assertGreater(len(patches), 0)

        for p in patches:
            self.assertEqual(p["image"].shape, (256, 256, 3))
            self.assertEqual(p["mask"].shape, (256, 256))

        stitched = stitch_patches(patches, target_shape=(512, 512), patch_size=256)
        self.assertEqual(stitched.shape, (512, 512))

    def test_synthetic_occlusions(self):
        """Test applying canopy, cloud, and shadow occlusions to satellite tile."""
        image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        occluded = apply_synthetic_occlusions(image, shadow_count=2, canopy_count=1, cloud_count=1, seed=42)

        self.assertEqual(occluded.shape, (256, 256, 3))
        self.assertEqual(occluded.dtype, np.uint8)
        self.assertFalse(np.array_equal(image, occluded))


if __name__ == "__main__":
    unittest.main()
