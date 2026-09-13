"""
Phase 1 Data Pipeline Execution Script.
Downloads/generates satellite tile, OSM ground truth mask, patches, and synthetic occlusions.
"""

import sys
from pathlib import Path
import yaml
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.logging_config import setup_logger
from route_resilience.data.sentinel_download import download_sentinel_tile
from route_resilience.data.osm_ground_truth import fetch_and_rasterize_osm, overlay_roads_on_satellite
from route_resilience.data.tiling import extract_patches
from route_resilience.data.augmentation import get_training_augmentation, apply_synthetic_occlusions

logger = setup_logger("scripts.prepare_data")


def prepare_dataset(config_path: Path):
    """Executes end-to-end data pipeline according to YAML configuration.

    Args:
        config_path: Path to default YAML config file.
    """
    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    bbox = tuple(data_cfg["bbox"])
    date_range = tuple(data_cfg["date_range"])
    tile_size = data_cfg["tile_size"]
    overlap = data_cfg["tile_overlap"]
    output_dir = PROJECT_ROOT / data_cfg["output_dir"]

    # Create dataset directories
    train_img_dir = output_dir / "train" / "images"
    train_mask_dir = output_dir / "train" / "masks"
    val_img_dir = output_dir / "val" / "images"
    val_mask_dir = output_dir / "val" / "masks"

    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Sentinel satellite tile fetch / synthetic generation
    logger.info("Step 1: Fetching Sentinel-2 satellite tile...")
    sat_tile, sat_meta = download_sentinel_tile(
        bbox=bbox,
        date_range=date_range,
        max_cloud_cover_pct=data_cfg["max_cloud_cover_pct"],
        synthetic_fallback=data_cfg.get("synthetic_fallback", True),
        target_shape=(512, 512),
    )

    # 2. OSM ground truth road vector fetch & rasterization
    logger.info("Step 2: Fetching & rasterizing OSM ground truth road vectors...")
    osm_mask, osm_meta = fetch_and_rasterize_osm(
        bbox=bbox,
        target_shape=(512, 512),
        synthetic_fallback=data_cfg.get("synthetic_fallback", True),
    )

    # Overlay ground truth road locations onto satellite tile for realistic alignment
    sat_tile = overlay_roads_on_satellite(sat_tile, osm_mask)

    # Save full tiles
    cv2.imwrite(str(output_dir / "full_sat_tile.png"), cv2.cvtColor(sat_tile, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "full_osm_mask.png"), osm_mask)

    # 3. Patching
    logger.info("Step 3: Extracting patches...")
    patches = extract_patches(sat_tile, osm_mask, patch_size=tile_size, overlap=overlap)

    # 4. Augmentation & Occlusion generation
    logger.info("Step 4: Applying synthetic occlusions & saving paired patch samples...")
    aug_fn = get_training_augmentation(enable_occlusion=True)

    split_idx = int(len(patches) * 0.8)
    train_patches = patches[:split_idx]
    val_patches = patches[split_idx:]

    # Save Train Patches (with occlusions)
    for idx, p in enumerate(train_patches):
        img_patch, mask_patch = p["image"], p["mask"]
        # Apply synthetic canopy, cloud & shadow occlusions
        occ_img, aug_mask = aug_fn(img_patch, mask_patch)

        fname = f"patch_{idx:03d}.png"
        cv2.imwrite(str(train_img_dir / fname), cv2.cvtColor(occ_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(train_mask_dir / fname), aug_mask)

    # Save Val Patches (unoccluded clean tiles)
    for idx, p in enumerate(val_patches):
        fname = f"val_patch_{idx:03d}.png"
        cv2.imwrite(str(val_img_dir / fname), cv2.cvtColor(p["image"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(val_mask_dir / fname), p["mask"])

    logger.info("==========================================")
    logger.info(f" Data Pipeline Completed Successfully! ")
    logger.info(f" Train Patches Saved: {len(train_patches)} in {train_img_dir}")
    logger.info(f" Val Patches Saved  : {len(val_patches)} in {val_img_dir}")
    logger.info("==========================================")


if __name__ == "__main__":
    config_file = PROJECT_ROOT / "configs" / "default.yaml"
    prepare_dataset(config_file)
