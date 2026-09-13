"""
Sentinel-2 Satellite Imagery Downloader & Synthetic Generator.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

from route_resilience.logging_config import setup_logger

logger = setup_logger("data.sentinel_download")


def download_sentinel_tile(
    bbox: Tuple[float, float, float, float],
    date_range: Tuple[str, str],
    max_cloud_cover_pct: float = 15.0,
    output_dir: Optional[Path] = None,
    synthetic_fallback: bool = True,
    target_shape: Tuple[int, int] = (512, 512),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fetches Sentinel-2 L2A imagery for a given BBox and date range.

    If SentinelHub API credentials are missing or API fails, falls back
    to generating realistic synthetic satellite imagery tile.

    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon).
        date_range: (start_date, end_date) strings ('YYYY-MM-DD').
        max_cloud_cover_pct: Maximum allowed cloud cover percentage.
        output_dir: Directory to save downloaded/generated tiles.
        synthetic_fallback: Whether to use synthetic tile if API is unavailable.
        target_shape: Target pixel dimensions (height, width) for generated image.

    Returns:
        Tuple of (RGB numpy array uint8 [H, W, 3], metadata dictionary).
    """
    client_id = os.environ.get("SH_CLIENT_ID")
    client_secret = os.environ.get("SH_CLIENT_SECRET")

    image_array: Optional[np.ndarray] = None
    source = "sentinelhub"

    if client_id and client_secret:
        try:
            logger.info("Attempting Sentinel-2 download via SentinelHub API...")
            from sentinelhub import (
                SHConfig,
                SentinelHubRequest,
                DataCollection,
                MimeType,
                BBox,
                CRS,
            )

            config = SHConfig()
            config.sh_client_id = client_id
            config.sh_client_secret = client_secret

            sh_bbox = BBox(bbox=[bbox[1], bbox[0], bbox[3], bbox[2]], crs=CRS.WGS84)

            evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B04", "B03", "B02"],
                    output: { bands: 3 }
                };
            }
            function evaluatePixel(sample) {
                return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
            }
            """

            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL1_IW,
                        time_interval=date_range,
                        maxcc=max_cloud_cover_pct / 100.0,
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
                bbox=sh_bbox,
                size=target_shape,
                config=config,
            )
            images = request.get_data()
            if images and len(images) > 0:
                image_array = (np.clip(images[0], 0, 1) * 255).astype(np.uint8)
                logger.info("Successfully fetched Sentinel-2 tile from SentinelHub API.")
        except Exception as e:
            logger.warning(f"SentinelHub API fetch failed ({e}). Fallback to synthetic tile.")

    if image_array is None:
        if not synthetic_fallback:
            raise RuntimeError("SentinelHub download failed and synthetic fallback is disabled.")
        logger.info("Generating synthetic Sentinel-2 satellite image tile...")
        image_array, source = generate_synthetic_sentinel_tile(target_shape, bbox)

    metadata = {
        "bbox": bbox,
        "date_range": date_range,
        "shape": image_array.shape,
        "source": source,
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"sentinel_tile_{source}.png"
        cv2.imwrite(str(out_file), cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        metadata["saved_path"] = str(out_file)
        logger.info(f"Saved satellite tile to {out_file}")

    return image_array, metadata


def generate_synthetic_sentinel_tile(
    shape: Tuple[int, int] = (512, 512),
    bbox: Optional[Tuple[float, float, float, float]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, str]:
    """Generates a synthetic 3-channel (RGB) satellite imagery tile with terrain textures.

    Args:
        shape: (height, width) tuple.
        bbox: Bounding box coordinates.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (uint8 RGB image array [H, W, 3], source string 'synthetic').
    """
    np.random.seed(seed)
    h, w = shape

    # Create terrain base background: combination of vegetation (greenish) and soil/urban (brownish/grey)
    base_green = np.full((h, w, 3), [34, 112, 45], dtype=np.float32)  # Vegetation green
    base_soil = np.full((h, w, 3), [140, 120, 95], dtype=np.float32)   # Soil / urban grey

    # Perlin-like noise for terrain variance using gaussian blur
    noise = np.random.randn(h // 4, w // 4)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    terrain_mask = 1 / (1 + np.exp(-noise * 3))  # Sigmoid scaling [0, 1]
    terrain_mask_3d = np.repeat(terrain_mask[:, :, np.newaxis], 3, axis=2)

    tile = base_green * terrain_mask_3d + base_soil * (1 - terrain_mask_3d)

    # Add realistic texture noise
    fine_noise = np.random.normal(0, 10, (h, w, 3))
    tile = np.clip(tile + fine_noise, 0, 255).astype(np.uint8)

    return tile, "synthetic"
