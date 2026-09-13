"""
OSM Ground Truth Road Vector Downloader, Rasterizer & Synthetic Mask Generator.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

from route_resilience.logging_config import setup_logger

logger = setup_logger("data.osm_ground_truth")


def fetch_and_rasterize_osm(
    bbox: Tuple[float, float, float, float],
    target_shape: Tuple[int, int] = (512, 512),
    output_dir: Optional[Path] = None,
    synthetic_fallback: bool = True,
    line_thickness: int = 5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fetches OpenStreetMap road vectors for BBox and rasterizes them to binary mask.

    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon).
        target_shape: (height, width) of output raster mask.
        output_dir: Optional output directory to save mask.
        synthetic_fallback: Whether to use synthetic road generator if OSM fails.
        line_thickness: Thickness in pixels for rasterized road vectors.

    Returns:
        Tuple of (binary ground truth mask [H, W] uint8 (0 or 255), metadata dict).
    """
    mask: Optional[np.ndarray] = None
    source = "osmnx"

    min_lat, min_lon, max_lat, max_lon = bbox

    try:
        logger.info(f"Attempting OSM road network fetch for BBox {bbox}...")
        import osmnx as ox

        # Fetch road graph from OSM
        # Note: osmnx API accepts north, south, east, west
        G = ox.graph_from_bbox(
            bbox=(max_lat, min_lat, max_lon, min_lon),
            network_type="drive",
            simplify=True,
        )
        gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

        h, w = target_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Map geometry coordinates to raster pixel grid
        for _, row in gdf_edges.iterrows():
            geom = row.geometry
            if geom.geom_type == "LineString":
                coords = list(geom.coords)
            elif geom.geom_type == "MultiLineString":
                coords = [pt for line in geom.geoms for pt in line.coords]
            else:
                continue

            pixel_pts = []
            for lon, lat in coords:
                x = int((lon - min_lon) / (max_lon - min_lon) * (w - 1))
                y = int((max_lat - lat) / (max_lat - min_lat) * (h - 1))
                pixel_pts.append((x, y))

            if len(pixel_pts) >= 2:
                pts = np.array(pixel_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=line_thickness)

        logger.info("Successfully fetched and rasterized OSM road vectors.")

    except Exception as e:
        logger.warning(f"OSMnx fetch/rasterization failed ({e}). Fallback to synthetic road mask.")

    if mask is None:
        if not synthetic_fallback:
            raise RuntimeError("OSMnx road vector fetch failed and synthetic fallback is disabled.")
        mask, source = generate_synthetic_road_mask(target_shape, line_thickness=line_thickness)

    metadata = {
        "bbox": bbox,
        "shape": mask.shape,
        "source": source,
        "road_pixel_count": int(np.sum(mask > 0)),
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"osm_mask_{source}.png"
        cv2.imwrite(str(out_file), mask)
        metadata["saved_path"] = str(out_file)
        logger.info(f"Saved OSM ground truth mask to {out_file}")

    return mask, metadata


def generate_synthetic_road_mask(
    shape: Tuple[int, int] = (512, 512),
    line_thickness: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, str]:
    """Generates a realistic synthetic road mask with connected intersections.

    Args:
        shape: Target (height, width) mask shape.
        line_thickness: Thickness of roads in pixels.
        seed: Random seed.

    Returns:
        Tuple of (binary uint8 array [H, W], source string 'synthetic').
    """
    np.random.seed(seed)
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw grid roads (arterial avenues & streets)
    # Main horizontal roads
    y_coords = [int(h * 0.25), int(h * 0.50), int(h * 0.75)]
    for y in y_coords:
        cv2.line(mask, (0, y), (w, y), color=255, thickness=line_thickness + 2)

    # Main vertical roads
    x_coords = [int(w * 0.20), int(w * 0.50), int(w * 0.80)]
    for x in x_coords:
        cv2.line(mask, (x, 0), (x, h), color=255, thickness=line_thickness + 2)

    # Diagonal / curved connecting road
    pts = np.array(
        [
            [int(w * 0.1), int(h * 0.1)],
            [int(w * 0.4), int(h * 0.35)],
            [int(w * 0.7), int(h * 0.85)],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=line_thickness)

    return mask, "synthetic"


def overlay_roads_on_satellite(
    satellite_tile: np.ndarray,
    road_mask: np.ndarray,
    road_color: Tuple[int, int, int] = (200, 200, 200),
    asphalt_darkening: float = 0.6,
) -> np.ndarray:
    """Overlays road network onto synthetic satellite imagery so image & ground truth align.

    Args:
        satellite_tile: RGB uint8 array [H, W, 3].
        road_mask: Binary uint8 mask [H, W] (255 for roads).
        road_color: RGB tint for asphalt road surface.
        asphalt_darkening: Blend ratio for road surface.

    Returns:
        Combined RGB tile array [H, W, 3].
    """
    result = satellite_tile.copy()
    road_indices = road_mask > 0

    # Darken and texture road pixels to look like asphalt in satellite imagery
    road_rgb = np.full_like(result[road_indices], road_color)
    blended = (result[road_indices].astype(np.float32) * (1 - asphalt_darkening) +
               road_rgb.astype(np.float32) * asphalt_darkening).astype(np.uint8)

    result[road_indices] = blended
    return result
