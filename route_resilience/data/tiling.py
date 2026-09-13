"""
Tile Patching and Reconstruction Utilities.
"""

import logging
from typing import List, Tuple, Dict, Any
import numpy as np

from route_resilience.logging_config import setup_logger

logger = setup_logger("data.tiling")


def extract_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    overlap: int = 32,
) -> List[Dict[str, Any]]:
    """Splits full tile image and mask into fixed-size overlapping patches.

    Args:
        image: RGB tile numpy array [H, W, 3].
        mask: Binary ground truth mask [H, W].
        patch_size: Square tile dimension (default 256).
        overlap: Overlap in pixels between adjacent patches (default 32).

    Returns:
        List of patch dictionary objects containing:
        - 'image': [patch_size, patch_size, 3]
        - 'mask': [patch_size, patch_size]
        - 'y_start': int
        - 'x_start': int
        - 'patch_id': str
    """
    h, w = image.shape[:2]
    step = patch_size - overlap

    patches = []
    patch_idx = 0

    y_starts = list(range(0, h - patch_size + 1, step))
    if y_starts[-1] + patch_size < h:
        y_starts.append(h - patch_size)

    x_starts = list(range(0, w - patch_size + 1, step))
    if x_starts[-1] + patch_size < w:
        x_starts.append(w - patch_size)

    for y in y_starts:
        for x in x_starts:
            img_patch = image[y : y + patch_size, x : x + patch_size].copy()
            mask_patch = mask[y : y + patch_size, x : x + patch_size].copy()

            patches.append(
                {
                    "image": img_patch,
                    "mask": mask_patch,
                    "y_start": y,
                    "x_start": x,
                    "patch_id": f"patch_y{y}_x{x}",
                }
            )
            patch_idx += 1

    logger.info(
        f"Extracted {len(patches)} patches of size {patch_size}x{patch_size} "
        f"(overlap={overlap}px) from tile of shape ({h}, {w})."
    )
    return patches


def stitch_patches(
    patches: List[Dict[str, Any]],
    target_shape: Tuple[int, int],
    patch_size: int = 256,
) -> np.ndarray:
    """Stitches predicted mask patches back into full tile prediction using linear blend.

    Args:
        patches: List of patch dictionaries with keys 'mask' (or 'pred_mask'), 'y_start', 'x_start'.
        target_shape: (height, width) of target stitched full tile.
        patch_size: Size of square patch.

    Returns:
        Stitched prediction array float32 [H, W] (unnormalized probability or binary).
    """
    h, w = target_shape
    stitched_sum = np.zeros((h, w), dtype=np.float32)
    stitched_count = np.zeros((h, w), dtype=np.float32)

    for p in patches:
        y, x = p["y_start"], p["x_start"]
        patch_mask = p.get("pred_mask", p["mask"]).astype(np.float32)

        stitched_sum[y : y + patch_size, x : x + patch_size] += patch_mask
        stitched_count[y : y + patch_size, x : x + patch_size] += 1.0

    # Avoid division by zero
    stitched_count = np.maximum(stitched_count, 1.0)
    stitched = stitched_sum / stitched_count
    return stitched
