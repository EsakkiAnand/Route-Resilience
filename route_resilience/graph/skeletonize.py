"""
Binary Mask Skeletonization & Centerline Extraction.
Supports scikit-image skeletonize with OpenCV morphological thinning fallback.
"""

import logging
import numpy as np
import cv2

try:
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from route_resilience.logging_config import setup_logger

logger = setup_logger("graph.skeletonize")


def opencv_skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """Morphological skeletonization using standard OpenCV erosion & dilation.

    Args:
        binary_mask: uint8 binary mask [H, W] (0 or 255).

    Returns:
        1-pixel wide skeleton uint8 array [H, W] (0 or 255).
    """
    img = binary_mask.copy()
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel


def mask_to_skeleton(
    binary_mask: np.ndarray,
    min_component_size: int = 15,
) -> np.ndarray:
    """Converts a binary road segmentation mask into a 1-pixel wide skeleton centerline.

    Args:
        binary_mask: Binary road mask array [H, W] uint8 (0 or 255).
        min_component_size: Minimum pixel area to keep (filters isolated noise blobs).

    Returns:
        1-pixel wide skeleton centerline array [H, W] uint8 (0 or 255).
    """
    if binary_mask.dtype != np.uint8:
        binary_mask = (binary_mask > 0.5).astype(np.uint8) * 255

    # Filter small noise blobs via connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
            cleaned_mask[labels == i] = 255

    if HAS_SKIMAGE:
        bool_mask = cleaned_mask > 0
        skel_bool = skeletonize(bool_mask)
        skeleton_uint8 = (skel_bool.astype(np.uint8)) * 255
    else:
        logger.info("scikit-image not found; utilizing OpenCV morphological skeletonization fallback.")
        skeleton_uint8 = opencv_skeletonize(cleaned_mask)

    logger.info(f"Extracted skeleton centerline ({np.sum(skeleton_uint8 > 0)} 1-pixel road points).")

    return skeleton_uint8
