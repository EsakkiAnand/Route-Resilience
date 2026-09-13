"""
Data Augmentation & Synthetic Occlusion Pipeline (Canopy, Shadow, Cloud).
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

from route_resilience.logging_config import setup_logger

logger = setup_logger("data.augmentation")


def apply_synthetic_occlusions(
    image: np.ndarray,
    shadow_count: int = 3,
    canopy_count: int = 2,
    cloud_count: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Applies realistic synthetic spectral blindness occlusions onto satellite imagery.

    Synthetic occlusions affect ONLY the input satellite image (occluding the road underneath),
    leaving the ground truth road mask untouched.

    Args:
        image: RGB uint8 numpy array [H, W, 3].
        shadow_count: Number of dark building/cloud shadow polygons to render.
        canopy_count: Number of blurred green tree canopy blobs to render over roads.
        cloud_count: Number of semi-transparent white cloud patches to render.
        seed: Optional random seed.

    Returns:
        Occluded RGB image uint8 array [H, W, 3].
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = image.shape[:2]
    occluded = image.copy().astype(np.float32)

    # 1. Shadow Polygons (darkening)
    for _ in range(shadow_count):
        pts_count = np.random.randint(3, 7)
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(20, min(h, w) // 4)
        angles = np.sort(np.random.uniform(0, 2 * np.pi, pts_count))

        pts = []
        for angle in angles:
            r = radius * np.random.uniform(0.6, 1.2)
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))
            pts.append([px, py])

        shadow_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(shadow_mask, [np.array(pts, dtype=np.int32)], 1.0)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)

        shadow_factor = np.random.uniform(0.25, 0.55)
        for c in range(3):
            occluded[:, :, c] = occluded[:, :, c] * (1.0 - shadow_mask * (1.0 - shadow_factor))

    # 2. Green Tree Canopy Blobs (foliage occlusions)
    for _ in range(canopy_count):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        rx, ry = np.random.randint(25, 60), np.random.randint(25, 60)
        angle = np.random.randint(0, 180)

        canopy_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(canopy_mask, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)
        canopy_mask = cv2.GaussianBlur(canopy_mask, (31, 31), 0)

        canopy_color = np.array([25, np.random.randint(90, 130), 30], dtype=np.float32)
        for c in range(3):
            occluded[:, :, c] = occluded[:, :, c] * (1.0 - canopy_mask) + canopy_color[c] * canopy_mask

    # 3. Cloud Patches (white semi-transparent clouds)
    for _ in range(cloud_count):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(40, min(h, w) // 3)

        cloud_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(cloud_mask, (cx, cy), radius, 1.0, -1)
        cloud_mask = cv2.GaussianBlur(cloud_mask, (51, 51), 0)

        cloud_opacity = np.random.uniform(0.6, 0.9)
        cloud_color = np.array([240, 245, 250], dtype=np.float32)
        for c in range(3):
            occluded[:, :, c] = (
                occluded[:, :, c] * (1.0 - cloud_mask * cloud_opacity)
                + cloud_color[c] * (cloud_mask * cloud_opacity)
            )

    return np.clip(occluded, 0, 255).astype(np.uint8)


def get_training_augmentation(
    enable_occlusion: bool = True,
    occlusion_params: Optional[Dict[str, int]] = None,
):
    """Creates a data augmentation transformation pipeline for training samples.

    Args:
        enable_occlusion: Config flag to toggle synthetic occlusion generation.
        occlusion_params: Dict with 'shadow_count', 'canopy_count', 'cloud_count'.

    Returns:
        Callable function accepting (image, mask) dict and returning transformed (image, mask).
    """

    def transform_fn(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        aug_image = image.copy()
        aug_mask = mask.copy()

        # Apply synthetic occlusions if enabled
        if enable_occlusion:
            params = occlusion_params or {"shadow_count": 2, "canopy_count": 2, "cloud_count": 1}
            aug_image = apply_synthetic_occlusions(aug_image, **params)

        # Apply spatial/geometric Albumentations if available
        if HAS_ALBUMENTATIONS:
            pipeline = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                ]
            )
            res = pipeline(image=aug_image, mask=aug_mask)
            aug_image, aug_mask = res["image"], res["mask"]

        return aug_image, aug_mask

    return transform_fn
