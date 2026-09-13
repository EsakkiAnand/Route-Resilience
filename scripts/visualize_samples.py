"""
Visualization script for Phase 1 Data Pipeline sanity check.
Plots 5 random paired (Image, Occluded Image, Mask) samples side by side.
"""

import sys
import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.logging_config import setup_logger
from route_resilience.data.augmentation import apply_synthetic_occlusions

logger = setup_logger("scripts.visualize_samples")


def visualize_paired_samples(num_samples: int = 5):
    """Plots and saves side-by-side visualization of sample image/mask pairs.

    Args:
        num_samples: Number of random sample rows to plot (default 5).
    """
    train_img_dir = PROJECT_ROOT / "data" / "processed" / "train" / "images"
    train_mask_dir = PROJECT_ROOT / "data" / "processed" / "train" / "masks"
    out_vis_path = PROJECT_ROOT / "data" / "processed" / "sample_visualization.png"

    img_files = sorted(list(train_img_dir.glob("*.png")))
    if not img_files:
        raise FileNotFoundError(
            f"No training image patches found in {train_img_dir}. "
            "Please run scripts/prepare_data.py first."
        )

    # Select random sample files
    sample_files = random.sample(img_files, min(num_samples, len(img_files)))

    fig, axes = plt.subplots(len(sample_files), 3, figsize=(12, 3.5 * len(sample_files)))
    if len(sample_files) == 1:
        axes = [axes]

    for idx, img_path in enumerate(sample_files):
        mask_path = train_mask_dir / img_path.name

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Plot 1: Occluded Satellite Tile (Input)
        axes[idx][0].imshow(img_rgb)
        axes[idx][0].set_title(f"Sample {idx+1}: Occluded Input Satellite", fontsize=10)
        axes[idx][0].axis("off")

        # Plot 2: Road Mask Overlaid on Tile
        overlay = img_rgb.copy()
        overlay[mask > 0] = [255, 50, 50]  # Highlight roads in bright red
        axes[idx][1].imshow(overlay)
        axes[idx][1].set_title(f"Sample {idx+1}: Satellite + Ground Truth Road Overlay", fontsize=10)
        axes[idx][1].axis("off")

        # Plot 3: Ground Truth Binary Mask
        axes[idx][2].imshow(mask, cmap="gray")
        axes[idx][2].set_title(f"Sample {idx+1}: Binary Road Mask (0 / 255)", fontsize=10)
        axes[idx][2].axis("off")

    plt.tight_layout()
    out_vis_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_vis_path, dpi=150)
    plt.close()

    logger.info("==========================================")
    logger.info(f" Sample visualization successfully generated!")
    logger.info(f" Saved visualization to: {out_vis_path}")
    logger.info("==========================================")


if __name__ == "__main__":
    visualize_paired_samples(5)
