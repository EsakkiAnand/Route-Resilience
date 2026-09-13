"""
PyTorch Dataset for Road Segmentation Patches.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from route_resilience.logging_config import setup_logger

logger = setup_logger("models.dataset")


class RoadSegmentationDataset(Dataset):
    """Dataset loading paired satellite image patches and ground truth road masks."""

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        transform=None,
    ):
        """
        Args:
            images_dir: Directory containing satellite image PNG patches.
            masks_dir: Directory containing ground truth binary mask PNG patches.
            transform: Optional callable augmentation/transform function.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        self.image_files = sorted(list(self.images_dir.glob("*.png")))
        if not self.image_files:
            logger.warning(f"No image patches found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask at {mask_path}")

        if self.transform is not None:
            img_rgb, mask = self.transform(img_rgb, mask)

        # Normalize image to [0, 1] float32 tensor [3, H, W]
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0

        # Normalize mask to [0, 1] float32 tensor [1, H, W]
        mask_binary = (mask > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

        return img_tensor, mask_tensor, img_path.name
