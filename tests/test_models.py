"""
Unit tests for Phase 2 Segmentation Models, Losses, and Metrics.
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.models.unet import UNet
from route_resilience.models.cbam_unet import CBAMUNet, get_model
from route_resilience.models.losses import OcclusionAwareLoss, DiceLoss, BoundaryLoss
from route_resilience.models.metrics import compute_iou_dice, compute_relaxed_iou, compute_occlusion_recall


class TestModelsAndLosses(unittest.TestCase):
    """Test suite for PyTorch segmentation models and loss/metric functions."""

    def test_unet_forward_shape(self):
        """Test baseline U-Net forward pass tensor dimensions."""
        model = UNet(in_channels=3, num_classes=1, init_features=16)
        x = torch.randn(2, 3, 128, 128)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 1, 128, 128))

    def test_cbam_unet_forward_shape(self):
        """Test CBAM attention U-Net forward pass tensor dimensions."""
        model = CBAMUNet(in_channels=3, num_classes=1, init_features=16)
        x = torch.randn(2, 3, 128, 128)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 1, 128, 128))

    def test_factory_get_model(self):
        """Test swappable get_model factory function."""
        m1 = get_model("unet_resnet34")
        self.assertIsInstance(m1, UNet)

        m2 = get_model("cbam_unet")
        self.assertIsInstance(m2, CBAMUNet)

    def test_occlusion_aware_loss_backward(self):
        """Test OcclusionAwareLoss computation and backpropagation gradient flow."""
        criterion = OcclusionAwareLoss(
            dice_weight=1.0,
            bce_weight=0.5,
            boundary_weight=0.2,
            use_connectivity_penalty=True,
            connectivity_weight=0.1,
        )
        logits = torch.randn(2, 1, 64, 64, requires_grad=True)
        targets = (torch.rand(2, 1, 64, 64) > 0.5).float()

        loss = criterion(logits, targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_metrics_computation(self):
        """Test IoU, Dice, Relaxed IoU, and Occlusion Recall calculations."""
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[30:70, 30:70] = 255

        pred = np.zeros((100, 100), dtype=np.float32)
        pred[32:68, 32:68] = 0.9

        iou, dice = compute_iou_dice(pred, gt)
        self.assertGreater(iou, 0.5)
        self.assertGreater(dice, 0.5)

        relaxed_iou = compute_relaxed_iou(pred, gt, tolerance_px=4)
        self.assertGreaterEqual(relaxed_iou, iou)

        occ_mask = np.zeros((100, 100), dtype=np.uint8)
        occ_mask[40:50, 40:50] = 1
        occ_recall = compute_occlusion_recall(pred, gt, occ_mask)
        self.assertGreaterEqual(occ_recall, 0.0)
        self.assertLessEqual(occ_recall, 1.0)


if __name__ == "__main__":
    unittest.main()
