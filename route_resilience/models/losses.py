"""
Loss Functions for Occlusion-Aware Road Segmentation.
Includes Dice Loss, BCE Loss, Boundary-Aware Loss, and Connectivity Penalty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class BoundaryLoss(nn.Module):
    """Boundary-Aware Loss using Sobel gradient filters to penalize edge misalignment."""

    def __init__(self):
        super().__init__()
        # Sobel filters for horizontal & vertical edges
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Compute boundary gradients
        pred_grad_x = F.conv2d(probs, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(probs, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-6)

        target_grad_x = F.conv2d(targets, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(targets, self.sobel_y, padding=1)
        target_edge = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-6)

        return F.l1_loss(pred_edge, target_edge)


class ConnectivityPenaltyLoss(nn.Module):
    """Connectivity Penalty Loss enforcing spatial continuity along road centerlines."""

    def __init__(self):
        super().__init__()
        # Laplacian filter to penalize isolated pixel fragments
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("laplacian", laplacian)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Compute laplacian spatial curvature
        pred_lap = F.conv2d(probs, self.laplacian, padding=1)
        target_lap = F.conv2d(targets, self.laplacian, padding=1)

        # Penalize predictions that break continuity where ground truth target is continuous
        continuity_mask = (target_lap.abs() < 0.5).float() * targets
        penalty = (pred_lap.abs() * continuity_mask).mean()
        return penalty


class OcclusionAwareLoss(nn.Module):
    """Composite loss combining Dice, BCE, Boundary-Aware, and Connectivity penalty."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 0.5,
        boundary_weight: float = 0.2,
        use_connectivity_penalty: bool = True,
        connectivity_weight: float = 0.1,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.use_connectivity_penalty = use_connectivity_penalty
        self.connectivity_weight = connectivity_weight

        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.boundary_loss = BoundaryLoss()
        self.connectivity_loss = ConnectivityPenaltyLoss() if use_connectivity_penalty else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_dice = self.dice_loss(logits, targets)
        l_bce = self.bce_loss(logits, targets)
        l_bound = self.boundary_loss(logits, targets)

        total_loss = (
            self.dice_weight * l_dice
            + self.bce_weight * l_bce
            + self.boundary_weight * l_bound
        )

        if self.use_connectivity_penalty and self.connectivity_loss is not None:
            l_conn = self.connectivity_loss(logits, targets)
            total_loss = total_loss + self.connectivity_weight * l_conn

        return total_loss
