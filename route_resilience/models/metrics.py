"""
Evaluation Metrics for Road Segmentation (IoU, Dice, Occlusion-Recall, Relaxed IoU).
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2
import torch


def compute_iou_dice(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """Computes overall IoU and Dice score for binary predicted mask vs target mask.

    Args:
        pred_mask: Float/Binary array [H, W] (probs [0, 1] or uint8 {0, 255}).
        target_mask: Binary array [H, W] (0 or 255/1).
        threshold: Binarization probability threshold.

    Returns:
        Tuple of (IoU: float, Dice: float).
    """
    pred_bin = (pred_mask > (threshold if pred_mask.max() <= 1.0 else threshold * 255)).astype(np.uint8)
    target_bin = (target_mask > 0).astype(np.uint8)

    intersection = np.sum((pred_bin == 1) & (target_bin == 1))
    union = np.sum((pred_bin == 1) | (target_bin == 1))
    pred_sum = np.sum(pred_bin == 1)
    target_sum = np.sum(target_bin == 1)

    iou = float(intersection) / float(max(union, 1))
    dice = float(2.0 * intersection) / float(max(pred_sum + target_sum, 1))

    return iou, dice


def compute_relaxed_iou(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    tolerance_px: int = 4,
    threshold: float = 0.5,
) -> float:
    """Computes Relaxed IoU with a configurable pixel tolerance buffer around ground truth.

    Args:
        pred_mask: Predicted mask [H, W].
        target_mask: Target ground truth mask [H, W].
        tolerance_px: Pixel distance tolerance buffer (default 4px).
        threshold: Binarization threshold.

    Returns:
        Relaxed IoU score float in [0.0, 1.0].
    """
    pred_bin = (pred_mask > (threshold if pred_mask.max() <= 1.0 else threshold * 255)).astype(np.uint8)
    target_bin = (target_mask > 0).astype(np.uint8)

    # Dilate target mask by tolerance buffer
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1))
    relaxed_target = cv2.dilate(target_bin, kernel)
    relaxed_pred = cv2.dilate(pred_bin, kernel)

    # Precision on relaxed target & Recall on relaxed pred
    relaxed_intersection = np.sum((pred_bin == 1) & (relaxed_target == 1))
    union = np.sum((pred_bin == 1) | (target_bin == 1))

    relaxed_iou = float(relaxed_intersection) / float(max(union, 1))
    return min(relaxed_iou, 1.0)


def compute_occlusion_recall(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    occlusion_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Computes Recall/IoU specifically within synthetically-occluded ground truth regions.

    Args:
        pred_mask: Predicted mask [H, W].
        target_mask: Ground truth road mask [H, W].
        occlusion_mask: Binary mask [H, W] indicating occluded pixels (shadow/canopy/cloud).
        threshold: Binarization threshold.

    Returns:
        Occlusion Recall score float in [0.0, 1.0].
    """
    pred_bin = (pred_mask > (threshold if pred_mask.max() <= 1.0 else threshold * 255)).astype(np.uint8)
    target_bin = (target_mask > 0).astype(np.uint8)
    occ_bin = (occlusion_mask > 0).astype(np.uint8)

    # Focus on ground truth road pixels that fall inside occluded regions
    target_occluded_roads = (target_bin == 1) & (occ_bin == 1)
    if np.sum(target_occluded_roads) == 0:
        return 1.0  # Default to 1.0 if no occluded roads present in sample

    correctly_detected_occluded = np.sum((pred_bin == 1) & target_occluded_roads)
    occlusion_recall = float(correctly_detected_occluded) / float(np.sum(target_occluded_roads))

    return occlusion_recall
