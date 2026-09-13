"""
Model Evaluation Script for Phase 2.
Computes Overall IoU/Dice, Occlusion-Recall, and Relaxed IoU, saving predicted mask image.
"""

import sys
from pathlib import Path
import yaml
import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.logging_config import setup_logger
from route_resilience.models.cbam_unet import get_model
from route_resilience.models.metrics import compute_iou_dice, compute_relaxed_iou, compute_occlusion_recall
from route_resilience.data.augmentation import apply_synthetic_occlusions

logger = setup_logger("scripts.evaluate")


def evaluate_model(config_path: Path):
    """Evaluates trained segmentation model checkpoint and outputs evaluation report.

    Args:
        config_path: Path to YAML configuration file.
    """
    logger.info(f"Loading configuration from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    arch_choice = model_cfg["architecture"]
    ckpt_dir = PROJECT_ROOT / model_cfg["training"]["checkpoint_dir"]
    ckpt_path = ckpt_dir / f"best_model_{arch_choice}.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Please train model first.")

    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            _ = torch.zeros(1).to(device)
            logger.info(f"Using execution device: {device}")
        except Exception as e:
            logger.warning(f"CUDA device unavailable ({e}). Falling back to CPU execution.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logger.info("Using execution device: cpu")

    model = get_model(arch_choice, in_channels=model_cfg["in_channels"], num_classes=model_cfg["num_classes"])
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load held-out full test tile and OSM ground truth mask
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    sat_path = processed_dir / "full_sat_tile.png"
    mask_path = processed_dir / "full_osm_mask.png"

    if not sat_path.exists() or not mask_path.exists():
        raise FileNotFoundError("Full satellite tile or OSM mask missing. Run prepare_data.py first.")

    sat_bgr = cv2.imread(str(sat_path))
    sat_rgb = cv2.cvtColor(sat_bgr, cv2.COLOR_BGR2RGB)
    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Generate synthetic occlusion mask for evaluation of Occlusion-Recall
    h, w = gt_mask.shape
    occ_tile = apply_synthetic_occlusions(sat_rgb, shadow_count=3, canopy_count=2, cloud_count=1, seed=999)
    # Create explicit occlusion mask map
    diff = np.abs(sat_rgb.astype(np.float32) - occ_tile.astype(np.float32)).mean(axis=2)
    occlusion_mask = (diff > 15).astype(np.uint8)

    # Perform Model Inference
    img_tensor = torch.from_numpy(occ_tile.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

    pred_binary = (probs > 0.5).astype(np.uint8) * 255

    # Compute Metrics
    overall_iou, overall_dice = compute_iou_dice(probs, gt_mask)
    relaxed_iou = compute_relaxed_iou(probs, gt_mask, tolerance_px=4)
    occlusion_recall = compute_occlusion_recall(probs, gt_mask, occlusion_mask)

    # Save Predicted Mask Image for visual inspection
    out_pred_path = processed_dir / "predicted_eval_mask.png"
    cv2.imwrite(str(out_pred_path), pred_binary)

    # Save Side-by-Side Comparison Image
    comparison = np.hstack([occ_tile, cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2RGB), cv2.cvtColor(pred_binary, cv2.COLOR_GRAY2RGB)])
    out_cmp_path = processed_dir / "evaluation_comparison.png"
    cv2.imwrite(str(out_cmp_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    # Print Evaluation Report
    report = f"""
==================================================
        ROUTE RESILIENCE MODEL EVALUATION REPORT  
==================================================
Model Architecture  : {arch_choice}
Checkpoint Path     : {ckpt_path.name}
Evaluated Tile Shape: {h}x{w}
--------------------------------------------------
Overall IoU (Jaccard) : {overall_iou:.4f}
Overall Dice (F1)     : {overall_dice:.4f}
Relaxed IoU (4px buf) : {relaxed_iou:.4f}
Occlusion-Recall      : {occlusion_recall:.4f}
--------------------------------------------------
Predicted Mask Saved  : {out_pred_path}
Comparison Plot Saved : {out_cmp_path}
==================================================
"""
    logger.info(report)
    print(report)

    return {
        "overall_iou": overall_iou,
        "overall_dice": overall_dice,
        "relaxed_iou": relaxed_iou,
        "occlusion_recall": occlusion_recall,
        "pred_mask_path": str(out_pred_path),
    }


if __name__ == "__main__":
    config_file = PROJECT_ROOT / "configs" / "default.yaml"
    evaluate_model(config_file)
