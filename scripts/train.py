"""
Training Script for Occlusion-Aware Road Segmentation Model.
Supports swappable baseline (U-Net) and CBAM attention architectures.
"""

import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.logging_config import setup_logger
from route_resilience.models.dataset import RoadSegmentationDataset
from route_resilience.models.cbam_unet import get_model
from route_resilience.models.losses import OcclusionAwareLoss
from route_resilience.models.metrics import compute_iou_dice

logger = setup_logger("scripts.train")


def train_model(config_path: Path, epochs_override: int = None):
    """Trains segmentation model according to configuration settings.

    Args:
        config_path: Path to YAML config file.
        epochs_override: Optional number of epochs to override config setting.
    """
    logger.info(f"Loading training configuration from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = model_cfg["training"]
    loss_cfg = model_cfg["loss"]

    processed_dir = PROJECT_ROOT / data_cfg["output_dir"]
    train_img_dir = processed_dir / "train" / "images"
    train_mask_dir = processed_dir / "train" / "masks"
    val_img_dir = processed_dir / "val" / "images"
    val_mask_dir = processed_dir / "val" / "masks"

    checkpoint_dir = PROJECT_ROOT / train_cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Device configuration with safe CPU fallback
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            # Test simple cuda tensor allocation
            _ = torch.zeros(1).to(device)
            logger.info(f"Using execution device: {device} ({torch.cuda.get_device_name(0)})")
        except Exception as e:
            logger.warning(f"CUDA device unavailable ({e}). Falling back to CPU execution.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logger.info("Using execution device: cpu")

    # Datasets & Dataloaders
    train_dataset = RoadSegmentationDataset(train_img_dir, train_mask_dir)
    val_dataset = RoadSegmentationDataset(val_img_dir, val_mask_dir)

    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Model initialization
    arch_choice = model_cfg["architecture"]
    logger.info(f"Instantiating model architecture: '{arch_choice}'")
    model = get_model(arch_choice, in_channels=model_cfg["in_channels"], num_classes=model_cfg["num_classes"])
    model.to(device)

    # Loss & Optimizer
    criterion = OcclusionAwareLoss(
        dice_weight=loss_cfg["dice_weight"],
        bce_weight=loss_cfg["bce_weight"],
        boundary_weight=loss_cfg["boundary_weight"],
        use_connectivity_penalty=loss_cfg["use_connectivity_penalty"],
        connectivity_weight=loss_cfg["connectivity_weight"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    num_epochs = epochs_override if epochs_override is not None else train_cfg["epochs"]
    best_val_dice = 0.0

    logger.info("Starting model training loop...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_ious, val_dices = [], []

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
                gt_mask = masks.squeeze(0).squeeze(0).cpu().numpy()

                iou, dice = compute_iou_dice(probs, gt_mask)
                val_ious.append(iou)
                val_dices.append(dice)

        val_loss /= max(len(val_dataset), 1)
        mean_val_iou = sum(val_ious) / max(len(val_ious), 1)
        mean_val_dice = sum(val_dices) / max(len(val_dices), 1)

        logger.info(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {mean_val_iou:.4f} | "
            f"Val Dice: {mean_val_dice:.4f}"
        )

        # Save Best Model Checkpoint
        if mean_val_dice >= best_val_dice or epoch == 1:
            best_val_dice = mean_val_dice
            best_ckpt_path = checkpoint_dir / f"best_model_{arch_choice}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": mean_val_dice,
                    "val_iou": mean_val_iou,
                    "architecture": arch_choice,
                },
                best_ckpt_path,
            )
            logger.info(f"Saved new best model checkpoint to {best_ckpt_path}")

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    config_file = PROJECT_ROOT / "configs" / "default.yaml"
    train_model(config_file)
