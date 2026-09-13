"""
End-to-End Pipeline Orchestrator for Route Resilience.
Executes Phase 1 (Data) -> Phase 2 (Segmentation) -> Phase 3 (Healing) -> Phase 4 (Resilience Analysis) -> Phase 5 (Dashboard).
"""

import sys
import argparse
from pathlib import Path
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.logging_config import setup_logger
from scripts.prepare_data import prepare_dataset
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.heal_map import run_map_healing
from scripts.analyze_resilience import run_resilience_analysis

logger = setup_logger("run_pipeline")


def run_full_pipeline(
    config_path: Path,
    epochs: int = 2,
    skip_training: bool = False,
    launch_dashboard: bool = False,
):
    """Executes full Route Resilience pipeline top to bottom.

    Args:
        config_path: Path to default YAML config file.
        epochs: Number of epochs to train segmentation model.
        skip_training: Whether to skip training if checkpoint exists.
        launch_dashboard: Whether to launch Streamlit dashboard at the end.
    """
    logger.info("==================================================")
    logger.info("      STARTING ROUTE RESILIENCE PIPELINE         ")
    logger.info("==================================================")

    # Step 1: Data Pipeline (Phase 1)
    logger.info("--- PHASE 1: Data Pipeline & Occlusion Augmentation ---")
    prepare_dataset(config_path)

    # Step 2: Segmentation Model (Phase 2)
    logger.info("--- PHASE 2: Occlusion-Aware Segmentation Model ---")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    arch_choice = config["model"]["architecture"]
    ckpt_path = PROJECT_ROOT / config["model"]["training"]["checkpoint_dir"] / f"best_model_{arch_choice}.pth"

    if not skip_training or not ckpt_path.exists():
        train_model(config_path, epochs_override=epochs)
    else:
        logger.info(f"Skipping training (Existing checkpoint found at {ckpt_path})")

    eval_results = evaluate_model(config_path)

    # Step 3: Skeletonization & Map Healing (Phase 3 - Stage A)
    logger.info("--- PHASE 3: Skeletonization & Topological Map Healing ---")
    healed_graph, conn_metrics, topo_metrics = run_map_healing(config_path)

    # Step 4: Structural Intelligence & Stress Testing (Phase 4 - Stage B)
    logger.info("--- PHASE 4: Structural Intelligence & Stress Testing ---")
    resilience_results = run_resilience_analysis(config_path=config_path)

    logger.info("==================================================")
    logger.info("     ROUTE RESILIENCE PIPELINE COMPLETED!        ")
    logger.info("==================================================")
    logger.info(f" Stage A Connectivity Gain Ratio: {conn_metrics['connectivity_gain_ratio']:.2f}x")
    logger.info(f" Stage B Resilience Index       : {resilience_results['resilience_index']:.4f}")
    logger.info(f" Stage B Travel Time Impact     : +{resilience_results['travel_time_impact_pct']:.1f}%")
    logger.info("==================================================")

    if launch_dashboard:
        logger.info("Launching Streamlit Dashboard Application...")
        import subprocess

        cmd = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]
        subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Route Resilience Pipeline Orchestrator")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs for quick pipeline training")
    parser.add_argument("--skip-training", action="store_true", help="Skip training if checkpoint exists")
    parser.add_argument("--launch-dashboard", action="store_true", help="Launch Streamlit dashboard after pipeline")

    args = parser.parse_args()
    cfg_file = PROJECT_ROOT / args.config
    run_full_pipeline(
        cfg_file,
        epochs=args.epochs,
        skip_training=args.skip_training,
        launch_dashboard=args.launch_dashboard,
    )
