"""
Phase 0 Smoke Test Script for Route Resilience.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import route_resilience
from route_resilience.logging_config import setup_logger

def run_smoke_test():
    print("==========================================")
    print("   ROUTE RESILIENCE - PHASE 0 SMOKE TEST  ")
    print("==========================================")
    
    # 1. Check version
    print(f"[OK] Package version: {route_resilience.__version__}")
    
    # 2. Check logger
    logger = setup_logger("smoke_test", log_level="INFO")
    logger.info("Logger initialized successfully!")
    
    # 3. Check config
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config missing at {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    print(f"[OK] Default YAML config loaded. Target bbox: {config['data']['bbox']}")
    print(f"[OK] Model arch choice: {config['model']['architecture']}")
    print(f"[OK] Dashboard port: {config['dashboard']['port']}")
    print("==========================================")
    print(" Phase 0 Verification PASSED Successfully ")
    print("==========================================")

if __name__ == "__main__":
    run_smoke_test()
