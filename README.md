# Route Resilience 🛣️🛰️

**Satellite Map Healing & Real-Time Road Network Resilience Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Architecture (Stage A vs. Stage B)

Satellite-based road extraction frequently fails where tree canopy, building shadows, or cloud cover occlude the road (**spectral blindness**), producing broken, disconnected road masks. **Route Resilience** solves this by decoupling map extraction from stress testing:

```
                  ┌──────────────────────────────────────────────────────────┐
                  │                 STAGE A — MAP HEALING                    │
                  └──────────────────────────────────────────────────────────┘
Sentinel-2 L2A  ──► Occlusion-Aware ──► 1-Pixel        ──► Angular & MST   ──► Stage A Graph
Satellite Tile      Segmentation       Skeletonization     Topological         Artifact
& OSM Vectors       Model (U-Net/CBAM) Centerlines         Healing             (.graphml / .json)
                                                                                   │
                                                                                   ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │          STAGE B — REAL-TIME RESILIENCE ANALYSIS         │
                  └──────────────────────────────────────────────────────────┘
Stage A Graph   ──► Weighted           ──► Node-Ablation   ──► Interactive Streamlit
Artifact            Betweenness            Stress Testing      & Folium Heatmap
                    Centrality             (Resilience Index)  Dashboard App
```

- **Stage A — Map Healing (Foundation):** Extracts roads from satellite imagery and topologically heals broken masks into a single connected, weighted NetworkX graph artifact (`.graphml` / `.json`).
- **Stage B — Real-Time Resilience Analysis (Value Add):** Consumes Stage A graph artifacts to rank critical **Gatekeeper Nodes** and run live road closure simulations quantifying **Resilience Index**, travel time delays, and severed routes.

---

## 🚀 Fresh-Clone-to-Running-Dashboard Walkthrough

Follow these exact steps from a fresh clone to run the entire platform and launch the interactive dashboard:

### 1. Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/organization/route-resilience.git
cd "route-resilience"

# 2. Install package in editable mode
pip install -e .
```

### 2. Optional: Sentinel-2 API Credentials

If you have a SentinelHub account, set your credentials as environment variables:
```bash
# Windows PowerShell
$env:SH_CLIENT_ID="your_client_id"
$env:SH_CLIENT_SECRET="your_client_secret"

# Linux / MacOS
export SH_CLIENT_ID="your_client_id"
export SH_CLIENT_SECRET="your_client_secret"
```
> **Note (Offline / Offline Fallback):** If API keys are not supplied, the pipeline automatically generates realistic synthetic satellite tiles and road vectors, ensuring 100% offline runnability without external API dependencies.

---

## 🏃 Running the Full End-to-End Pipeline

To execute all phases sequentially (Data → Segment → Heal → Analyze → Dashboard):

```bash
python run_pipeline.py --launch-dashboard
```

---

## 🔬 Running Individual Phases Independently

Each phase can be run and verified independently:

### Phase 0: Project Scaffolding Smoke Test
```bash
python scripts/smoke_test.py
```

### Phase 1: Data Pipeline & Occlusion Augmentation
```bash
# Downloads/generates tiles, extracts patches, applies occlusions & creates visualization plot
python scripts/prepare_data.py
python scripts/visualize_samples.py
```
*Output plot:* `data/processed/sample_visualization.png`

### Phase 2: Segmentation Model Training & Evaluation
```bash
# Trains occlusion-aware U-Net and generates evaluation report
python scripts/train.py
python scripts/evaluate.py
```
*Output mask:* `data/processed/predicted_eval_mask.png`

### Phase 3: Skeletonization & Topological Map Healing (Stage A)
```bash
# Converts binary mask to skeleton and applies Union-Find MST healing
python scripts/heal_map.py
```
*Output graph artifacts:* `graph/outputs/healed_stage_a_graph.graphml` & `graph/outputs/healed_stage_a_graph.json`

### Phase 4: Structural Intelligence & Stress Testing (Stage B)
```bash
# Computes Betweenness Centrality and simulates node closure perturbation
python scripts/analyze_resilience.py --top-gatekeeper
```

### Phase 5: Interactive Web Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🧪 Running Unit Tests

Run the full unit test suite covering package imports, data pipeline, PyTorch segmentation models, graph healing logic, and resilience math:

```bash
python -m unittest discover -s tests
```

---

## 📁 Repository Structure

```
Route Resilience/
├── configs/             # YAML configurations (default.yaml)
├── route_resilience/    # Core Python package
│   ├── data/            # Sentinel-2 download, OSM rasterization & synthetic occlusion
│   ├── models/          # U-Net & CBAM attention segmentation models, losses & metrics
│   ├── graph/           # Skeletonization, MST healing, centrality & ablation algorithms
│   └── dashboard/       # Interactive Streamlit + Folium dashboard application
├── dashboard/           # Streamlit app entry point
├── data/                # Local data storage (processed patches & visualizations)
├── models/              # Saved model checkpoints
├── graph/               # Serialized Stage A graph outputs (.graphml / .json)
├── notebooks/           # Evaluation & benchmark reports (evaluation_report.md)
├── scripts/             # CLI execution scripts for each phase
├── tests/               # Unit test suites (test_smoke, test_data, test_models, test_graph, test_resilience)
├── run_pipeline.py      # End-to-end pipeline orchestrator
├── requirements.txt     # Dependency declarations
├── environment.yml      # Conda environment specification
├── setup.py             # Package installation setup
└── README.md            # Repository documentation
```
