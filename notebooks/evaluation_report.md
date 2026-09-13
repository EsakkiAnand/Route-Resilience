# Route Resilience — Consolidated Evaluation & Benchmark Report

This document consolidates quantitative evaluation metrics across Stage A (Map Healing) and Stage B (Structural Intelligence & Resilience Stress Testing).

---

## 1. Stage A: Occlusion-Aware Segmentation Performance

Evaluated on held-out 512×512 satellite tiles subject to synthetic spectral blindness occlusions (shadow polygons, green tree canopy blobs, and cloud cover patches).

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Overall IoU (Jaccard)** | `0.0000` | Baseline initial IoU score across unthinned raw tile |
| **Overall Dice (F1)** | `0.0000` | Baseline F1 harmonic mean |
| **Relaxed IoU (4px Buffer)** | `0.0000` | IoU with 4-pixel spatial tolerance buffer around ground truth |
| **Occlusion-Recall** | `0.0000` | Recall specifically within occluded road pixels |

*Saved Prediction Artifact:* [`data/processed/predicted_eval_mask.png`](file:///f:/project/Route%20Resilience/data/processed/predicted_eval_mask.png)  
*Side-by-Side Comparison Plot:* [`data/processed/evaluation_comparison.png`](file:///f:/project/Route%20Resilience/data/processed/evaluation_comparison.png)

---

## 2. Stage A: Topological Map Healing Metrics

Evaluated by converting binary segmentations to 1-pixel skeleton centerlines and applying trajectory angular alignment ($<45^\circ$) + Union-Find Kruskal's MST edge insertion.

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Graph Nodes Count** | `246` | Total intersection and endpoint nodes |
| **Graph Edges Count** | `272` | Total road segments |
| **Components (Before Healing)** | `122` | Fragmented components from spectral blindness |
| **Components (After Healing)** | `52` | Merged components after MST healing |
| **Healing Edges Added** | `119` | Valid trajectory-aligned bridges inserted |
| **Connectivity Gain Ratio** | `10.33x` | Ratio of largest component size after vs before |
| **Connectivity Coverage** | `75.6%` | Largest component node coverage |
| **Topological Accuracy Error** | `28.95%` | Shortest path error vs OpenStreetMap ground truth |
| **Reachable Pairs Percentage** | `58.0%` | Origin-Destination reachable path percentage |

*Serialized Stage A Artifacts:*
- GraphML: [`graph/outputs/healed_stage_a_graph.graphml`](file:///f:/project/Route%20Resilience/graph/outputs/healed_stage_a_graph.graphml)
- GeoJSON: [`graph/outputs/healed_stage_a_graph.json`](file:///f:/project/Route%20Resilience/graph/outputs/healed_stage_a_graph.json)

---

## 3. Stage B: Structural Resilience & Node Ablation Stress Testing

Evaluated by computing weighted Betweenness Centrality and performing node-ablation perturbation simulations over 100 OD pairs.

### Top-5 Gatekeeper Junctions (Critical Vulnerability Nodes)

| Rank | Node ID | Betweenness Centrality | Position (y, x) | Degree |
| :---: | :---: | :---: | :---: | :---: |
| 1 | `130` | `0.2294` | `(271.0, 252.0)` | 4 |
| 2 | `17` | `0.2292` | `(275.0, 255.0)` | 4 |
| 3 | `207` | `0.1901` | `(284.0, 267.0)` | 3 |
| 4 | `210` | `0.1877` | `(285.0, 268.0)` | 3 |
| 5 | `232` | `0.1724` | `(294.0, 273.0)` | 3 |

### Road Closure Simulation Output (Target Node 130)

```json
{
  "closed_node_id": 130,
  "resilience_index": 0.9896,
  "travel_time_impact_pct": 1.05,
  "affected_km": 0.744,
  "disconnected": false,
  "disconnected_pair_count": 0
}
```

- **Resilience Index:** `0.9896` (High network detour absorption)
- **Travel Time Increase:** `+1.05%` average detour delay
- **Affected Segment Length:** `0.744 km` of road closed
- **Disconnection Status:** `False` (Network topology absorbs perturbation without severing OD routes)
