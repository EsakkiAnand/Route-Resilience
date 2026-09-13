# Route Resilience — Technical Document

**Problem Statement 04 — Bharatiya Antariksh Hackathon 2026**
**Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility**

---

## 1. Project Overview

Route Resilience is a two-stage pipeline that (A) extracts a connected, routable road network from satellite imagery even where roads are occluded by tree canopy, building shadows, or clouds, and (B) analyzes that network to identify structurally critical junctions and simulate the impact of their closure — producing a quantitative Resilience Index for disaster and urban-planning use.

- **Stage A — Map Healing (periodic, foundation layer):** raw satellite tile → occlusion-aware segmentation → skeletonization → topological (MST) healing → one connected weighted graph.
- **Stage B — Real-Time Resilience Analysis (the value-add):** healed graph → betweenness centrality → gatekeeper node ranking → node-ablation simulation → Resilience Index, travel-time impact, affected segment length.

Stage B is only as accurate as Stage A's output, so the two are built and tested as separable modules connected by one serialized graph artifact (GeoJSON / `networkx` pickle).

---

## 2. System Architecture

```
+---------------------------+
| DATA INGESTION LAYER      |  Sentinel-2 (Copernicus) . OSM (OSMnx / Geofabrik)
+-------------+--------------+
              v
+---------------------------+
| DEEP LEARNING LAYER       |  Attention U-Net / DeepLabV3+ segmentation
+-------------+--------------+
              v
+---------------------------+
| GRAPH ENGINE LAYER        |  Skeletonization -> MST/Disjoint-Set healing ->
|                           |  Betweenness centrality -> Node ablation
+-------------+--------------+
              v
+---------------------------+
| API LAYER  (NOT BUILT)    |  Serves graph + resilience results to frontend
+-------------+--------------+
              v
+---------------------------+
| APPLICATION LAYER         |  Streamlit + Leaflet/Folium dashboard,
|                           |  real Esri satellite basemap
+---------------------------+
```

---

## 3. Module-Wise Technical Description

### 3.1 Data Pipeline
- **Input:** bounding box + date range.
- **Sources:** Sentinel-2 L2A (10 m, via `sentinelhub`), OSM road vectors (via `OSMnx` / Geofabrik) as training ground truth.
- **Processing:** tiling (256x256, configurable overlap), synthetic occlusion augmentation (Albumentations) for training robustness.
- **Output:** paired (image, mask) tiles.

### 3.2 Occlusion-Aware Segmentation
- **Model:** U-Net (ResNet34 encoder) baseline; attention/transformer-enhanced variant as an upgrade path.
- **Loss:** Dice + BCE + boundary-aware, optional connectivity-penalty term.
- **Output:** binary road mask per tile.
- **Evaluation:** IoU, Dice, Occlusion-Recall (metric computed only on synthetically occluded regions), Relaxed IoU (3-5 px tolerance).

### 3.3 Topological Healing
- **Skeletonization:** `scikit-image.morphology.skeletonize` -> 1-pixel centerlines.
- **Graph conversion:** nodes = intersections/endpoints, edges = segments (weighted by real-world length).
- **Healing algorithm:** endpoint pairs within a distance threshold, filtered by angular trajectory alignment, connected via Union-Find + Kruskal's MST logic.
- **Output:** one connected, weighted graph (this is the artifact Stage B consumes).
- **Evaluation:** Connectivity Ratio, Topological Accuracy (shortest-path error vs. OSM ground truth).

### 3.4 Structural Intelligence & Stress Testing
- **Centrality:** `networkx.betweenness_centrality`, weighted by segment length.
- **Gatekeeper ranking:** top-N nodes by centrality.
- **Node ablation:** remove node -> recompute average shortest path length over sampled OD pairs.
- **Resilience Index:** baseline average shortest path length divided by perturbed average shortest path length. Full disconnection is explicitly detected and reported, not silently dropped.
- **Additional outputs:** Travel Time Impact (%), Affected Segment Length (km).

### 3.5 Dashboard (Application Layer)
- **Framework:** Streamlit + Folium.
- **Basemap:** real satellite imagery via Esri World Imagery tile layer (no API key required), togglable with a standard OSM street layer.
- **Overlay:** healed road graph, edges colored by 3-tier criticality (`#D85A30` critical / `#EF9F27` medium / `#5DCAA5` low).
- **Interaction:** click a node (sidebar list or map marker) -> simulates closure -> updates Resilience Index, Travel Time Impact, Affected Length live.
- **Status:** a working reference implementation (`app.py`) exists using a small 5-node sample graph; it is NOT yet wired to real Stage A/B output (see Section 4).

---

## 4. Current Implementation Status

| Component | Status | Notes |
|---|---|---|
| Data pipeline design | Specified, not coded | Sentinel-2 + OSM download/tiling/augmentation logic defined; no script written yet |
| Segmentation model | Not started | Architecture and loss function chosen; no training run yet |
| Skeletonization | Not started | Algorithm chosen (`scikit-image` + `sknw`); not implemented |
| Topological healing (MST) | Not started | Algorithm logic specified (distance + angle threshold, Union-Find, Kruskal's); not implemented or tested |
| Centrality / Resilience Index | Not started | Formula and disconnection handling specified; not implemented; no unit tests written |
| Dashboard UI/UX | Prototyped | Reference `app.py` runs end-to-end against a **sample hardcoded 5-node graph**, not real extracted data |
| Satellite map view | Partially done | Real Esri tile layer wired into the reference `app.py`; **not yet tested against a real bounding box's actual healed graph** -- currently renders sample coordinates only |
| Backend / API layer | **Not started** | No service layer exists; the Streamlit app currently reads the graph directly in-process (see Section 5) |

---

## 5. Remaining Work (Explicitly Scoped)

### 5.1 Backend

Currently there is no backend -- the Streamlit app loads and computes everything in-process from a Python object. This works for a single-user demo but not for a real deployment where multiple planners, multiple cities, or a mobile/other frontend need access to the same computed results. Remaining work:

- Stand up a backend service (recommended: **FastAPI**) that owns:
  - The pipeline execution (Stage A + Stage B) as background/async jobs, since segmentation + healing for a city-sized area will not complete within a single HTTP request.
  - Persistent storage for: raw tiles, trained model checkpoints, healed graphs (per bounding box + date), and computed centrality/resilience results -- a lightweight option is PostGIS (for graph/geo data) or a simple file store + SQLite/Postgres metadata table if PostGIS is out of scope for the hackathon timeline.
  - A job status mechanism (e.g. `pending` / `running` / `done` / `failed`) so the frontend can poll for long-running pipeline runs.
- Decide and document a deployment target (local Docker Compose is sufficient for a hackathon submission; cloud hosting is a stated post-hackathon concern, not needed for the demo).

### 5.2 API Logic

No API endpoints exist yet. At minimum, the following need to be defined and implemented:

| Endpoint (suggested) | Purpose |
|---|---|
| `POST /pipeline/run` | Trigger Stage A+B for a given bounding box + date range; returns a job ID |
| `GET /pipeline/status/{job_id}` | Poll job status |
| `GET /graph/{job_id}` | Return the healed graph (GeoJSON) for a completed job |
| `GET /graph/{job_id}/gatekeepers` | Return ranked top-N gatekeeper nodes with centrality scores |
| `POST /simulate` | Body: `{job_id, node_id}` -> returns `{resilience_index, travel_time_impact_pct, affected_km, disconnected}` |
| `GET /health` | Basic liveness check |

None of these currently exist. The dashboard's current "simulate on click" behavior calls Python functions directly in the same process -- this needs to be refactored to call the above endpoints once the backend exists, so the dashboard becomes a thin client rather than owning the computation.

### 5.3 Satellite Map View -- Remaining Work

The map rendering approach (Folium + Esri World Imagery tiles) is chosen and demonstrated in the reference `app.py`, but the following is NOT yet done:

- The map currently displays a **hardcoded 5-node sample graph** with manually chosen coordinates -- it has never been rendered against a real extracted/healed graph from actual satellite data.
- No verification yet that tile loading, zoom behavior, and marker/edge rendering perform acceptably once the graph has realistic node/edge counts (a real city sub-area could have hundreds of nodes, not 5) -- rendering performance and marker clustering have not been tested at that scale.
- The click-to-simulate interaction currently updates local Streamlit session state directly; once the API layer (5.2) exists, this needs to be refactored to call `POST /simulate` instead.
- No mobile/responsive testing has been done on the map view.

---

## 6. Tech Stack Summary

| Layer | Technologies |
|---|---|
| Geospatial/data | GDAL, Rasterio, OpenCV, NumPy, Albumentations, `sentinelhub`, OSMnx |
| Deep learning | PyTorch, U-Net/DeepLabV3+/UNet++, attention/transformer variants |
| Graph | NetworkX (core), PyTorch Geometric (optional GNN extension) |
| Backend / API | **To be built** -- FastAPI recommended |
| Storage | **To be decided** -- PostGIS or file store + SQLite/Postgres |
| Frontend / dashboard | Streamlit, Folium (Leaflet under the hood), Esri World Imagery tiles |

---

## 7. Evaluation Metrics (Defined, Pending Real Measurement)

| Metric | Definition | Status |
|---|---|---|
| IoU / Dice | Standard segmentation accuracy | Not yet measured -- no trained model |
| Occlusion-Recall | IoU/Dice computed only on synthetically occluded regions | Not yet measured |
| Relaxed IoU | 3-5 px tolerance buffer | Not yet measured |
| Connectivity Ratio | Largest-connected-component size before/after healing | Not yet measured |
| Topological Accuracy | Shortest-path error vs. OSM ground truth | Not yet measured |
| Resilience Index | Baseline vs. perturbed avg. shortest path length | Formula defined, unit tests not written |

---

## 8. Summary of Outstanding Items

1. **Backend service** -- does not exist; required for multi-user/production use, not strictly required for a single-machine hackathon demo.
2. **API layer** -- does not exist; endpoints listed in Section 5.2 need implementation.
3. **Satellite map view with real data** -- tile integration works in isolation but has never been run against a real Stage A/B output; needs end-to-end validation at realistic node/edge scale.
4. **All of Stage A and Stage B core logic** (segmentation training, skeletonization, MST healing, centrality, ablation) -- currently specified in detail (see the accompanying Implementation Plan Prompt) but not yet implemented or tested.
