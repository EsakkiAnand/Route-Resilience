"""
Route Resilience Interactive Streamlit Dashboard Application.
Stage A Map Healing Visualization & Stage B Real-Time Road Closure Stress Testing.
"""

import sys
import json
from pathlib import Path
import yaml
import numpy as np
import networkx as nx

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_resilience.graph.centrality import compute_betweenness_centrality, get_gatekeeper_nodes
from route_resilience.graph.ablation import simulate_node_closure

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


def load_stage_a_graph():
    """Loads Stage A healed graph and precomputes centrality scores."""
    json_path = PROJECT_ROOT / "graph" / "outputs" / "healed_stage_a_graph.json"
    graphML_path = PROJECT_ROOT / "graph" / "outputs" / "healed_stage_a_graph.graphml"

    if graphML_path.exists():
        G = nx.read_graphml(graphML_path)
        mapping = {n: int(n) if str(n).isdigit() else n for n in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
    else:
        # Generate synthetic demo graph if graph artifact is missing
        G = nx.grid_2d_graph(6, 6)
        G = nx.convert_node_labels_to_integers(G)
        for u, v in G.edges():
            G.edges[u, v]["weight"] = float(np.random.uniform(50, 200))
            G.edges[u, v]["healed"] = False
        for n in G.nodes():
            G.nodes[n]["y"] = float(12.97 + (n // 6) * 0.003)
            G.nodes[n]["x"] = float(77.59 + (n % 6) * 0.003)

    centrality = compute_betweenness_centrality(G, weight_attribute="weight")
    gatekeepers = get_gatekeeper_nodes(G, top_n=10, centrality_map=centrality)
    return G, centrality, gatekeepers


def main():
    if not HAS_STREAMLIT:
        print("==================================================")
        print("  ROUTE RESILIENCE DASHBOARD (CLI MODE)           ")
        print("==================================================")
        print("[!] Streamlit is not installed in the active environment.")
        print("    To launch the interactive web dashboard, run:")
        print("    pip install streamlit folium streamlit-folium")
        print("    streamlit run dashboard/app.py")
        print("==================================================")
        G, centrality_map, gatekeepers = load_stage_a_graph()
        print(f"[OK] Stage A Graph Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print("[OK] Top Gatekeeper Junctions:")
        for gk in gatekeepers[:3]:
            print(f"    - Node {gk['node_id']} | Centrality: {gk['centrality']:.4f}")
        return

    # Page Configuration
    st.set_page_config(
        page_title="Route Resilience — Real-Time Road Stress Testing",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Styling (Dark Glassmorphism UI)
    st.markdown(
        """
        <style>
        .main {
            background-color: #0e1117;
            color: #e0e0e0;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        .metric-title {
            font-size: 0.85rem;
            color: #9e9e9e;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #00e676;
            margin-top: 4px;
        }
        .alert-disconnected {
            background-color: rgba(255, 23, 68, 0.15);
            border: 1px solid #ff1744;
            color: #ff5252;
            padding: 12px;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .alert-stable {
            background-color: rgba(0, 230, 118, 0.15);
            border: 1px solid #00e676;
            color: #69f0ae;
            padding: 12px;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🛰️ Route Resilience — Road Network Intelligence")
    st.caption("Satellite Map Healing (Stage A) & Real-Time Stress Testing Simulation (Stage B)")

    # Load Graph Data
    G, centrality_map, gatekeepers = load_stage_a_graph()

    # Session State for Selected Closed Node
    if "closed_node" not in st.session_state:
        st.session_state["closed_node"] = None

    # Sidebar Controls
    st.sidebar.header("🎯 Gatekeeper Node Intelligence")
    st.sidebar.write("Top Critical Junctions ranked by Weighted Betweenness Centrality:")

    gatekeeper_options = {
        f"Node {gk['node_id']} (Centrality: {gk['centrality']:.3f})": gk["node_id"]
        for gk in gatekeepers
    }

    selected_gk_label = st.sidebar.selectbox(
        "Select Gatekeeper Junction to Test Closure:",
        options=["-- Baseline (No Closures) --"] + list(gatekeeper_options.keys()),
    )

    if selected_gk_label != "-- Baseline (No Closures) --":
        st.session_state["closed_node"] = gatekeeper_options[selected_gk_label]
    else:
        st.session_state["closed_node"] = None

    # Manual Node ID Search
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Specific Node Target")
    all_node_ids = sorted(list(G.nodes()))
    custom_node = st.sidebar.selectbox("Choose Node ID directly:", options=[None] + all_node_ids)

    if custom_node is not None:
        st.session_state["closed_node"] = custom_node

    # Reset Button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset to Baseline"):
        st.session_state["closed_node"] = None
        st.rerun()

    # Run Simulation Metrics if a node is closed
    closed_node = st.session_state["closed_node"]
    if closed_node is not None:
        impact = simulate_node_closure(G, closed_node_id=closed_node, sample_pairs_count=100)
    else:
        impact = {
            "closed_node_id": None,
            "resilience_index": 1.0,
            "travel_time_impact_pct": 0.0,
            "affected_segment_length_km": 0.0,
            "disconnected": False,
            "disconnected_pair_count": 0,
            "total_pairs_evaluated": 100,
            "baseline_avg_length_m": 0.0,
            "perturbed_avg_length_m": 0.0,
        }

    # Top KPI Metrics Header
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Resilience Index</div>
                <div class="metric-value" style="color: {'#ff1744' if impact['resilience_index'] < 0.8 else '#00e676'};">
                    {impact['resilience_index']:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Travel Time Impact</div>
                <div class="metric-value" style="color: {'#ff9100' if impact['travel_time_impact_pct'] > 5.0 else '#00e676'};">
                    +{impact['travel_time_impact_pct']:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Affected Segment Length</div>
                <div class="metric-value" style="color: #29b6f6;">
                    {impact['affected_segment_length_km']:.2f} km
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Active Closure Target</div>
                <div class="metric-value" style="color: #e0e0e0;">
                    {f"Node {closed_node}" if closed_node is not None else "None (Baseline)"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Disconnection Alert Banner
    if impact["disconnected"]:
        st.markdown(
            f"""
            <div class="alert-disconnected">
                ⚠️ SEVERE INFRASTRUCTURE DISCONNECTION: Closing Node {closed_node} severs {impact['disconnected_pair_count']}/{impact['total_pairs_evaluated']} origin-destination routes into isolated network components!
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif closed_node is not None:
        st.markdown(
            f"""
            <div class="alert-stable">
                ✅ NETWORK STABLE: Alternative detour routes absorb the closure of Node {closed_node} without severing origin-destination paths.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Map Rendering
    st.subheader("🗺️ Network Topology & Centrality Heatmap")

    # Center map on graph centroid
    nodes_y = [G.nodes[n].get("y", G.nodes[n].get("pos", (12.97, 77.59))[0]) for n in G.nodes()]
    nodes_x = [G.nodes[n].get("x", G.nodes[n].get("pos", (12.97, 77.59))[1]) for n in G.nodes()]
    center_y = float(np.mean(nodes_y)) if nodes_y else 12.97
    center_x = float(np.mean(nodes_x)) if nodes_x else 77.59

    if HAS_FOLIUM:
        m = folium.Map(location=[center_y, center_x], zoom_start=14, tiles="CartoDB dark_matter")

        # Render Edges
        for u, v, data in G.edges(data=True):
            pos_u = (G.nodes[u].get("y", G.nodes[u].get("pos", (0, 0))[0]), G.nodes[u].get("x", G.nodes[u].get("pos", (0, 0))[1]))
            pos_v = (G.nodes[v].get("y", G.nodes[v].get("pos", (0, 0))[0]), G.nodes[v].get("x", G.nodes[v].get("pos", (0, 0))[1]))

            is_healed = data.get("healed", False)
            if closed_node is not None and (u == closed_node or v == closed_node):
                edge_color = "#ff1744"  # Closed edge (red)
                weight = 5
                opacity = 0.9
            elif is_healed:
                edge_color = "#00e5ff"  # Stage A Healed edge (cyan)
                weight = 3
                opacity = 0.8
            else:
                edge_color = "#76ff03"  # Regular road edge (green)
                weight = 2
                opacity = 0.6

            folium.PolyLine(
                locations=[pos_u, pos_v],
                color=edge_color,
                weight=weight,
                opacity=opacity,
                tooltip=f"Edge {u}-{v} | Weight: {data.get('weight', 0):.1f}m | Healed: {is_healed}",
            ).add_to(m)

        # Render Nodes
        max_cent = max(centrality_map.values()) if centrality_map and max(centrality_map.values()) > 0 else 1.0
        for n, data in G.nodes(data=True):
            pos = (data.get("y", data.get("pos", (0, 0))[0]), data.get("x", data.get("pos", (0, 0))[1]))
            cent = centrality_map.get(n, 0.0)

            if n == closed_node:
                color = "#ff1744"  # Target closed node
                radius = 10
            elif cent >= 0.1:
                color = "#ff9100"  # High centrality gatekeeper
                radius = 7
            else:
                color = "#29b6f6"  # Regular node
                radius = 4

            folium.CircleMarker(
                location=pos,
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=f"Node ID: {n}<br>Centrality: {cent:.4f}<br>Degree: {G.degree(n)}",
            ).add_to(m)

        st_folium(m, width="100%", height=550)
    else:
        st.info("Folium library not detected. Rendering node table view:")
        node_table = [
            {
                "Node ID": n,
                "Centrality": round(centrality_map.get(n, 0.0), 4),
                "Degree": G.degree(n),
                "Position": (round(G.nodes[n].get("y", 0), 4), round(G.nodes[n].get("x", 0), 4)),
            }
            for n in G.nodes()
        ]
        st.dataframe(node_table)


if __name__ == "__main__":
    main()
