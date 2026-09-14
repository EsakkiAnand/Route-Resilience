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
from route_resilience.graph.provider import get_osm_graph, graph_summary, node_location

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
        page_icon="R",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling for a focused operations dashboard.
    st.markdown(
        """
        <style>
        :root {
            --canvas: #f4f6f8;
            --panel: #ffffff;
            --ink: #17202a;
            --muted: #64717d;
            --line: #d9e0e6;
            --green: #176b3a;
        }
        .main {
            background-color: var(--canvas);
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background-color: #202a33;
        }
        [data-testid="stSidebar"] * {
            color: #edf2f5;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] input {
            background-color: #151c22;
            border-color: #46535e;
        }
        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }
        .dashboard-kicker {
            color: var(--green);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .metric-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 4px;
            padding: 14px 16px;
            text-align: left;
            box-shadow: 0 1px 2px rgba(23, 32, 42, 0.06);
        }
        .metric-title {
            font-size: 0.72rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .metric-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--ink);
            margin-top: 4px;
        }
        .alert-disconnected {
            background-color: #fff1f1;
            border-left: 4px solid #c62828;
            color: #8e2020;
            padding: 12px 14px;
            border-radius: 2px;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .alert-stable {
            background-color: #edf7f0;
            border-left: 4px solid var(--green);
            color: #185b34;
            padding: 12px 14px;
            border-radius: 2px;
            font-weight: 600;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='dashboard-kicker'>Network operations view</div>", unsafe_allow_html=True)
    st.title("Route Resilience")
    st.caption("Reference road network monitoring and junction failure analysis")

    # Load a small, geographic OSM reference graph for the live demonstration.
    st.sidebar.header("Map area")
    latitude = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=12.9716, format="%.4f")
    longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.5946, format="%.4f")
    radius_m = st.sidebar.slider("Road search radius (meters)", min_value=100, max_value=5000, value=1000, step=100)
    st.sidebar.caption("Default demo area: Bengaluru. Road data is fetched from OpenStreetMap.")

    try:
        with st.spinner("Loading the OpenStreetMap reference network..."):
            G = get_osm_graph(latitude, longitude, radius_m)
    except Exception as exc:
        st.error(f"Unable to load road data for this area. Please reduce the radius and try again. ({exc})")
        st.stop()

    centrality_map = compute_betweenness_centrality(G, weight_attribute="weight")
    gatekeepers = get_gatekeeper_nodes(G, top_n=10, centrality_map=centrality_map)
    summary = graph_summary(G)
    st.caption(
        f"Real OSM Reference Network · {summary['nodes']} nodes · {summary['edges']} roads · "
        f"{summary['components']} connected component(s) · AI-Healed Network: Stage A future output"
    )
    with st.expander("How to read this dashboard", expanded=False):
        st.markdown(
            "**Dark green routes** are the current OpenStreetMap reference network. "
            "**Orange nodes** are the highest-ranked junctions by weighted betweenness centrality. "
            "Choose a junction to simulate its closure; red routes show the directly affected roads. "
            "The satellite layer is visual context only and is not yet used for AI extraction."
        )

    # Session State for Selected Closed Node
    if "closed_node" not in st.session_state:
        st.session_state["closed_node"] = None

    # Sidebar Controls
    st.sidebar.header("Closure analysis")
    st.sidebar.write("Select a high-centrality junction to simulate a closure.")

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
    st.sidebar.subheader("Specific node target")
    all_node_ids = sorted(list(G.nodes()))
    custom_node = st.sidebar.selectbox("Choose Node ID directly:", options=[None] + all_node_ids)

    if custom_node is not None:
        st.session_state["closed_node"] = custom_node

    # Reset Button
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset to baseline"):
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

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Reference nodes", f"{summary['nodes']:,}")
    summary_col2.metric("Reference roads", f"{summary['edges']:,}")
    summary_col3.metric("Critical nodes shown", len(gatekeepers))

    st.markdown("<br>", unsafe_allow_html=True)

    # Disconnection Alert Banner
    if impact["disconnected"]:
        st.markdown(
            f"""
            <div class="alert-disconnected">
                SEVERE INFRASTRUCTURE DISCONNECTION: Closing Node {closed_node} severs {impact['disconnected_pair_count']}/{impact['total_pairs_evaluated']} origin-destination routes into isolated network components.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif closed_node is not None:
        st.markdown(
            f"""
            <div class="alert-stable">
                NETWORK STABLE: Alternative detour routes absorb the closure of Node {closed_node} without severing origin-destination paths.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Map Rendering
    st.subheader("Network topology and critical junctions")
    st.markdown(
        "<span style='color:#176b3a; font-size:1.25rem;'>━</span> OSM reference road&nbsp;&nbsp;"
        "<span style='color:#ff1744; font-size:1.25rem;'>━</span> closure-affected road&nbsp;&nbsp;"
        "<span style='color:#ff9100;'>●</span> critical junction",
        unsafe_allow_html=True,
    )

    if HAS_FOLIUM:
        m = folium.Map(location=[latitude, longitude], zoom_start=15, tiles=None, control_scale=True)
        folium.TileLayer(
            tiles="OpenStreetMap",
            name="🗺️ OpenStreetMap",
            control=True,
            show=True,
        ).add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
            name="🛰️ Esri Satellite",
            overlay=False,
            control=True,
            show=False,
        ).add_to(m)

        # Render Edges
        for u, v, data in G.edges(data=True):
            pos_u = node_location(G, u)
            pos_v = node_location(G, v)

            if closed_node is not None and (u == closed_node or v == closed_node):
                edge_color = "#ff1744"  # Closed edge (red)
                weight = 5
                opacity = 0.9
            else:
                edge_color = "#176b3a"  # OSM reference road (dark green)
                weight = 2
                opacity = 0.6

            folium.PolyLine(
                locations=[pos_u, pos_v],
                color=edge_color,
                weight=weight,
                opacity=opacity,
                tooltip=f"OSM road {u}-{v} | Length: {data.get('weight', 0):.1f}m",
            ).add_to(m)

        # Render only the highest-ranked nodes to keep the map responsive.
        critical_nodes = {gk["node_id"] for gk in gatekeepers}
        if closed_node is not None:
            critical_nodes.add(closed_node)
        for n in critical_nodes:
            data = G.nodes[n]
            pos = node_location(G, n)
            cent = centrality_map.get(n, 0.0)

            if n == closed_node:
                color = "#ff1744"  # Target closed node
                radius = 10
            elif n in critical_nodes:
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

        folium.LayerControl(collapsed=False).add_to(m)
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
