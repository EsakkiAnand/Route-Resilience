"""Graph providers used by the dashboard and future Stage A artifacts."""

from typing import Any, Dict, Tuple

import networkx as nx

try:
    import streamlit as st
except ImportError:  # pragma: no cover - allows core imports without Streamlit
    st = None


def validate_location(latitude: float, longitude: float, radius_m: int) -> None:
    """Validate a map query before making a network request."""
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")
    if not 100 <= radius_m <= 5000:
        raise ValueError("Radius must be between 100 and 5000 meters.")


def _download_osm_graph(latitude: float, longitude: float, radius_m: int) -> nx.Graph:
    """Download and normalize a small OSM driving graph for dashboard use."""
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError("OSMnx is not installed. Install the project requirements first.") from exc

    graph = ox.graph_from_point(
        (latitude, longitude),
        dist=radius_m,
        network_type="drive",
        simplify=True,
        retain_all=False,
    )
    normalized = nx.Graph()

    for node_id, data in graph.nodes(data=True):
        normalized.add_node(
            node_id,
            y=float(data["y"]),
            x=float(data["x"]),
            pos=(float(data["y"]), float(data["x"])),
        )

    for u, v, data in graph.edges(data=True):
        length = float(data.get("length", 1.0))
        existing = normalized.get_edge_data(u, v)
        if existing is None or length < existing["weight"]:
            normalized.add_edge(
                u,
                v,
                weight=length,
                length=length,
                highway=str(data.get("highway", "road")),
            )

    if normalized.number_of_nodes() == 0 or normalized.number_of_edges() == 0:
        raise RuntimeError("OpenStreetMap returned no drivable roads for this area.")
    return normalized


def get_osm_graph(latitude: float, longitude: float, radius_m: int) -> nx.Graph:
    """Return a cached OSM graph for a geographic query."""
    validate_location(latitude, longitude, radius_m)
    return _download_osm_graph(latitude, longitude, radius_m)


if st is not None:
    get_osm_graph = st.cache_resource(ttl=3600, show_spinner=False)(get_osm_graph)


def graph_summary(graph: nx.Graph) -> Dict[str, Any]:
    """Return display-safe graph statistics."""
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": nx.number_connected_components(graph) if graph else 0,
    }


def node_location(graph: nx.Graph, node_id: Any) -> Tuple[float, float]:
    """Read a node's geographic position in Folium's (latitude, longitude) order."""
    data = graph.nodes[node_id]
    if "y" not in data or "x" not in data:
        raise ValueError(f"Graph node {node_id} has no geographic coordinates.")
    return float(data["y"]), float(data["x"])