"""
Interactive Map Visualisation for Species Extinction Risk Predictions
=====================================================================
Build Folium-based HTML maps with colour-coded IUCN risk markers, heatmap
layers, clustering, and optional attention overlays.

Dependencies:
    pip install folium branca
    pip install folium[plugins]   # for HeatMap, MarkerCluster
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import folium
    from folium import plugins as folium_plugins
    from branca.element import Template, MacroElement
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── IUCN colour palette ──────────────────────────────────────────────────────
IUCN_COLOURS: Dict[str, str] = {
    "LC": "green",
    "NT": "orange",
    "VU": "orange",
    "EN": "red",
    "CR": "darkred",
}

IUCN_HEX: Dict[str, str] = {
    "LC": "#4CAF50",
    "NT": "#CDDC39",
    "VU": "#FF9800",
    "EN": "#F44336",
    "CR": "#8B0000",
}

IUCN_LABELS: Dict[str, str] = {
    "LC": "Least Concern",
    "NT": "Near Threatened",
    "VU": "Vulnerable",
    "EN": "Endangered",
    "CR": "Critically Endangered",
}


# ═════════════════════════════════════════════════════════════════════════════
# Legend macro (injected into the Folium map HTML)
# ═════════════════════════════════════════════════════════════════════════════

_LEGEND_HTML = """
{% macro html(this, kwargs) %}
<div style="
    position: fixed;
    bottom: 30px; left: 30px;
    z-index: 1000;
    background: white;
    padding: 12px 16px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    line-height: 1.6;
">
    <b style="font-size:14px;">IUCN Risk Categories</b><br>
    <span style="color:#4CAF50;">&#9679;</span> LC – Least Concern<br>
    <span style="color:#CDDC39;">&#9679;</span> NT – Near Threatened<br>
    <span style="color:#FF9800;">&#9679;</span> VU – Vulnerable<br>
    <span style="color:#F44336;">&#9679;</span> EN – Endangered<br>
    <span style="color:#8B0000;">&#9679;</span> CR – Critically Endangered
</div>
{% endmacro %}
"""


def _add_legend(m: "folium.Map") -> None:
    """Inject an IUCN legend into the map."""
    macro = MacroElement()
    macro._template = Template(_LEGEND_HTML)
    m.get_root().add_child(macro)


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def create_prediction_map(
    predictions_df: pd.DataFrame,
    output_path: str = "map.html",
    *,
    center: Tuple[float, float] = (20.0, 0.0),
    zoom_start: int = 3,
    use_clustering: bool = True,
    show_heatmap: bool = True,
    show_deforestation: bool = False,
) -> Optional[str]:
    """Create an interactive world map with species extinction-risk markers.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain columns: ``species_name``, ``lat``, ``lon``,
        ``predicted_class``, ``confidence``.  Optional columns:
        ``habitat_type``, ``LC_prob``, ``NT_prob``, ``VU_prob``,
        ``EN_prob``, ``CR_prob``.
    output_path : str
        File path for the standalone HTML output.
    center : tuple
        Initial map centre (lat, lon).
    zoom_start : int
        Initial zoom level.
    use_clustering : bool
        Enable marker clustering for dense areas.
    show_heatmap : bool
        Add a heatmap layer for high-risk species concentration.
    show_deforestation : bool
        Overlay known deforestation regions (placeholder WMS layer).

    Returns
    -------
    str or None
        Absolute path to the saved HTML file, or None on failure.
    """
    if not _FOLIUM_AVAILABLE:
        logger.error("Folium is not installed.  Run: pip install folium branca")
        return None

    df = predictions_df.copy()

    # ── base map ─────────────────────────────────────────────────────────
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Add an alternative tile layer
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)

    # ── feature groups per IUCN class (for layer control) ────────────────
    feature_groups: Dict[str, folium.FeatureGroup] = {}
    for cls, label in IUCN_LABELS.items():
        fg = folium.FeatureGroup(name=f"{cls} – {label}", show=True)
        feature_groups[cls] = fg

    # ── optional: marker clustering ──────────────────────────────────────
    cluster = None
    if use_clustering:
        cluster = folium_plugins.MarkerCluster(
            name="Clustered Markers",
            options={"maxClusterRadius": 50, "disableClusteringAtZoom": 8},
        )

    # ── add markers ──────────────────────────────────────────────────────
    high_risk_coords: List[List[float]] = []

    for _, row in df.iterrows():
        lat, lon = row["lat"], row["lon"]
        cls = row.get("predicted_class", "LC")
        confidence = row.get("confidence", 0.0)
        species = row.get("species_name", "Unknown")
        habitat = row.get("habitat_type", "N/A")

        colour = IUCN_COLOURS.get(cls, "gray")

        # Build popup HTML
        prob_rows = ""
        for c in ["LC", "NT", "VU", "EN", "CR"]:
            prob_col = f"{c}_prob"
            if prob_col in row:
                pct = row[prob_col] * 100
                bar_colour = IUCN_HEX.get(c, "#999")
                prob_rows += (
                    f"<tr>"
                    f"<td style='padding:2px 6px;'>{c}</td>"
                    f"<td style='padding:2px 6px;'>"
                    f"<div style='background:{bar_colour};width:{pct:.0f}%;height:12px;"
                    f"border-radius:3px;'></div></td>"
                    f"<td style='padding:2px 6px;'>{pct:.1f}%</td>"
                    f"</tr>"
                )

        popup_html = f"""
        <div style="font-family:'Segoe UI',Arial,sans-serif;min-width:220px;">
            <h4 style="margin:0 0 6px 0;color:#333;"><i>{species}</i></h4>
            <table style="font-size:12px;width:100%;">
                <tr><td><b>Risk Level</b></td><td style="color:{IUCN_HEX.get(cls, '#333')};
                    font-weight:bold;">{cls} – {IUCN_LABELS.get(cls, cls)}</td></tr>
                <tr><td><b>Confidence</b></td><td>{confidence:.1%}</td></tr>
                <tr><td><b>Habitat</b></td><td>{habitat}</td></tr>
                <tr><td><b>Location</b></td><td>{lat:.4f}, {lon:.4f}</td></tr>
            </table>
            {"<hr style='margin:6px 0;'><table style='font-size:11px;width:100%;'>"
             + prob_rows + "</table>" if prob_rows else ""}
        </div>
        """

        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{species} ({cls})",
        )

        # Route to feature group
        fg = feature_groups.get(cls)
        if fg is not None:
            if cluster is not None:
                marker.add_to(cluster)
            else:
                marker.add_to(fg)

        # Collect high-risk coords for heatmap
        if cls in ("EN", "CR"):
            weight = 2.0 if cls == "CR" else 1.0
            high_risk_coords.append([lat, lon, weight])

    # Add feature groups to map
    for fg in feature_groups.values():
        fg.add_to(m)
    if cluster is not None:
        cluster.add_to(m)

    # ── heatmap layer ────────────────────────────────────────────────────
    if show_heatmap and high_risk_coords:
        heat_layer = folium_plugins.HeatMap(
            high_risk_coords,
            name="High-Risk Heatmap",
            min_opacity=0.3,
            radius=25,
            blur=15,
            gradient={0.4: "yellow", 0.65: "orange", 1.0: "red"},
            show=False,  # hidden by default
        )
        heat_layer.add_to(m)

    # ── deforestation overlay (placeholder WMS) ──────────────────────────
    if show_deforestation:
        folium.WmsTileLayer(
            url="https://firms.modaps.eosdis.nasa.gov/wms/",
            layers="fires_viirs_24",
            name="Active Fire / Deforestation",
            fmt="image/png",
            transparent=True,
            show=False,
        ).add_to(m)

    # ── legend + layer control ───────────────────────────────────────────
    _add_legend(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # ── save ─────────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    logger.info("Interactive map saved to %s (%d species plotted)", out, len(df))
    return str(out.resolve())


def create_attention_overlay(
    lat: float,
    lon: float,
    attention_map: np.ndarray,
    base_map: Optional["folium.Map"] = None,
    *,
    size_deg: float = 0.5,
    opacity: float = 0.6,
) -> "folium.Map":
    """Overlay a spatial attention heatmap on the map centred at (lat, lon).

    Parameters
    ----------
    lat, lon : float
        Centre coordinates.
    attention_map : np.ndarray
        2-D array of attention weights (H, W), values in [0, 1].
    base_map : folium.Map or None
        Existing map to add to; if None, a new map is created.
    size_deg : float
        Spatial extent of the overlay in degrees.
    opacity : float
        Overlay opacity.

    Returns
    -------
    folium.Map
        Map with the attention overlay added.
    """
    if not _FOLIUM_AVAILABLE:
        raise ImportError("Folium is required. Install via: pip install folium")

    if base_map is None:
        base_map = folium.Map(location=[lat, lon], zoom_start=10)

    # Normalise attention to [0, 255] for image overlay
    attn = attention_map.astype(np.float64)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    # Create RGBA image (red channel = attention intensity)
    h, w = attn.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = (attn * 255).astype(np.uint8)  # R
    rgba[:, :, 3] = (attn * 255 * opacity).astype(np.uint8)  # A

    # Compute bounds
    half = size_deg / 2
    bounds = [[lat - half, lon - half], [lat + half, lon + half]]

    # We need to convert the numpy array to a PNG image
    try:
        from io import BytesIO
        from PIL import Image
        import base64

        img = Image.fromarray(rgba, mode="RGBA")
        buf = BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_url = f"data:image/png;base64,{img_b64}"

        folium.raster_layers.ImageOverlay(
            image=img_url,
            bounds=bounds,
            opacity=opacity,
            name="Attention Overlay",
            interactive=True,
        ).add_to(base_map)

    except ImportError:
        logger.warning("Pillow not installed — using HeatMap fallback for attention.")
        # Fallback: convert attention grid to point-based heatmap
        coords = []
        for i in range(h):
            for j in range(w):
                if attn[i, j] > 0.1:
                    pt_lat = lat - half + (i / h) * size_deg
                    pt_lon = lon - half + (j / w) * size_deg
                    coords.append([pt_lat, pt_lon, float(attn[i, j])])
        if coords:
            folium_plugins.HeatMap(
                coords, name="Attention Heatmap", radius=10,
            ).add_to(base_map)

    folium.LayerControl().add_to(base_map)
    return base_map
