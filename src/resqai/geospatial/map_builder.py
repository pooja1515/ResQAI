from __future__ import annotations

import argparse
import logging
from pathlib import Path

from resqai.geospatial.event_mapper import build_events
from resqai.geospatial.geo_utils import LatLon, ensure_dir, load_json, try_extract_latlon
from resqai.geospatial.timeline_mapper import build_timeline_markers

logger = logging.getLogger("resqai.geospatial")


def _import_folium():
    try:
        import folium  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("folium is required. Install with: pip install -r requirements.txt") from exc
    return folium


def build_map(input_json: dict, *, output_path: Path) -> Path:
    folium = _import_folium()
    try:
        from folium.plugins import HeatMap  # type: ignore
    except Exception:
        HeatMap = None  # type: ignore

    # Determine anchor coordinates from weather agent debug info if present.
    anchor = None
    debug = input_json.get("_debug") if isinstance(input_json.get("_debug"), dict) else None
    agents = debug.get("agents") if isinstance(debug, dict) else None
    weather_data = None
    if isinstance(agents, dict):
        w = agents.get("weather") or {}
        if isinstance(w, dict) and isinstance(w.get("data"), dict):
            weather_data = w["data"]
    if isinstance(weather_data, dict):
        anchor = try_extract_latlon(weather_data)

    if anchor is None:
        # Fallback: generic world center.
        anchor = LatLon(lat=20.5937, lon=78.9629)  # India centroid-ish
        logger.warning("No coordinates found in input; using fallback anchor (India). Run orchestrator with --debug + weather agent for best results.")

    # Dark mode default: start with no tiles, then add dark as the first visible layer.
    m = folium.Map(location=[anchor.lat, anchor.lon], zoom_start=12, control_scale=True, tiles=None)

    # Basemap styling + command-center overlays
    folium.TileLayer("CartoDB dark_matter", name="Dark (default)", control=True, show=True).add_to(m)
    folium.TileLayer("CartoDB positron", name="Light", control=True, show=False).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Streets", control=True, show=False).add_to(m)

    # Legend (simple HTML)
    legend = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 9999;
        background: rgba(255,255,255,0.92); padding: 10px 12px; border-radius: 10px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.2); font-size: 12px;">
      <div style="font-weight:700; margin-bottom:6px;">ResQAI Risk Legend</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#2ecc71;border-radius:50%;margin-right:6px;"></span>Low</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#f39c12;border-radius:50%;margin-right:6px;"></span>Moderate</div>
      <div><span style="display:inline-block;width:10px;height:10px;background:#e74c3c;border-radius:50%;margin-right:6px;"></span>High/Critical</div>
      <div style="margin-top:6px; color:#444;">Circles indicate hotspots/escalation.</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    # Command-center styling + pulse animation
    css = """
    <style>
      .leaflet-popup-content { margin: 10px 12px; }
      .resqai-popup { width: 360px; color: #0b0f14; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }
      .resqai-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
      .resqai-title { font-weight:800; font-size:14px; letter-spacing:0.2px; }
      .resqai-badge { font-weight:800; font-size:10px; padding:4px 8px; border-radius:999px; color:white; }
      .resqai-badge-red { background:#e74c3c; }
      .resqai-badge-orange { background:#f39c12; }
      .resqai-badge-green { background:#2ecc71; color:#0b0f14; }
      .resqai-badge-gray { background:#95a5a6; }
      .resqai-grid { display:grid; grid-template-columns: 1fr 1fr; gap:8px; }
      .resqai-card { background: rgba(255,255,255,0.92); border: 1px solid rgba(0,0,0,0.06); border-radius:10px; padding:8px; }
      .resqai-span-2 { grid-column: 1 / span 2; }
      .resqai-section-title { font-weight:800; font-size:11px; margin-bottom:6px; color:#111; }
      .resqai-muted { color:#333; font-size:11px; margin-top:4px; }
      .resqai-summary { margin-top:8px; font-size:12px; background: rgba(255,255,255,0.92); border-radius:10px; padding:8px; border: 1px solid rgba(0,0,0,0.06); }
      .resqai-popup ul { margin: 0; padding-left: 18px; }
      .resqai-popup li { margin: 2px 0; font-size: 11.5px; }

      .resqai-pulse {
        width: 20px; height: 20px;
        border-radius: 50%;
        background: rgba(231,76,60,0.85);
        position: relative;
        box-shadow: 0 0 18px rgba(231,76,60,0.55);
      }
      .resqai-pulse:before {
        content:"";
        position:absolute; left:50%; top:50%;
        width:20px; height:20px;
        transform: translate(-50%, -50%);
        border-radius:50%;
        background: rgba(231,76,60,0.35);
        animation: resqai-pulse 1.6s ease-out infinite;
      }
      @keyframes resqai-pulse {
        0% { width: 20px; height: 20px; opacity: 0.9; }
        100% { width: 70px; height: 70px; opacity: 0.0; }
      }
    </style>
    """
    m.get_root().html.add_child(folium.Element(css))

    # Add core markers and overlays as separate layers
    fg_markers = folium.FeatureGroup(name="Intelligence markers", show=True)
    fg_hotspots = folium.FeatureGroup(name="Hotspots & escalation", show=True)
    fg_heat = folium.FeatureGroup(name="Intensity heatmap", show=False)

    events = build_events(input_json)
    heat_points: list[list[float]] = []
    for e in events:
        if e.kind == "circle":
            folium.CircleMarker(
                location=[e.latlon.lat, e.latlon.lon],
                radius=int(e.radius),
                color=e.color,
                fill=True,
                fill_color=e.color,
                fill_opacity=0.28,
                weight=int(e.weight),
                popup=folium.Popup(e.popup_html, max_width=520),
                tooltip=e.title,
            ).add_to(fg_hotspots)
            heat_points.append([e.latlon.lat, e.latlon.lon, float(e.intensity)])
        elif e.kind == "pulse":
            # Pulsing DivIcon for command-center focus
            folium.Marker(
                location=[e.latlon.lat, e.latlon.lon],
                popup=folium.Popup(e.popup_html, max_width=520),
                tooltip=e.title,
                icon=folium.DivIcon(html="<div class='resqai-pulse'></div>"),
            ).add_to(fg_hotspots)
            heat_points.append([e.latlon.lat, e.latlon.lon, float(e.intensity)])
        else:
            folium.Marker(
                location=[e.latlon.lat, e.latlon.lon],
                popup=folium.Popup(e.popup_html, max_width=520),
                tooltip=e.title,
                icon=folium.Icon(color=e.color, icon=e.icon),
            ).add_to(fg_markers)
            heat_points.append([e.latlon.lat, e.latlon.lon, float(e.intensity)])

    # Add temporal memory marker (if present)
    memory_insight = None
    if isinstance(agents, dict):
        mem = agents.get("memory") or {}
        if isinstance(mem, dict) and isinstance(mem.get("data"), dict):
            memory_insight = mem["data"]
    for tm in build_timeline_markers(anchor=anchor, memory_insight=memory_insight):
        folium.CircleMarker(
            location=[tm.latlon.lat, tm.latlon.lon],
            radius=int(tm.radius),
            color=tm.color,
            fill=True,
            fill_opacity=0.55,
            weight=int(tm.weight),
            popup=folium.Popup(tm.popup_html, max_width=420),
            tooltip=tm.label,
        ).add_to(fg_hotspots)

        # Timestamp overlay label (simple DivIcon)
        if tm.timestamp:
            folium.Marker(
                location=[tm.latlon.lat, tm.latlon.lon],
                icon=folium.DivIcon(
                    html=f"<div style='font-size:11px;color:#111;background:rgba(255,255,255,0.85);padding:2px 6px;border-radius:8px;border:1px solid #ddd;display:inline-block;'>"
                    f"{tm.timestamp}</div>"
                ),
            ).add_to(fg_hotspots)

    fg_markers.add_to(m)
    fg_hotspots.add_to(m)
    if HeatMap is not None and heat_points:
        HeatMap(heat_points, name="Heatmap", min_opacity=0.25, radius=35, blur=26, max_zoom=14).add_to(fg_heat)
        fg_heat.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    ensure_dir(output_path.parent)
    m.save(str(output_path))
    return output_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="Build an interactive Leaflet map from ResQAI orchestrator outputs.")
    p.add_argument("--input-json", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/maps/resqai_map.html"))
    args = p.parse_args()

    data = load_json(args.input_json)
    out = build_map(data, output_path=args.output)
    logger.info("map_saved=%s", str(out))


if __name__ == "__main__":
    main()
