from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from resqai.geospatial.geo_utils import LatLon, summarize_text, try_extract_latlon
from resqai.geospatial.risk_visualizer import (
    badge_text,
    class_to_style,
    escalation_to_style,
    risk_to_style,
 )

logger = logging.getLogger("resqai.geospatial")


@dataclass(frozen=True)
class MapEvent:
    latlon: LatLon
    title: str
    popup_html: str
    color: str
    icon: str
    kind: str = "marker"  # marker|circle
    intensity: float = 0.5
    radius: int = 18
    weight: int = 3


def build_events(orchestrator_output: dict[str, Any]) -> list[MapEvent]:
    """Convert orchestrator output (optionally with `_debug`) into map events."""
    out: list[MapEvent] = []

    # Final fused output (operational summary)
    fused = orchestrator_output

    # Primary anchor: weather agent location if present (best chance of coords).
    debug = orchestrator_output.get("_debug") if isinstance(orchestrator_output.get("_debug"), dict) else None
    agents = debug.get("agents") if isinstance(debug, dict) else None

    weather_data = None
    if isinstance(agents, dict):
        w = agents.get("weather") or {}
        if isinstance(w, dict) and isinstance(w.get("data"), dict):
            weather_data = w["data"]

    anchor_ll = try_extract_latlon(weather_data) if isinstance(weather_data, dict) else None

    def add_event(
        ll: LatLon,
        title: str,
        popup: str,
        color: str,
        icon: str,
        *,
        kind: str = "marker",
        intensity: float = 0.5,
        radius: int = 18,
        weight: int = 3,
    ) -> None:
        out.append(
            MapEvent(
                latlon=ll,
                title=title,
                popup_html=popup,
                color=color,
                icon=icon,
                kind=kind,
                intensity=intensity,
                radius=radius,
                weight=weight,
            )
        )

    if anchor_ll:
        # Multimodal operational popup panel (command-center)
        style = risk_to_style(str(fused.get("overall_risk") or ""))
        badge = badge_text(fused.get("overall_risk"), fused.get("weather_escalation"))
        badge_html = (
            f"<span class='resqai-badge resqai-badge-{style.color}'>{badge}</span>" if badge else ""
        )

        def _ul(items: list[str]) -> str:
            items = [str(x) for x in (items or []) if str(x).strip()][:6]
            if not items:
                return "<div class='resqai-muted'>(none)</div>"
            return "<ul>" + "".join([f"<li>{summarize_text(i, max_chars=110)}</li>" for i in items]) + "</ul>"

        vision_section = ""
        voice_section = ""
        weather_section = ""
        memory_section = ""
        fusion_section = ""
        if isinstance(agents, dict):
            v = (agents.get("vision") or {}).get("data") if isinstance(agents.get("vision"), dict) else None
            if isinstance(v, dict):
                vision_section = (
                    f"<div class='resqai-section-title'>Vision Agent</div>"
                    f"<div><b>Pred</b>: {v.get('predicted_class')} &nbsp; <b>Conf</b>: {v.get('confidence')}</div>"
                )
            vo = (agents.get("voice") or {}).get("data") if isinstance(agents.get("voice"), dict) else None
            if isinstance(vo, dict):
                voice_section = (
                    f"<div class='resqai-section-title'>Voice Agent</div>"
                    f"<div><b>Urgency</b>: {vo.get('urgency')} &nbsp; <b>Distress</b>: {vo.get('distress_level')}</div>"
                    f"<div class='resqai-muted'>{summarize_text(str(vo.get('transcription') or ''), max_chars=180)}</div>"
                )
            w = (agents.get("weather") or {}).get("data") if isinstance(agents.get("weather"), dict) else None
            if isinstance(w, dict):
                weather_section = (
                    f"<div class='resqai-section-title'>Weather Agent</div>"
                    f"<div><b>Severity</b>: {w.get('weather_severity')} &nbsp; <b>Flood risk</b>: {w.get('flood_risk')}</div>"
                    f"<div><b>Escalation</b>: {w.get('expected_escalation')}</div>"
                    f"<div class='resqai-muted'>{summarize_text(str(w.get('reasoning_summary') or ''), max_chars=180)}</div>"
                )
            mem = (agents.get("memory") or {}).get("data") if isinstance(agents.get("memory"), dict) else None
            if isinstance(mem, dict):
                memory_section = (
                    f"<div class='resqai-section-title'>Memory Agent</div>"
                    f"<div><b>Trend</b>: {mem.get('crisis_trend')} &nbsp; <b>Priority</b>: {mem.get('recommended_priority')}</div>"
                    f"<div class='resqai-muted'>{summarize_text(str(mem.get('reasoning_summary') or ''), max_chars=180)}</div>"
                )
            fu = (agents.get("fusion") or {}).get("data") if isinstance(agents.get("fusion"), dict) else None
            if isinstance(fu, dict):
                fusion_section = (
                    f"<div class='resqai-section-title'>Fusion Coordinator</div>"
                    f"<div><b>Overall</b>: {fu.get('overall_risk')} &nbsp; <b>Trend</b>: {fu.get('crisis_trend')}</div>"
                    f"<div class='resqai-muted'>{summarize_text(str(fu.get('reasoning_summary') or ''), max_chars=180)}</div>"
                )

        popup = (
            "<div class='resqai-popup'>"
            "<div class='resqai-header'>"
            "<div class='resqai-title'>ResQAI Command Brief</div>"
            f"{badge_html}"
            "</div>"
            "<div class='resqai-grid'>"
            "<div class='resqai-card'>"
            "<div class='resqai-section-title'>Situation</div>"
            f"<div><b>Overall risk</b>: {fused.get('overall_risk')}</div>"
            f"<div><b>Crisis trend</b>: {fused.get('crisis_trend')}</div>"
            f"<div><b>Weather escalation</b>: {fused.get('weather_escalation')}</div>"
            "</div>"
            "<div class='resqai-card'>"
            "<div class='resqai-section-title'>Vulnerable Groups</div>"
            f"{_ul(list(fused.get('vulnerable_groups') or []))}"
            "</div>"
            "<div class='resqai-card'>"
            "<div class='resqai-section-title'>Recommended Actions</div>"
            f"{_ul(list(fused.get('recommended_actions') or []))}"
            "</div>"
            "<div class='resqai-card'>"
            "<div class='resqai-section-title'>Operational Notes</div>"
            f"{_ul(list(fused.get('operational_notes') or []))}"
            "</div>"
            "<div class='resqai-card resqai-span-2'>"
            "<div class='resqai-section-title'>AI Coordination</div>"
            f"{vision_section}{voice_section}{weather_section}{memory_section}{fusion_section}"
            "</div>"
            "</div>"
            f"<div class='resqai-summary'><b>Summary</b>: {summarize_text(str(fused.get('reasoning_summary') or ''), max_chars=260)}</div>"
            "</div>"
        )

        add_event(
            anchor_ll,
            "ResQAI command brief",
            popup,
            style.color,
            style.icon,
            kind="marker",
            intensity=style.intensity,
            radius=style.radius,
            weight=style.weight,
        )

        # Add a pulsing critical hotspot overlay at anchor for immediate attention.
        add_event(
            anchor_ll,
            "Crisis hotspot",
            popup,
            style.color,
            style.icon,
            kind="pulse",
            intensity=style.intensity,
            radius=max(26, style.radius + 10),
            weight=style.weight,
        )

    if anchor_ll and isinstance(weather_data, dict):
        style = risk_to_style(weather_data.get("flood_risk") or weather_data.get("weather_severity"))
        popup = "<br/>".join(
            [
                f"<b>Weather</b>: {weather_data.get('weather_severity')}",
                f"<b>Flood risk</b>: {weather_data.get('flood_risk')}",
                f"<b>Escalation</b>: {weather_data.get('expected_escalation')}",
                f"<b>Summary</b>: {summarize_text(str(weather_data.get('reasoning_summary') or ''))}",
            ]
        )
        add_event(anchor_ll, "Weather intelligence", popup, style.color, style.icon, intensity=style.intensity, radius=style.radius, weight=style.weight)

        # Escalation hotspot overlay
        esc_style = escalation_to_style(weather_data.get("expected_escalation"))
        add_event(
            anchor_ll,
            "Escalation indicator",
            popup,
            esc_style.color,
            esc_style.icon,
            kind="circle",
            intensity=esc_style.intensity,
            radius=esc_style.radius,
            weight=esc_style.weight,
        )

    # Vision marker (same anchor point for now, unless geo is present in vision payload later).
    vision_data = None
    if isinstance(agents, dict):
        v = agents.get("vision") or {}
        if isinstance(v, dict) and isinstance(v.get("data"), dict):
            vision_data = v["data"]
    if anchor_ll and isinstance(vision_data, dict):
        style = class_to_style(vision_data.get("predicted_class"))
        popup = "<br/>".join(
            [
                f"<b>Vision</b>: {vision_data.get('predicted_class')}",
                f"<b>Confidence</b>: {vision_data.get('confidence')}",
            ]
        )
        add_event(anchor_ll, "Flood severity (vision)", popup, style.color, style.icon, intensity=style.intensity, radius=style.radius, weight=style.weight)

        # Hotspot indicator (radius scales with confidence)
        conf = vision_data.get("confidence")
        radius = style.radius
        if isinstance(conf, (int, float)):
            radius = int(max(style.radius, 14 + min(1.0, max(0.0, float(conf))) * 26))
        add_event(
            anchor_ll,
            "Flood hotspot",
            popup,
            style.color,
            style.icon,
            kind="circle",
            intensity=style.intensity,
            radius=radius,
            weight=style.weight,
        )

    # Voice marker
    voice_data = None
    if isinstance(agents, dict):
        v = agents.get("voice") or {}
        if isinstance(v, dict) and isinstance(v.get("data"), dict):
            voice_data = v["data"]
    if anchor_ll and isinstance(voice_data, dict):
        style = risk_to_style(voice_data.get("distress_level") or voice_data.get("urgency"))
        popup = "<br/>".join(
            [
                f"<b>Voice urgency</b>: {voice_data.get('urgency')}",
                f"<b>Distress level</b>: {voice_data.get('distress_level')}",
                f"<b>Transcript</b>: {summarize_text(str(voice_data.get('transcription') or ''))}",
            ]
        )
        add_event(anchor_ll, "Distress signal (voice)", popup, style.color, style.icon, intensity=style.intensity, radius=style.radius, weight=style.weight)

    return out
