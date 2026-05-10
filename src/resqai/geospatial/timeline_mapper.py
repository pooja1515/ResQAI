from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from resqai.geospatial.geo_utils import LatLon, summarize_text
from resqai.geospatial.risk_visualizer import risk_to_style

logger = logging.getLogger("resqai.geospatial")


@dataclass(frozen=True)
class TimelineMarker:
    latlon: LatLon
    label: str
    color: str
    popup_html: str
    timestamp: str | None = None
    severity: str | None = None
    kind: str = "circle"
    radius: int = 16
    weight: int = 4


def build_timeline_markers(
    *,
    anchor: LatLon,
    memory_insight: dict[str, Any] | None,
) -> list[TimelineMarker]:
    # For v1, map only the latest insight at anchor location.
    if not isinstance(memory_insight, dict):
        return []
    style = risk_to_style(memory_insight.get("recommended_priority"))
    sev = str(memory_insight.get("severity_progression") or "") or None
    popup = "<br/>".join(
        [
            f"<b>Crisis trend</b>: {memory_insight.get('crisis_trend')}",
            f"<b>Severity</b>: {memory_insight.get('severity_progression')}",
            f"<b>Distress</b>: {memory_insight.get('distress_trend')}",
            f"<b>Priority</b>: {memory_insight.get('recommended_priority')}",
            f"<b>Summary</b>: {summarize_text(str(memory_insight.get('reasoning_summary') or ''))}",
        ]
    )
    ts = memory_insight.get("timestamp") or memory_insight.get("ts")
    return [
        TimelineMarker(
            latlon=anchor,
            label="Memory escalation",
            color=style.color,
            popup_html=popup,
            timestamp=str(ts) if ts else None,
            severity=sev,
            kind="circle",
            radius=max(16, style.radius - 6),
            weight=style.weight,
        )
    ]
