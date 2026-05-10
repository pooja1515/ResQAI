from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from resqai.memory.memory_store import MemoryStore, MemoryStoreConfig
from resqai.memory.memory_utils import utc_now_iso

logger = logging.getLogger("resqai.memory")


@dataclass(frozen=True)
class EventTrackerConfig:
    store: MemoryStoreConfig = MemoryStoreConfig()


def _infer_event_type(payload: dict[str, Any]) -> str:
    et = payload.get("event_type") or payload.get("type")
    if isinstance(et, str) and et.strip():
        return et.strip()
    if "predicted_class" in payload:
        return "vision"
    if "transcription" in payload:
        return "voice"
    if "forecast_summary" in payload or "current" in payload:
        return "weather"
    if "recommended_actions" in payload and "safety_guidelines" in payload:
        return "rag"
    return "unknown"


def _infer_labels(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    severity = None
    urgency = None
    for key in ("distress_level", "weather_severity", "flood_risk", "risk_level", "overall_risk"):
        v = payload.get(key)
        if isinstance(v, str) and v.strip() and severity is None:
            severity = v.strip()
    for key in ("urgency", "rescue_priority"):
        v = payload.get(key)
        if isinstance(v, str) and v.strip() and urgency is None:
            urgency = v.strip()
    return severity, urgency


class EventTracker:
    def __init__(self, cfg: EventTrackerConfig | None = None) -> None:
        self.cfg = cfg or EventTrackerConfig()
        self.store = MemoryStore(self.cfg.store)

    def close(self) -> None:
        self.store.close()

    def record(self, payload: dict[str, Any], *, source: str | None = None) -> int:
        event_type = _infer_event_type(payload)
        severity_label, urgency_label = _infer_labels(payload)
        ts = str(payload.get("timestamp") or payload.get("ts") or utc_now_iso())
        return self.store.insert_event(
            ts=ts,
            event_type=event_type,
            payload=payload,
            severity_label=severity_label,
            urgency_label=urgency_label,
            source=source,
        )

