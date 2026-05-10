from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("resqai.geospatial")


@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"input-json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("input-json must be a JSON object.")
    return data


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def try_extract_latlon(obj: dict[str, Any]) -> LatLon | None:
    """Best-effort extraction of coordinates from orchestrator/agent payloads."""
    # Common shapes we produce:
    # - weather subsystem: data.location.latitude/longitude OR geocode.lat/lon
    # - weather subsystem: data.location.lat/lon
    candidates: list[tuple[str, str]] = [
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("lat", "lng"),
    ]

    def _extract(d: dict[str, Any]) -> LatLon | None:
        for a, b in candidates:
            if a in d and b in d and isinstance(d[a], (int, float)) and isinstance(d[b], (int, float)):
                return LatLon(lat=float(d[a]), lon=float(d[b]))
        return None

    # direct
    ll = _extract(obj)
    if ll:
        return ll

    # nested common keys
    for key in ("location", "geocode", "coords", "coordinates"):
        v = obj.get(key)
        if isinstance(v, dict):
            ll = _extract(v)
            if ll:
                return ll

    return None


def summarize_text(text: str, *, max_chars: int = 180) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."

