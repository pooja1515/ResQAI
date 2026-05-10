from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger("resqai.weather")


class OpenWeatherError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenWeatherConfig:
    api_key: str
    base_url: str = "https://api.openweathermap.org"
    timeout_s: float = 12.0
    retries: int = 2
    backoff_s: float = 0.8
    units: str = "metric"


def config_from_env() -> OpenWeatherConfig:
    key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not key:
        raise OpenWeatherError("OPENWEATHER_API_KEY is not set.")
    return OpenWeatherConfig(api_key=key)


def _get_json(url: str, *, timeout_s: float) -> dict[str, Any]:
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _get_with_retries(url: str, cfg: OpenWeatherConfig) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(cfg.retries + 1):
        try:
            return _get_json(url, timeout_s=cfg.timeout_s)
        except HTTPError as exc:
            last_exc = exc
            # 401 is invalid API key; do not retry aggressively.
            if exc.code == 401:
                raise OpenWeatherError("Invalid OPENWEATHER_API_KEY (HTTP 401).") from exc
            if attempt >= cfg.retries:
                break
            time.sleep(cfg.backoff_s * (2**attempt))
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt >= cfg.retries:
                break
            time.sleep(cfg.backoff_s * (2**attempt))
    raise OpenWeatherError("OpenWeatherMap request failed after retries.") from last_exc


def geocode_location(name: str, cfg: OpenWeatherConfig) -> dict[str, Any]:
    qs = urlencode({"q": name, "limit": 1, "appid": cfg.api_key})
    url = f"{cfg.base_url}/geo/1.0/direct?{qs}"
    data = _get_with_retries(url, cfg)
    if not isinstance(data, list) or not data:
        raise OpenWeatherError(f"Location not found: {name}")
    top = data[0]
    return {
        "name": top.get("name"),
        "lat": float(top["lat"]),
        "lon": float(top["lon"]),
        "country": top.get("country"),
        "state": top.get("state"),
    }


def fetch_current(lat: float, lon: float, cfg: OpenWeatherConfig) -> dict[str, Any]:
    qs = urlencode({"lat": lat, "lon": lon, "appid": cfg.api_key, "units": cfg.units})
    url = f"{cfg.base_url}/data/2.5/weather?{qs}"
    return _get_with_retries(url, cfg)


def fetch_forecast(lat: float, lon: float, cfg: OpenWeatherConfig) -> dict[str, Any]:
    # 5 day / 3 hour forecast
    qs = urlencode({"lat": lat, "lon": lon, "appid": cfg.api_key, "units": cfg.units})
    url = f"{cfg.base_url}/data/2.5/forecast?{qs}"
    return _get_with_retries(url, cfg)


def summarize_forecast(forecast: dict[str, Any], *, horizon_steps: int = 8) -> dict[str, Any]:
    """Summarize next ~24h (8 * 3h) forecast in a compact form."""
    items = forecast.get("list") or []
    if not isinstance(items, list):
        items = []
    items = items[:horizon_steps]

    total_rain_mm = 0.0
    max_wind = 0.0
    min_temp = None
    max_temp = None
    conditions: set[str] = set()

    for it in items:
        if not isinstance(it, dict):
            continue
        main = it.get("main") or {}
        wind = it.get("wind") or {}
        weather = it.get("weather") or []
        if isinstance(weather, list) and weather:
            desc = weather[0].get("main") or weather[0].get("description")
            if isinstance(desc, str) and desc:
                conditions.add(desc)

        t = main.get("temp")
        if isinstance(t, (int, float)):
            min_temp = t if min_temp is None else min(min_temp, t)
            max_temp = t if max_temp is None else max(max_temp, t)

        ws = wind.get("speed")
        if isinstance(ws, (int, float)):
            max_wind = max(max_wind, float(ws))

        rain = it.get("rain") or {}
        if isinstance(rain, dict):
            r3 = rain.get("3h")
            if isinstance(r3, (int, float)):
                total_rain_mm += float(r3)

    return {
        "next_24h_total_rain_mm": round(total_rain_mm, 2),
        "next_24h_max_wind_mps": round(max_wind, 2),
        "next_24h_min_temp_c": round(min_temp, 1) if isinstance(min_temp, (int, float)) else None,
        "next_24h_max_temp_c": round(max_temp, 1) if isinstance(max_temp, (int, float)) else None,
        "notable_conditions": sorted(list(conditions))[:4],
    }


def fetch_weather_bundle(location: str, cfg: OpenWeatherConfig) -> dict[str, Any]:
    geo = geocode_location(location, cfg)
    cur = fetch_current(geo["lat"], geo["lon"], cfg)
    fc = fetch_forecast(geo["lat"], geo["lon"], cfg)
    summary = summarize_forecast(fc)

    # Extract compact fields for reasoning (avoid huge payloads).
    main = cur.get("main") or {}
    wind = cur.get("wind") or {}
    rain = cur.get("rain") or {}
    weather = cur.get("weather") or []
    cond = ""
    if isinstance(weather, list) and weather:
        cond = str(weather[0].get("description") or weather[0].get("main") or "")

    return {
        "geocode": geo,
        "current": {
            "temperature_c": main.get("temp"),
            "humidity_pct": main.get("humidity"),
            "wind_speed_mps": wind.get("speed"),
            "rain_1h_mm": (rain.get("1h") if isinstance(rain, dict) else None),
            "rain_3h_mm": (rain.get("3h") if isinstance(rain, dict) else None),
            "condition": cond,
        },
        "forecast_summary": summary,
    }

