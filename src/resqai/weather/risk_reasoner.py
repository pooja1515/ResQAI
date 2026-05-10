from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from resqai.pipelines.voice_intelligence.gemma_client import GemmaOllamaClient, OllamaConfig
from resqai.weather.utils import (
    WeatherIntelligence,
    as_list_of_str,
    cleanup_json_text,
    count_sentences,
    find_json_object,
)
from resqai.weather.weather_prompts import build_repair_prompt, build_weather_prompt

logger = logging.getLogger("resqai.weather")


@dataclass(frozen=True)
class WeatherReasonerConfig:
    model: str = "gemma4"
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 256


def _parse_weather_json(raw: str) -> WeatherIntelligence:
    s = cleanup_json_text(raw)
    if not s:
        raise ValueError("Empty model output.")
    try:
        parsed = json.loads(s)
        if not isinstance(parsed, dict):
            raise ValueError("Model output must be a JSON object.")
        data = parsed
    except json.JSONDecodeError:
        data = json.loads(find_json_object(s))

    allowed = {
        "location",
        "weather_severity",
        "flood_risk",
        "expected_escalation",
        "environmental_risks",
        "recommended_actions",
        "reasoning_summary",
    }
    extra = set(data.keys()) - allowed
    if extra:
        raise ValueError(f"Unexpected keys: {sorted(extra)}")

    location = str(data.get("location", "")).strip()
    ws = str(data.get("weather_severity", "")).strip().lower()
    fr = str(data.get("flood_risk", "")).strip().lower()
    expected = data.get("expected_escalation")
    env = as_list_of_str(data.get("environmental_risks", []))[:4]
    acts = as_list_of_str(data.get("recommended_actions", []))[:4]
    summary = str(data.get("reasoning_summary", "")).strip()

    if not location:
        raise ValueError("location must be non-empty.")
    if ws not in {"low", "moderate", "high", "severe"}:
        raise ValueError("weather_severity must be low|moderate|high|severe")
    if fr not in {"low", "moderate", "high", "critical"}:
        raise ValueError("flood_risk must be low|moderate|high|critical")
    if not isinstance(expected, bool):
        raise ValueError("expected_escalation must be boolean.")
    if not summary:
        raise ValueError("reasoning_summary must be non-empty.")
    if count_sentences(summary) > 2:
        raise ValueError("reasoning_summary must be under 2 sentences.")

    return WeatherIntelligence(
        location=location,
        weather_severity=ws,
        flood_risk=fr,
        expected_escalation=expected,
        environmental_risks=env,
        recommended_actions=acts,
        reasoning_summary=summary,
    )


def reason_weather_risk(
    *,
    location: str,
    weather_payload: dict,
    cfg: WeatherReasonerConfig,
    ollama: OllamaConfig | None = None,
) -> WeatherIntelligence:
    client = GemmaOllamaClient(cfg=ollama)
    prompt = build_weather_prompt(location=location, weather_payload=weather_payload)

    last_raw: str | None = None
    last_exc: Exception | None = None
    for attempt in range(3):
        raw = client.generate(
            model=cfg.model,
            prompt=prompt if attempt == 0 else build_repair_prompt(
                location=location, weather_payload=weather_payload, bad_output=(last_raw or "")[:2000]
            ),
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            json_only=True,
            stop=["\n\n", "```"],
        )
        last_raw = raw
        try:
            return _parse_weather_json(raw)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Weather JSON parse/validation failed (attempt %s/3): %s", attempt + 1, exc)

    raise RuntimeError("Gemma did not return valid weather JSON after retries.") from last_exc

