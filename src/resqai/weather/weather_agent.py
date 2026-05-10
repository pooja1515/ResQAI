from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from resqai.pipelines.voice_intelligence.gemma_client import OllamaConfig
from resqai.weather.risk_reasoner import WeatherReasonerConfig, reason_weather_risk
from resqai.weather.utils import dumps_json
from resqai.weather.weather_client import OpenWeatherError, config_from_env, fetch_weather_bundle

logger = logging.getLogger("resqai.weather")


@dataclass(frozen=True)
class WeatherAgentConfig:
    gemma_model: str = "gemma4"


def run_weather_intelligence(location: str, *, cfg: WeatherAgentConfig) -> dict[str, object]:
    ow_cfg = config_from_env()
    bundle = fetch_weather_bundle(location, ow_cfg)

    intel = reason_weather_risk(
        location=location,
        weather_payload=bundle,
        cfg=WeatherReasonerConfig(model=cfg.gemma_model),
        ollama=OllamaConfig(),
    )
    return intel.to_dict()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="ResQAI weather intelligence (OpenWeatherMap + Gemma via Ollama).")
    p.add_argument("--location", type=str, required=True)
    p.add_argument("--model", type=str, default=WeatherAgentConfig.gemma_model, help="Ollama model name (gemma4, gemma3:4b).")
    args = p.parse_args()

    try:
        out = run_weather_intelligence(args.location, cfg=WeatherAgentConfig(gemma_model=args.model))
    except OpenWeatherError as exc:
        logger.error("%s", exc)
        raise
    print(dumps_json(out))


if __name__ == "__main__":
    main()

