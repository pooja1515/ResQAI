from __future__ import annotations

import time
from typing import Any

from resqai.agents.base_agent import AgentContext, AgentResult, BaseAgent


class WeatherAgent(BaseAgent):
    name = "weather_agent"

    def _run(self, ctx: AgentContext, inp: dict[str, Any]) -> dict[str, Any]:
        location = inp.get("location")
        if not location or not str(location).strip():
            raise ValueError("missing_required_input:location")

        from resqai.weather.weather_agent import WeatherAgentConfig, run_weather_intelligence

        t0 = time.perf_counter()
        gemma_model = str(inp.get("gemma_model", WeatherAgentConfig.gemma_model))
        out = run_weather_intelligence(str(location).strip(), cfg=WeatherAgentConfig(gemma_model=gemma_model))
        return {**out, "_perf": {"weather_total_s": round(time.perf_counter() - t0, 3)}}

    def run(self, ctx: AgentContext, inp: dict[str, Any]) -> AgentResult:
        return super().run(ctx, inp)
