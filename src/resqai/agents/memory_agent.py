from __future__ import annotations

import logging
import time
from typing import Any

from resqai.agents.base_agent import AgentContext, AgentResult, BaseAgent
from resqai.memory.memory_reasoner import MemoryReasonerConfig, reason_over_memory

logger = logging.getLogger("resqai.agents.memory")


class MemoryAgent(BaseAgent):
    name = "memory_agent"

    def _run(self, ctx: AgentContext, inp: dict[str, Any]) -> dict[str, Any]:
        latest_event = inp.get("latest_event")
        if not isinstance(latest_event, dict):
            raise ValueError("missing_required_input:latest_event")

        model = str(inp.get("gemma_model", "gemma4"))

        t0 = time.perf_counter()
        insight = reason_over_memory(latest_event=latest_event, cfg=MemoryReasonerConfig(model=model))
        out = insight.to_dict()
        out["_perf"] = {"memory_total_s": round(time.perf_counter() - t0, 3)}
        return out

    def run(self, ctx: AgentContext, inp: dict[str, Any]) -> AgentResult:
        return super().run(ctx, inp)
