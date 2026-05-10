from __future__ import annotations

import logging
import time
from typing import Any

from resqai.agents.base_agent import AgentContext, AgentResult, BaseAgent
from resqai.rag.rag_reasoner import RAGReasonerConfig, answer_query
from resqai.rag.vector_store import VectorStoreConfig

logger = logging.getLogger("resqai.agents.rag")


class RAGAgent(BaseAgent):
    name = "rag_agent"

    def _run(self, ctx: AgentContext, inp: dict[str, Any]) -> dict[str, Any]:
        query = inp.get("query")
        if not query or not str(query).strip():
            raise ValueError("missing_required_input:query")

        model = str(inp.get("gemma_model", "gemma4"))
        k = int(inp.get("k", 3))

        t0 = time.perf_counter()
        out = answer_query(
            str(query),
            cfg=RAGReasonerConfig(model=model, k=k),
            vs_cfg=VectorStoreConfig(),
        )
        logger.info("rag_answered duration_s=%.3f k=%s", time.perf_counter() - t0, k)
        return out

    def run(self, ctx: AgentContext, inp: dict[str, Any]) -> AgentResult:
        return super().run(ctx, inp)
