from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from resqai.agents.agent_registry import default_registry
from resqai.agents.base_agent import AgentContext
from resqai.pipelines.voice_intelligence.semantic_reasoner import (
    GemmaSemanticReasoner,
    ReasonerConfig,
)

logger = logging.getLogger("resqai.api.analyze")

router = APIRouter(prefix="/api", tags=["analyze"])


class AnalyzeRequest(BaseModel):
    message: str = Field(..., min_length=1)
    location: str | None = None


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _format_intelligence(fused: dict[str, Any]) -> str:
    # Keep it conversational/operational, no JSON.
    overall = str(fused.get("overall_risk") or "unknown").strip()
    trend = str(fused.get("crisis_trend") or "unknown").strip()
    escalation = fused.get("weather_escalation")
    actions = fused.get("recommended_actions") or []
    notes = fused.get("operational_notes") or []

    lines: list[str] = []
    lines.append(f"{overall.title()} flood conditions detected.")
    if trend and trend != "unknown":
        lines.append(f"Crisis trend: {trend}.")
    if isinstance(escalation, bool) and escalation:
        lines.append("Weather indicates potential escalation—prioritize timely evacuation and rescue readiness.")

    lines.append("")
    if isinstance(actions, list) and actions:
        lines.append("Recommended actions:")
        for a in actions[:4]:
            if isinstance(a, str) and a.strip():
                lines.append(f"- {a.strip()}")
    if isinstance(notes, list) and notes:
        lines.append("")
        lines.append("Operational notes:")
        for n in notes[:3]:
            if isinstance(n, str) and n.strip():
                lines.append(f"- {n.strip()}")

    summary = str(fused.get("reasoning_summary") or "").strip()
    if summary:
        lines.append("")
        lines.append(summary)
    return "\n".join(lines).strip()


@router.post("/analyze")
async def analyze(req: AnalyzeRequest) -> StreamingResponse:
    async def safe_stream() -> AsyncGenerator[str, None]:
        try:
            async for chunk in _event_stream_impl(req):
                yield chunk
        except Exception as exc:  # noqa: BLE001
            logger.exception("analyze_stream_failed")
            yield _sse("error", {"message": "Backend error while streaming response."})
            yield _sse("done", {"ok": False})

    return StreamingResponse(
        safe_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Content-Type-Options": "nosniff",
        },
    )


async def _event_stream_impl(req: AnalyzeRequest) -> AsyncGenerator[str, None]:
    reg = default_registry()
    ctx = AgentContext(request_id=f"api-{int(time.time())}", shared={})

    message = req.message.strip()
    location = req.location.strip() if req.location else None

    def sse(event: str, data: dict[str, Any]) -> str:
        return _sse(event, data)

    yield sse("status", {"text": "Analyzing crisis signals…"})
    yield sse("status", {"text": "Interpreting emergency semantics…"})

    voice_like: dict[str, Any]
    try:
        reasoner = GemmaSemanticReasoner(cfg=ReasonerConfig(model=os.getenv("RESQAI_GEMMA_MODEL", "gemma4")))
        sem = await asyncio.to_thread(reasoner.reason, message, None, {"input_mode": "text"})
        voice_like = sem.to_dict()
        voice_like["transcription"] = message
        yield sse("status", {"text": "Distress assessment complete."})
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic_reasoning_failed: %s", exc)
        voice_like = {"transcription": message}
        yield sse("status", {"text": "Continuing without semantic distress model."})

    weather_data = None
    if location:
        yield sse("status", {"text": "Retrieving weather intelligence…"})
        wres = await asyncio.to_thread(reg.weather.run, ctx, {"location": location})
        if wres.ok and isinstance(wres.data, dict):
            weather_data = wres.data
            yield sse("status", {"text": "Weather intelligence retrieved."})
        else:
            yield sse("status", {"text": "Weather intelligence unavailable (continuing)."})

    yield sse("status", {"text": "Retrieving grounded safety guidance…"})
    rag = await asyncio.to_thread(reg.rag.run, ctx, {"query": message})
    rag_data = rag.data if rag.ok and isinstance(rag.data, dict) else None
    yield sse(
        "status",
        {"text": "Grounded guidance ready." if rag_data else "No grounded sources found (continuing)."},
    )

    yield sse("status", {"text": "Updating temporal memory…"})
    latest_event = {
        "event_type": "text_orchestration",
        "timestamp": time.time(),
        "message": message,
        "location": location,
        "weather": weather_data,
        "rag": rag_data,
        "semantic": voice_like,
    }
    mem = await asyncio.to_thread(reg.memory.run, ctx, {"latest_event": latest_event})
    mem_data = mem.data if mem.ok and isinstance(mem.data, dict) else None
    yield sse("status", {"text": "Memory correlation complete." if mem_data else "Memory unavailable (continuing)."})

    yield sse("status", {"text": "Synthesizing operational intelligence…"})
    fusion = await asyncio.to_thread(
        reg.fusion.run,
        ctx,
        {"voice": voice_like, "weather": weather_data, "rag": rag_data, "memory": mem_data},
    )
    fused = fusion.data if fusion.ok and isinstance(fusion.data, dict) else None
    if not fused:
        yield sse("error", {"message": "Unable to synthesize intelligence at this time."})
        yield sse("done", {"ok": False})
        return

    final_text = _format_intelligence(fused)
    yield sse("status", {"text": "Finalizing response…"})

    # Token stream
    for tok in _split_tokens(final_text):
        yield sse("token", {"text": tok})
        await asyncio.sleep(0.01)

    yield sse("done", {"ok": True})


def _split_tokens(text: str) -> list[str]:
    # Split while preserving whitespace to render naturally
    out: list[str] = []
    buf = ""
    for ch in text:
        if ch.isspace():
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out
