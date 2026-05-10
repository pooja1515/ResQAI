from __future__ import annotations

import argparse
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from resqai.agents.agent_registry import default_registry
from resqai.agents.base_agent import AgentContext, dumps_json
from resqai.memory.memory_utils import utc_now_iso
from resqai.pipelines.flood_severity.utils import resolve_from_repo

logger = logging.getLogger("resqai.orchestrator")


def _make_query(vision: dict[str, Any] | None, voice: dict[str, Any] | None) -> str:
    parts: list[str] = []
    if voice and voice.get("transcription"):
        parts.append(str(voice.get("transcription")))
    if voice and voice.get("urgency"):
        parts.append(f"urgency={voice.get('urgency')}")
    if vision and vision.get("predicted_class"):
        parts.append(f"vision={vision.get('predicted_class')} ({vision.get('confidence')})")
    return " | ".join(parts) if parts else "disaster guidance"


def _fallback_final() -> dict[str, Any]:
    return {
        "overall_risk": "unknown",
        "crisis_trend": "unknown",
        "weather_escalation": False,
        "vulnerable_groups": [],
        "recommended_actions": [],
        "operational_notes": [],
        "reasoning_summary": "Insufficient agent outputs to produce a fused assessment.",
    }


def run_pipeline(image: Path | None, audio: Path | None, location: str | None, *, debug: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    ctx = AgentContext(request_id=str(uuid.uuid4()), shared={})
    reg = default_registry()

    agent_debug: dict[str, Any] = {"request_id": ctx.request_id, "agents": {}}

    vision_res = None
    voice_res = None
    weather_res = None

    # Run independent agents concurrently.
    futures = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        if image is not None:
            futures[ex.submit(reg.vision.run, ctx, {"image": image, "explain": True})] = "vision"
        if audio is not None:
            futures[ex.submit(reg.voice.run, ctx, {"audio": audio, "semantic_reasoning": True})] = "voice"
        if location:
            futures[ex.submit(reg.weather.run, ctx, {"location": location})] = "weather"

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                # Failure isolation: record as error but continue.
                agent_debug["agents"][name] = {
                    "agent": name,
                    "ok": False,
                    "data": None,
                    "error": str(exc),
                    "attempts": 1,
                    "duration_s": 0.0,
                }
                continue
            agent_debug["agents"][name] = res.to_dict()
            if name == "vision":
                vision_res = res
            elif name == "voice":
                voice_res = res
            elif name == "weather":
                weather_res = res

    # RAG uses a query constructed from available signals.
    rag_res = reg.rag.run(
        ctx,
        {
            "query": _make_query(vision_res.data if vision_res and vision_res.ok else None, voice_res.data if voice_res and voice_res.ok else None),
        },
    )
    agent_debug["agents"]["rag"] = rag_res.to_dict()

    # Memory reasoning over the latest combined event.
    latest_event: dict[str, Any] = {
        "event_type": "multimodal",
        "timestamp": utc_now_iso(),
        "vision": vision_res.data if vision_res and vision_res.ok else None,
        "voice": voice_res.data if voice_res and voice_res.ok else None,
        "weather": weather_res.data if weather_res and weather_res.ok else None,
        "rag": rag_res.data if rag_res and rag_res.ok else None,
    }
    memory_res = reg.memory.run(ctx, {"latest_event": latest_event})
    agent_debug["agents"]["memory"] = memory_res.to_dict()

    # Final fusion: combine all available agent outputs (each may be None).
    fusion_payload = {
        "vision": vision_res.data if vision_res and vision_res.ok else None,
        "voice": voice_res.data if voice_res and voice_res.ok else None,
        "weather": weather_res.data if weather_res and weather_res.ok else None,
        "rag": rag_res.data if rag_res and rag_res.ok else None,
        "memory": memory_res.data if memory_res and memory_res.ok else None,
    }
    fusion_res = reg.fusion.run(ctx, fusion_payload)
    agent_debug["agents"]["fusion"] = fusion_res.to_dict()

    final = fusion_res.data if fusion_res and fusion_res.ok and isinstance(fusion_res.data, dict) else _fallback_final()
    total_s = time.perf_counter() - t0
    logger.info("orchestration_done request_id=%s duration_s=%.3f", ctx.request_id, total_s)
    if debug:
        final = {**final, "_debug": agent_debug}
    return final


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="ResQAI multi-agent orchestrator (vision+voice+rag+weather+fusion).")
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--audio", type=Path, default=None)
    p.add_argument("--location", type=str, default=None)
    p.add_argument("--debug", action="store_true", help="Include per-agent debug output.")
    args = p.parse_args()

    image = resolve_from_repo(args.image) if args.image else None
    audio = resolve_from_repo(args.audio) if args.audio else None
    out = run_pipeline(image=image, audio=audio, location=args.location, debug=bool(args.debug))
    print(dumps_json(out))


if __name__ == "__main__":
    main()
