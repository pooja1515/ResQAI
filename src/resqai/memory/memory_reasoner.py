from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from resqai.pipelines.voice_intelligence.gemma_client import GemmaOllamaClient, OllamaConfig
from resqai.memory.event_tracker import EventTracker
from resqai.memory.memory_utils import MemoryInsight, dumps_json, parse_memory_insight, resolve_from_repo
from resqai.memory.timeline_manager import TimelineManager

logger = logging.getLogger("resqai.memory")


PROMPT_SCHEMA = """Return ONLY valid JSON (no markdown) matching EXACTLY:
{
  "crisis_trend": "...",
  "severity_progression": "...",
  "distress_trend": "...",
  "recommended_priority": "...",
  "reasoning_summary": "..."
}

Rules:
- Output must be a single JSON object with ONLY these keys.
- Keep output compact (<= ~1200 characters).
- Keep `reasoning_summary` under 2 sentences.
- Base your answer ONLY on the event timeline provided (no invented facts).
"""


SYSTEM = """You are ResQAI's temporal crisis intelligence analyst.
You receive a timeline of multimodal disaster events (vision, voice distress, weather, RAG reasoning).
Infer trends: escalation, stabilization, or deterioration; note severity progression and distress/urgency patterns.
Provide an operational recommended priority (e.g., "immediate rescue", "evacuate", "monitor", "medical triage").
Be concise and grounded in the timeline.
"""


def build_prompt(event_timeline: list[str], latest_event: dict) -> str:
    timeline_block = "\n".join([f"- {line}" for line in event_timeline]) or "(no history)"
    return (
        f"{SYSTEM}\n\n"
        "CRITICAL: Reply with ONLY the JSON object, no extra text.\n\n"
        f"Latest event (JSON): {latest_event}\n\n"
        f"Event timeline (most recent first):\n{timeline_block}\n\n"
        f"{PROMPT_SCHEMA}\n"
    )


def build_repair_prompt(bad_output: str, event_timeline: list[str], latest_event: dict) -> str:
    timeline_block = "\n".join([f"- {line}" for line in event_timeline]) or "(no history)"
    return (
        f"{SYSTEM}\n\n"
        "You previously returned an invalid response.\n"
        "Rewrite it as STRICT JSON matching the schema. Output ONLY JSON.\n\n"
        f"Latest event (JSON): {latest_event}\n\n"
        f"Event timeline (most recent first):\n{timeline_block}\n\n"
        f"Invalid response:\n{bad_output}\n\n"
        f"{PROMPT_SCHEMA}\n"
    )


@dataclass(frozen=True)
class MemoryReasonerConfig:
    model: str = "gemma4"
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 256


def reason_over_memory(*, latest_event: dict, cfg: MemoryReasonerConfig) -> MemoryInsight:
    tracker = EventTracker()
    timeline = TimelineManager()
    try:
        tracker.record(latest_event, source="cli")
        events = timeline.recent_events()
        context = timeline.to_context(events)
    finally:
        tracker.close()
        timeline.close()

    client = GemmaOllamaClient(cfg=OllamaConfig())
    prompt = build_prompt(context, latest_event)

    last_raw: str | None = None
    last_exc: Exception | None = None
    for attempt in range(3):
        raw = client.generate(
            model=cfg.model,
            prompt=prompt if attempt == 0 else build_repair_prompt((last_raw or "")[:2000], context, latest_event),
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            json_only=True,
            stop=["\n\n", "```"],
        )
        last_raw = raw
        try:
            return parse_memory_insight(raw)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Memory JSON parse/validation failed (attempt %s/3): %s", attempt + 1, exc)

    raise RuntimeError("Gemma did not return valid memory JSON after retries.") from last_exc


def _load_event_json(path: Path) -> dict:
    path = resolve_from_repo(path)
    if not path.exists():
        raise FileNotFoundError(f"event-json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("event-json must contain a JSON object.")
    return data


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="ResQAI temporal crisis memory reasoner (SQLite + Gemma via Ollama).")
    p.add_argument("--event-json", type=Path, required=True)
    p.add_argument("--model", type=str, default=MemoryReasonerConfig.model, help="Ollama model name (gemma4, gemma3:4b).")
    args = p.parse_args()

    latest_event = _load_event_json(args.event_json)
    insight = reason_over_memory(latest_event=latest_event, cfg=MemoryReasonerConfig(model=args.model))
    print(dumps_json(insight.to_dict()))


if __name__ == "__main__":
    main()
