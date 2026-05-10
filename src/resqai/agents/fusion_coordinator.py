from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from typing import Any

from resqai.agents.base_agent import AgentContext, AgentResult, BaseAgent
from resqai.pipelines.voice_intelligence.gemma_client import GemmaOllamaClient, OllamaConfig

from resqai.memory.memory_utils import cleanup_json_text, find_json_object, count_sentences

logger = logging.getLogger("resqai.agents.fusion")


@dataclass(frozen=True)
class FusionCoordinatorConfig:
    gemma_model: str = "gemma4"
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9


class FusionCoordinator(BaseAgent):
    name = "fusion_coordinator"

    def __init__(self, cfg: FusionCoordinatorConfig | None = None) -> None:
        self.cfg = cfg or FusionCoordinatorConfig()

    def _run(self, ctx: AgentContext, inp: dict[str, Any]) -> dict[str, Any]:
        vision = inp.get("vision") if isinstance(inp.get("vision"), dict) else None
        voice = inp.get("voice") if isinstance(inp.get("voice"), dict) else None
        weather = inp.get("weather") if isinstance(inp.get("weather"), dict) else None
        rag = inp.get("rag") if isinstance(inp.get("rag"), dict) else None
        memory = inp.get("memory") if isinstance(inp.get("memory"), dict) else None

        if vision is None and voice is None and weather is None and rag is None and memory is None:
            raise ValueError("missing_required_inputs:any_of(vision,voice,weather,rag,memory)")

        schema = {
            "overall_risk": "...",
            "crisis_trend": "...",
            "weather_escalation": True,
            "vulnerable_groups": ["..."],
            "recommended_actions": ["..."],
            "operational_notes": ["..."],
            "reasoning_summary": "...",
        }

        def _signal(d: dict | None, key: str) -> str | None:
            if not isinstance(d, dict):
                return None
            v = d.get(key)
            return str(v).strip() if isinstance(v, str) and v.strip() else None

        derived = {
            "vision_class": _signal(vision, "predicted_class"),
            "vision_confidence": vision.get("confidence") if isinstance(vision, dict) else None,
            "voice_distress": _signal(voice, "distress_level"),
            "voice_urgency": _signal(voice, "urgency"),
            "weather_expected_escalation": weather.get("expected_escalation") if isinstance(weather, dict) else None,
            "weather_flood_risk": _signal(weather, "flood_risk"),
            "memory_crisis_trend": _signal(memory, "crisis_trend"),
            "memory_severity_progression": _signal(memory, "severity_progression"),
            "memory_distress_trend": _signal(memory, "distress_trend"),
            "rag_risk_level": _signal(rag, "risk_level"),
        }

        prompt = (
            "ROLE: You are ResQAI's crisis fusion coordinator for emergency response operations.\n"
            "TASK: Fuse available agent outputs (vision, voice, weather, RAG, memory) into ONE compact operational brief.\n"
            "STYLE: Command-center intelligence. Short, actionable, no chatbot phrasing.\n\n"
            "FUSION RULES (prioritize consistency and safety):\n"
            "- Prioritize multimodal agreement. If vision+voice+memory align on worsening flooding/distress, escalate overall_risk.\n"
            "- If voice indicates water entering homes/trapped people, treat as high/critical unless strong contradictory evidence.\n"
            "- If memory trend indicates escalation/worsening, amplify urgency and operational notes.\n"
            "- If signals conflict, explain the uncertainty briefly in reasoning_summary and choose conservative actions.\n\n"
            "CRITICAL:\n"
            "- Output ONLY valid JSON (no markdown, no extra text).\n"
            "- Output must match EXACTLY this schema (no extra keys):\n"
            f"{schema}\n"
            "- Keep reasoning_summary under 2 sentences.\n"
            "- Use at most 5 items in arrays.\n\n"
            f"derived_signals={derived}\n\n"
            f"vision={vision}\n"
            f"voice={voice}\n"
            f"weather={weather}\n"
            f"rag={rag}\n"
            f"memory={memory}\n"
        )

        def _parse(raw: str) -> dict[str, Any]:
            s = cleanup_json_text(raw)
            if not s:
                raise ValueError("Empty model output.")
            try:
                data = json.loads(s)
                if not isinstance(data, dict):
                    raise ValueError("Output must be a JSON object.")
            except json.JSONDecodeError:
                data = json.loads(find_json_object(s))

            allowed = {
                "overall_risk",
                "crisis_trend",
                "weather_escalation",
                "vulnerable_groups",
                "recommended_actions",
                "operational_notes",
                "reasoning_summary",
            }
            extra = set(data.keys()) - allowed
            if extra:
                raise ValueError(f"Unexpected keys: {sorted(extra)}")

            for k in ("overall_risk", "crisis_trend", "reasoning_summary"):
                v = str(data.get(k, "")).strip()
                if not v:
                    raise ValueError(f"{k} must be non-empty.")
                data[k] = v
            if count_sentences(str(data["reasoning_summary"])) > 2:
                raise ValueError("reasoning_summary must be under 2 sentences.")

            if not isinstance(data.get("weather_escalation"), bool):
                raise ValueError("weather_escalation must be boolean.")

            for lk in ("vulnerable_groups", "recommended_actions", "operational_notes"):
                v = data.get(lk, [])
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    raise ValueError(f"{lk} must be a list of strings.")
                data[lk] = [x.strip() for x in v if x.strip()][:5]

            return data

        client = GemmaOllamaClient(cfg=OllamaConfig())
        raw = client.generate(
            model=self.cfg.gemma_model,
            prompt=prompt,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_new_tokens=self.cfg.max_new_tokens,
            json_only=True,
            stop=["\n\n", "```"],
        )
        try:
            return _parse(raw)
        except Exception:
            repair = (
                "REPAIR TASK: Your previous response was invalid.\n"
                "Return ONLY valid JSON matching the schema EXACTLY.\n\n"
                f"Schema:\n{schema}\n\n"
                f"Previous response:\n{raw}\n"
            )
            raw2 = client.generate(
                model=self.cfg.gemma_model,
                prompt=repair,
                temperature=0.0,
                top_p=0.9,
                max_new_tokens=self.cfg.max_new_tokens,
                json_only=True,
                stop=["\n\n", "```"],
            )
            return _parse(raw2)

    def run(self, ctx: AgentContext, inp: dict[str, Any]) -> AgentResult:
        return super().run(ctx, inp)
