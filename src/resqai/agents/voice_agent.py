from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from functools import lru_cache
import re

from resqai.agents.base_agent import AgentContext, AgentResult, BaseAgent
from resqai.pipelines.flood_severity.utils import resolve_from_repo
from resqai.pipelines.voice_intelligence.config import VoiceConfig
from resqai.pipelines.voice_intelligence.infer import run_inference
from resqai.pipelines.voice_intelligence.semantic_reasoner import GemmaSemanticReasoner, ReasonerConfig

logger = logging.getLogger("resqai.agents.voice")

@lru_cache(maxsize=4)
def _get_reasoner(model: str) -> GemmaSemanticReasoner:
    return GemmaSemanticReasoner(cfg=ReasonerConfig(model=model))

_WS_RE = re.compile(r"\s+")


def _normalize_transcript(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = _WS_RE.sub(" ", t)
    # Collapse obvious stutters/repeats like "heelllppp" -> "heelpp"
    t = re.sub(r"(.)\1{3,}", r"\1\1", t)
    return t


def _keyword_signals(text: str, language: str | None) -> dict[str, Any]:
    t = (text or "").lower()
    cues = {
        "flood": ["flood", "flooding", "inondation", "inundación", "बाढ़", "बाढ़", "जलभराव", "पानी घर"],
        "trapped": ["trapped", "stuck", "piégé", "atrapado", "फँस", "फंसे", "फसे", "फँसे"],
        "water_entering": ["water is entering", "water entering", "entra agua", "l'eau entre", "पानी घर में घुस", "घर में पानी घुस"],
    }
    hits: dict[str, bool] = {}
    for k, lst in cues.items():
        hits[k] = any(kw in t for kw in lst)
    hits["language"] = language
    return hits


class VoiceAgent(BaseAgent):
    name = "voice_agent"

    def _run(self, ctx: AgentContext, inp: dict[str, Any]) -> dict[str, Any]:
        audio = inp.get("audio")
        if not audio:
            raise ValueError("missing_required_input:audio")

        audio_path = resolve_from_repo(Path(str(audio)))
        whisper_model = str(inp.get("whisper_model", VoiceConfig.whisper_model))
        language = inp.get("language")

        # Semantic reasoner via Ollama Gemma
        semantic = bool(inp.get("semantic_reasoning", True))
        gemma_model = str(inp.get("gemma_model", "gemma4"))

        voice_out = run_inference(audio_path, VoiceConfig(whisper_model=whisper_model, language=language))
        if semantic:
            reasoner = _get_reasoner(gemma_model)
            raw_tx = str(voice_out.get("transcription", "") or "")
            norm_tx = _normalize_transcript(raw_tx)
            lang = voice_out.get("language")
            sem = reasoner.reason(
                transcription=norm_tx or raw_tx,
                language=lang,
                signals={
                    "heuristic_distress_score": voice_out.get("distress_score"),
                    "heuristic_urgency": voice_out.get("urgency"),
                    "raw_transcription": raw_tx[:500],
                    "normalized_transcription": norm_tx[:500],
                    "keyword_signals": _keyword_signals(norm_tx or raw_tx, str(lang) if lang else None),
                },
            )
            voice_out.update(sem.to_dict())
        return voice_out

    def run(self, ctx: AgentContext, inp: dict[str, Any]) -> AgentResult:
        return super().run(ctx, inp)
