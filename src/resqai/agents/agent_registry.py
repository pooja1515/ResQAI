from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from resqai.agents.fusion_coordinator import FusionCoordinator
from resqai.agents.memory_agent import MemoryAgent
from resqai.agents.rag_agent import RAGAgent
from resqai.agents.vision_agent import VisionAgent
from resqai.agents.voice_agent import VoiceAgent
from resqai.agents.weather_agent import WeatherAgent


@dataclass(frozen=True)
class AgentRegistry:
    vision: VisionAgent
    voice: VoiceAgent
    rag: RAGAgent
    weather: WeatherAgent
    memory: MemoryAgent
    fusion: FusionCoordinator


def default_registry() -> AgentRegistry:
    return _default_registry()


@lru_cache(maxsize=1)
def _default_registry() -> AgentRegistry:
    # Cache across calls in a long-lived process to avoid repeated init costs.
    return AgentRegistry(
        vision=VisionAgent(),
        voice=VoiceAgent(),
        rag=RAGAgent(),
        weather=WeatherAgent(),
        memory=MemoryAgent(),
        fusion=FusionCoordinator(),
    )
