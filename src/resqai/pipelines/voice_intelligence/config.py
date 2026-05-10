from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceConfig:
    # Whisper model name. Common options: tiny, base, small, medium, large
    whisper_model: str = "small"

    # Optional forced language for Whisper ("en", "hi", "fr", "es"). If None, auto-detect.
    language: str | None = None

    # When True, attempts to use a transformers-based classifier if available.
    use_transformers: bool = False

    # Controls heuristic sensitivity when transformers are not used.
    distress_keyword_weight: float = 0.18
    distress_base: float = 0.05
