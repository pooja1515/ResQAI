from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SupportedLanguages:
    # Placeholder for future locale negotiation / translation components.
    english: str = "en"
    hindi: str = "hi"
    french: str = "fr"
    spanish: str = "es"


SUPPORTED_LANGUAGES = SupportedLanguages()

