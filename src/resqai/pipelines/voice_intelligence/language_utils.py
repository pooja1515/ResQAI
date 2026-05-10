from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageInfo:
    code: str
    name: str


SUPPORTED_LANGUAGES: dict[str, LanguageInfo] = {
    "en": LanguageInfo(code="en", name="English"),
    "hi": LanguageInfo(code="hi", name="Hindi"),
    "fr": LanguageInfo(code="fr", name="French"),
    "es": LanguageInfo(code="es", name="Spanish"),
}


def normalize_language_code(code: str | None) -> str | None:
    if code is None:
        return None
    code = code.strip().lower()
    if not code:
        return None
    return code


def is_supported_language(code: str | None) -> bool:
    code = normalize_language_code(code)
    return code in SUPPORTED_LANGUAGES

