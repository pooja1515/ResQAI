from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("resqai.voice_intelligence")


@dataclass(frozen=True)
class DistressResult:
    distress_score: float  # 0..1
    urgency: str  # low|medium|high


_KEYWORDS: dict[str, list[str]] = {
    # English
    "en": [
        "help",
        "emergency",
        "urgent",
        "fire",
        "flood",
        "trapped",
        "stuck",
        "injured",
        "bleeding",
        "rescue",
        "evacuate",
        "danger",
        "collapse",
    ],
    # Hindi (basic, common terms)
    "hi": [
        "madad",
        "bachao",
        "aapaat",
        "aapda",
        "jaldi",
        "badh",
        "pani",
        "fas",
        "ghayal",
        "bachaav",
        "nikasi",
        "khatra",
    ],
    # French
    "fr": [
        "aidez",
        "aide",
        "urgence",
        "secours",
        "inondation",
        "piégé",
        "bloqué",
        "blessé",
        "évacuer",
        "danger",
    ],
    # Spanish
    "es": [
        "ayuda",
        "emergencia",
        "urgente",
        "socorro",
        "inundación",
        "atrapado",
        "bloqueado",
        "herido",
        "evacuar",
        "peligro",
    ],
}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def score_distress_heuristic(text: str, language: str | None, *, base: float, weight: float) -> DistressResult:
    text_norm = (text or "").strip().lower()
    if not text_norm:
        return DistressResult(distress_score=0.0, urgency="low")

    lang = (language or "").strip().lower()
    keywords = _KEYWORDS.get(lang, _KEYWORDS["en"])

    hits = 0
    for kw in keywords:
        if kw in text_norm:
            hits += 1

    score = base + hits * weight
    score = _clip01(score)

    if score >= 0.7:
        urgency = "high"
    elif score >= 0.35:
        urgency = "medium"
    else:
        urgency = "low"

    return DistressResult(distress_score=score, urgency=urgency)


def score_distress_transformers(text: str) -> DistressResult | None:
    """Optional transformers-based distress scoring.

    This is best-effort and may require model downloads. If unavailable, returns None.
    """
    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        return None

    try:
        clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        labels = ["high urgency emergency", "low urgency non-emergency"]
        out = clf(text, candidate_labels=labels, multi_label=False)
        scores = {lab: float(score) for lab, score in zip(out["labels"], out["scores"])}
        high = scores.get("high urgency emergency", 0.0)
        score = _clip01(high)
        urgency = "high" if score >= 0.7 else ("medium" if score >= 0.35 else "low")
        return DistressResult(distress_score=score, urgency=urgency)
    except Exception:
        logger.exception("Transformers-based distress scoring failed; falling back to heuristic.")
        return None

