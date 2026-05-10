from __future__ import annotations


JSON_SCHEMA_HINT = """Return ONLY valid JSON (no markdown) matching EXACTLY:
{
  "distress_level": "low|medium|high|critical",
  "urgency": "low|medium|high",
  "needs": ["..."],
  "risk_factors": ["..."],
  "priority_groups": ["..."],
  "recommended_actions": ["..."],
  "reasoning_summary": "..."
}

Rules:
- Output must be a single JSON object.
- Keep outputs operational and concise (emergency-response intelligence tone).
- Use short, concrete bullet-like strings inside arrays (<= 6 items each).
- Keep `reasoning_summary` under 2 sentences.
- If unsure, be conservative (do not invent facts); use empty arrays when appropriate.
- Do not include additional keys.
"""


SYSTEM_INSTRUCTIONS = """You are ResQAI's emergency semantic reasoner.
Your job: interpret transcribed emergency speech and produce a structured emergency assessment.

Focus on:
- Distress level and urgency
- Emergency needs (food, water, medical, rescue, shelter, evacuation, etc.)
- Risk factors (flooding, trapped people, injuries, fire, structural collapse, etc.)
- Priority groups (elderly, children, pregnant people, injured, disabled, etc.)
- Recommended actions (immediate steps responders or callers should take)

Be multilingual: the transcript language may be English, Hindi, French, or Spanish.

CALIBRATION (very important):
- If the transcript indicates FLOOD WATER ENTERING A HOME/BUILDING, treat as at least HIGH distress.
- If people are TRAPPED / cannot evacuate / rising water, treat as CRITICAL distress and HIGH urgency.
- If the transcript describes immediate danger (injury, collapse, fire, drowning risk), treat as CRITICAL.

Multilingual flood cues examples (non-exhaustive):
- Hindi: "बाढ़", "पानी घर में घुस", "घर में पानी", "फँस", "फंसे", "फँसे हुए"
- French: "inondation", "l'eau entre", "piégé"
- Spanish: "inundación", "entra agua", "atrapado"

Do not downplay emergencies. When in doubt and conditions imply danger, choose the higher category.

ASR ROBUSTNESS:
- The transcript may contain Whisper errors, mixed languages, or missing punctuation.
- Use the provided signals (if any) and interpret intent conservatively for safety.
- If you detect flooding/trapped semantics even with noisy text, escalate appropriately.
"""


def build_user_prompt(transcription: str, language: str | None, signals: dict | None = None) -> str:
    lang = language or "unknown"
    signals_block = ""
    if isinstance(signals, dict) and signals:
        signals_block = f"Additional signals (JSON): {signals}\n"
    return (
        f"Language code: {lang}\n"
        f"{signals_block}"
        f"Transcript:\n{transcription.strip()}\n\n"
        f"{JSON_SCHEMA_HINT}"
    )


def build_ollama_prompt(transcription: str, language: str | None, signals: dict | None = None) -> str:
    # Ollama /api/generate is prompt-only; embed system + user content.
    user = build_user_prompt(transcription, language, signals=signals)
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        "CRITICAL: Reply with ONLY the JSON object, with no extra text.\n\n"
        f"{user}\n"
    )


def build_repair_prompt(bad_output: str) -> str:
    return (
        "You previously returned an invalid response.\n"
        "Rewrite it as STRICT JSON matching the schema. Output ONLY JSON.\n\n"
        f"Invalid response:\n{bad_output}\n\n"
        f"{JSON_SCHEMA_HINT}\n"
    )
