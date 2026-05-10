from __future__ import annotations


JSON_SCHEMA = """Return ONLY valid JSON (no markdown) matching EXACTLY:
{
  "location": "...",
  "weather_severity": "low|moderate|high|severe",
  "flood_risk": "low|moderate|high|critical",
  "expected_escalation": true,
  "environmental_risks": ["..."],
  "recommended_actions": ["..."],
  "reasoning_summary": "..."
}

Rules:
- Output must be a single JSON object with ONLY these keys.
- Keep output compact (<= ~1200 characters).
- Use at most 4 items per array; each item <= 12 words.
- Keep `reasoning_summary` under 2 sentences.
- Base your answer ONLY on the provided weather observations + forecast summary.
- Do NOT use rule-based flood logic; reason contextually from conditions.
"""


SYSTEM = """You are ResQAI's weather intelligence analyst for disaster response.
Given live weather observations and a short forecast summary, infer flood escalation risk, severe weather danger,
evacuation urgency, environmental hazards, and practical response actions.
Be conservative and operational: recommend actions that reduce risk quickly.
"""


def build_weather_prompt(*, location: str, weather_payload: dict) -> str:
    return (
        f"{SYSTEM}\n\n"
        "CRITICAL: Reply with ONLY the JSON object, no extra text.\n\n"
        f"Location: {location}\n"
        f"Weather data (JSON): {weather_payload}\n\n"
        f"{JSON_SCHEMA}\n"
    )


def build_repair_prompt(*, location: str, weather_payload: dict, bad_output: str) -> str:
    return (
        f"{SYSTEM}\n\n"
        "You previously returned an invalid or non-JSON response.\n"
        "Rewrite it as STRICT JSON matching the schema. Output ONLY JSON.\n\n"
        f"Location: {location}\n"
        f"Weather data (JSON): {weather_payload}\n\n"
        f"Invalid response:\n{bad_output}\n\n"
        f"{JSON_SCHEMA}\n"
    )

