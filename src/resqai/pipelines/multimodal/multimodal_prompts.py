from __future__ import annotations

FUSION_SCHEMA_HINT = """Return ONLY valid JSON (no markdown) matching EXACTLY:
{
  "overall_risk": "...",
  "rescue_priority": "...",
  "recommended_actions": ["..."],
  "environmental_risks": ["..."],
  "vulnerable_groups": ["..."],
  "operational_notes": ["..."],
  "reasoning_summary": "..."
}

Rules:
- Output must be a single JSON object with ONLY these keys.
- Use short, concrete bullet-like strings inside arrays.
- Keep `reasoning_summary` concise (1–3 sentences).
- If uncertain, be conservative and use empty arrays.
"""


SYSTEM_INSTRUCTIONS = """You are ResQAI's multimodal crisis fusion agent.
You fuse structured signals from vision and voice pipelines into a single emergency intelligence report.

Priorities:
- Determine overall disaster severity (overall risk) and rescue priority using both vision and voice signals.
- Provide actionable response recommendations and response coordination strategy.
- Identify environmental risks and vulnerable groups.
- Provide operational notes (communications, access, safety, staging, resource needs).

Be multilingual-aware: the transcript may be English, Hindi, French, or Spanish.
Do not invent details not supported by inputs.
"""


def build_fusion_prompt(vision: dict, voice: dict) -> str:
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        "CRITICAL: Reply with ONLY the JSON object, with no extra text.\n\n"
        "Input expectations:\n"
        "- vision contains: predicted_class, confidence\n"
        "- voice contains: transcription, language, distress_level, urgency, needs, risk_factors, recommended_actions\n\n"
        "Inputs (JSON):\n"
        f"vision = {vision}\n"
        f"voice = {voice}\n\n"
        f"{FUSION_SCHEMA_HINT}\n"
    )


def build_repair_prompt(bad_output: str, vision: dict, voice: dict) -> str:
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        "You previously returned an invalid response.\n"
        "Rewrite it as STRICT JSON matching the schema. Output ONLY JSON.\n\n"
        "Inputs (JSON):\n"
        f"vision = {vision}\n"
        f"voice = {voice}\n\n"
        f"Invalid response:\n{bad_output}\n\n"
        f"{FUSION_SCHEMA_HINT}\n"
    )
