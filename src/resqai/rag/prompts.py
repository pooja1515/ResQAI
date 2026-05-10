from __future__ import annotations


RAG_JSON_SCHEMA = """Return ONLY valid JSON (no markdown) matching EXACTLY:
{
  "risk_level": "low|medium|high|critical",
  "recommended_actions": ["..."],
  "safety_guidelines": ["..."],
  "reasoning_summary": "..."
}

Rules:
- Output must be a single JSON object with ONLY these keys.
- Keep it SHORT and compact; do not exceed ~1200 characters.
- Use at most 3 items per array; each item <= 12 words.
- Keep `reasoning_summary` under 2 sentences.
- Be operational and concise (what to do now, what to avoid).
- Ground your answer ONLY in the retrieved context. If insufficient, say so in reasoning_summary and keep lists conservative.
"""


SYSTEM_INSTRUCTIONS = """You are ResQAI's grounded disaster intelligence assistant.
You must answer user questions using ONLY the provided retrieved context from an emergency knowledge base.
If information is missing, do not invent facts; instead provide general safe guidance only if supported by the context.

Multilingual:
- The query may be English, Hindi, French, or Spanish.
- Prefer responding in the same language as the query when possible, while still returning strict JSON.

Operational tone:
- Write like an emergency-response brief: clear, actionable, conservative.
"""


def build_rag_prompt(query: str, context_snippets: list[str]) -> str:
    context_block = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context_snippets)]) or "(no context)"
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        "CRITICAL: Reply with ONLY the JSON object, with no extra text.\n\n"
        f"User query:\n{query.strip()}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"{RAG_JSON_SCHEMA}\n"
    )


def build_stricter_rag_prompt(query: str, context_snippets: list[str]) -> str:
    context_block = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context_snippets)]) or "(no context)"
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        "CRITICAL:\n"
        "- Output ONLY the JSON object.\n"
        "- No markdown.\n"
        "- No extra keys.\n"
        "- Keep output under 1200 characters.\n"
        "- Ensure JSON is complete and closed (no truncation).\n\n"
        f"User query:\n{query.strip()}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"{RAG_JSON_SCHEMA}\n"
    )


def build_repair_prompt(bad_output: str, query: str, context_snippets: list[str]) -> str:
    context_block = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context_snippets)]) or "(no context)"
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        "You previously returned an invalid response.\n"
        "Rewrite it as STRICT JSON matching the schema. Output ONLY JSON.\n\n"
        f"User query:\n{query.strip()}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"Invalid response:\n{bad_output}\n\n"
        f"{RAG_JSON_SCHEMA}\n"
    )
