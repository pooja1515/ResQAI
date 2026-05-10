from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger("resqai.voice_intelligence")


ALLOWED_DISTRESS = {"low", "medium", "high", "critical"}
ALLOWED_URGENCY = {"low", "medium", "high"}


@dataclass(frozen=True)
class SemanticResult:
    distress_level: str
    urgency: str
    needs: list[str]
    risk_factors: list[str]
    priority_groups: list[str]
    recommended_actions: list[str]
    reasoning_summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "distress_level": self.distress_level,
            "urgency": self.urgency,
            "needs": self.needs,
            "risk_factors": self.risk_factors,
            "priority_groups": self.priority_groups,
            "recommended_actions": self.recommended_actions,
            "reasoning_summary": self.reasoning_summary,
        }


def _find_json_object(text: str) -> str:
    """Extract the first top-level JSON object from a string."""
    s = text.strip()
    start = s.find("{")
    if start == -1:
        raise ValueError("Model output did not contain a JSON object.")

    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    raise ValueError("Model output contained an unterminated JSON object.")


def _as_list_of_str(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return [x.strip() for x in value if x.strip()]
    raise ValueError("Expected a list of strings.")


def parse_semantic_json(raw_output: str) -> SemanticResult:
    """Parse and validate a model-generated JSON response.

    Recovery:
    - If raw_output is already JSON -> parse directly
    - Else extract the first JSON object and parse that
    """
    s = (raw_output or "").strip()
    if not s:
        raise ValueError("Model returned empty output.")

    data: dict
    try:
        parsed = json.loads(s)
        if not isinstance(parsed, dict):
            raise ValueError("Model JSON must be an object.")
        data = parsed
    except json.JSONDecodeError:
        obj_str = _find_json_object(s)
        try:
            parsed = json.loads(obj_str)
            if not isinstance(parsed, dict):
                raise ValueError("Model JSON must be an object.")
            data = parsed
        except json.JSONDecodeError as exc:
            logger.exception("Failed to decode model JSON: %s", obj_str[:500])
            raise ValueError("Model returned invalid JSON.") from exc

    distress_level = str(data.get("distress_level", "")).strip().lower()
    urgency = str(data.get("urgency", "")).strip().lower()
    needs = _as_list_of_str(data.get("needs", []))
    risk_factors = _as_list_of_str(data.get("risk_factors", []))
    priority_groups = _as_list_of_str(data.get("priority_groups", []))
    recommended_actions = _as_list_of_str(data.get("recommended_actions", []))
    reasoning_summary = str(data.get("reasoning_summary", "")).strip()

    if distress_level not in ALLOWED_DISTRESS:
        raise ValueError(f"Invalid distress_level: {distress_level!r}")
    if urgency not in ALLOWED_URGENCY:
        raise ValueError(f"Invalid urgency: {urgency!r}")
    if not reasoning_summary:
        raise ValueError("Invalid reasoning_summary; expected a non-empty string.")

    # Enforce strict schema: reject unknown keys to keep downstream stable.
    allowed_keys = {
        "distress_level",
        "urgency",
        "needs",
        "risk_factors",
        "priority_groups",
        "recommended_actions",
        "reasoning_summary",
    }
    extra_keys = set(data.keys()) - allowed_keys
    if extra_keys:
        raise ValueError(f"Unexpected keys in model output: {sorted(extra_keys)}")

    return SemanticResult(
        distress_level=distress_level,
        urgency=urgency,
        needs=needs,
        risk_factors=risk_factors,
        priority_groups=priority_groups,
        recommended_actions=recommended_actions,
        reasoning_summary=reasoning_summary,
    )
