from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger("resqai.multimodal_fusion")


@dataclass(frozen=True)
class FusionResult:
    overall_risk: str
    rescue_priority: str
    recommended_actions: list[str]
    environmental_risks: list[str]
    vulnerable_groups: list[str]
    operational_notes: list[str]
    reasoning_summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_risk": self.overall_risk,
            "rescue_priority": self.rescue_priority,
            "recommended_actions": self.recommended_actions,
            "environmental_risks": self.environmental_risks,
            "vulnerable_groups": self.vulnerable_groups,
            "operational_notes": self.operational_notes,
            "reasoning_summary": self.reasoning_summary,
        }


def load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def validate_inputs(vision: dict, voice: dict) -> None:
    """Validate expected input shapes to keep fusion predictable."""
    if not isinstance(vision, dict):
        raise ValueError("vision JSON must be an object.")
    if not isinstance(voice, dict):
        raise ValueError("voice JSON must be an object.")

    for key in ("predicted_class", "confidence"):
        if key not in vision:
            raise ValueError(f"vision missing required key: {key}")
    if not isinstance(vision["predicted_class"], str) or not vision["predicted_class"].strip():
        raise ValueError("vision.predicted_class must be a non-empty string.")
    if not isinstance(vision["confidence"], (int, float)):
        raise ValueError("vision.confidence must be a number.")

    required_voice = (
        "transcription",
        "language",
        "distress_level",
        "urgency",
        "needs",
        "risk_factors",
        "recommended_actions",
    )
    for key in required_voice:
        if key not in voice:
            raise ValueError(f"voice missing required key: {key}")

    if not isinstance(voice["transcription"], str):
        raise ValueError("voice.transcription must be a string.")
    if not isinstance(voice["language"], (str, type(None))):
        raise ValueError("voice.language must be a string or null.")
    if not isinstance(voice["distress_level"], str) or not voice["distress_level"].strip():
        raise ValueError("voice.distress_level must be a non-empty string.")
    if not isinstance(voice["urgency"], str) or not voice["urgency"].strip():
        raise ValueError("voice.urgency must be a non-empty string.")
    for list_key in ("needs", "risk_factors", "recommended_actions"):
        val = voice.get(list_key)
        if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
            raise ValueError(f"voice.{list_key} must be a list of strings.")


def _find_json_object(text: str) -> str:
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


def parse_fusion_json(raw_output: str) -> FusionResult:
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
        parsed = json.loads(obj_str)
        if not isinstance(parsed, dict):
            raise ValueError("Model JSON must be an object.")
        data = parsed

    allowed_keys = {
        "overall_risk",
        "rescue_priority",
        "recommended_actions",
        "environmental_risks",
        "vulnerable_groups",
        "operational_notes",
        "reasoning_summary",
    }
    extra = set(data.keys()) - allowed_keys
    if extra:
        raise ValueError(f"Unexpected keys in model output: {sorted(extra)}")

    overall_risk = str(data.get("overall_risk", "")).strip()
    rescue_priority = str(data.get("rescue_priority", "")).strip()
    recommended_actions = _as_list_of_str(data.get("recommended_actions", []))
    environmental_risks = _as_list_of_str(data.get("environmental_risks", []))
    vulnerable_groups = _as_list_of_str(data.get("vulnerable_groups", []))
    operational_notes = _as_list_of_str(data.get("operational_notes", []))
    reasoning_summary = str(data.get("reasoning_summary", "")).strip()

    if not overall_risk:
        raise ValueError("overall_risk must be a non-empty string.")
    if not rescue_priority:
        raise ValueError("rescue_priority must be a non-empty string.")
    if not reasoning_summary:
        raise ValueError("reasoning_summary must be a non-empty string.")

    return FusionResult(
        overall_risk=overall_risk,
        rescue_priority=rescue_priority,
        recommended_actions=recommended_actions,
        environmental_risks=environmental_risks,
        vulnerable_groups=vulnerable_groups,
        operational_notes=operational_notes,
        reasoning_summary=reasoning_summary,
    )
