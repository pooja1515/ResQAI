from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger("resqai.weather")


def dumps_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def find_json_object(text: str) -> str:
    s = (text or "").strip()
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


def cleanup_json_text(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return s[first : last + 1].strip()
    return s


def as_list_of_str(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return [x.strip() for x in value if x.strip()]
    raise ValueError("Expected a list of strings.")


def count_sentences(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    seps = [".", "!", "?", "।"]
    count = 0
    for ch in t:
        if ch in seps:
            count += 1
    return max(1, count)


@dataclass(frozen=True)
class WeatherIntelligence:
    location: str
    weather_severity: str
    flood_risk: str
    expected_escalation: bool
    environmental_risks: list[str]
    recommended_actions: list[str]
    reasoning_summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "location": self.location,
            "weather_severity": self.weather_severity,
            "flood_risk": self.flood_risk,
            "expected_escalation": self.expected_escalation,
            "environmental_risks": self.environmental_risks,
            "recommended_actions": self.recommended_actions,
            "reasoning_summary": self.reasoning_summary,
        }

