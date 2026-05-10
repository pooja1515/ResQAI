from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("resqai.memory")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dumps_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def repo_root() -> Path:
    # This file lives at: <repo>/src/resqai/memory/memory_utils.py
    return Path(__file__).resolve().parents[3]


def resolve_from_repo(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def safe_time_hhmm(ts: str) -> str:
    """Format an ISO timestamp to HH:MM; returns original on parse failure."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        # Display in local timezone if tz-aware; else keep as-is.
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%H:%M")
    except Exception:
        return ts


def summarize_event(event_type: str, payload: dict[str, Any]) -> str:
    et = (event_type or "").lower()
    if et == "vision":
        cls = payload.get("predicted_class") or payload.get("predicted") or ""
        conf = payload.get("confidence")
        if cls:
            return f"Flood classification: {cls}" + (f" ({conf:.2f})" if isinstance(conf, (int, float)) else "")
        return "Vision update recorded"
    if et == "weather":
        summary = payload.get("reasoning_summary") or payload.get("status") or ""
        return str(summary)[:120] if summary else "Weather update recorded"
    if et == "voice":
        tx = payload.get("transcription") or ""
        tx = str(tx).strip()
        if tx:
            return (tx[:117] + "...") if len(tx) > 120 else tx
        return "Voice distress update recorded"
    if et == "rag":
        rs = payload.get("reasoning_summary") or ""
        return str(rs)[:120] if rs else "RAG guidance recorded"
    return "Event recorded"


def cleanup_json_text(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return s[first : last + 1].strip()
    return s


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
class MemoryInsight:
    crisis_trend: str
    severity_progression: str
    distress_trend: str
    recommended_priority: str
    reasoning_summary: str

    def to_dict(self) -> dict[str, str]:
        return {
            "crisis_trend": self.crisis_trend,
            "severity_progression": self.severity_progression,
            "distress_trend": self.distress_trend,
            "recommended_priority": self.recommended_priority,
            "reasoning_summary": self.reasoning_summary,
        }


def parse_memory_insight(raw: str) -> MemoryInsight:
    s = cleanup_json_text(raw)
    if not s:
        raise ValueError("Empty model output.")
    try:
        data = json.loads(s)
        if not isinstance(data, dict):
            raise ValueError("Model output must be a JSON object.")
    except json.JSONDecodeError:
        data = json.loads(find_json_object(s))

    allowed = {
        "crisis_trend",
        "severity_progression",
        "distress_trend",
        "recommended_priority",
        "reasoning_summary",
    }
    extra = set(data.keys()) - allowed
    if extra:
        raise ValueError(f"Unexpected keys: {sorted(extra)}")

    ct = str(data.get("crisis_trend", "")).strip()
    sp = str(data.get("severity_progression", "")).strip()
    dt = str(data.get("distress_trend", "")).strip()
    rp = str(data.get("recommended_priority", "")).strip()
    rs = str(data.get("reasoning_summary", "")).strip()

    if not (ct and sp and dt and rp and rs):
        raise ValueError("All fields must be non-empty strings.")
    if count_sentences(rs) > 2:
        raise ValueError("reasoning_summary must be under 2 sentences.")

    return MemoryInsight(
        crisis_trend=ct,
        severity_progression=sp,
        distress_trend=dt,
        recommended_priority=rp,
        reasoning_summary=rs,
    )
