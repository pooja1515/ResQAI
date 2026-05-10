from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger("resqai.rag")


@dataclass(frozen=True)
class RAGPaths:
    persist_dir: Path = Path("artifacts/rag/chroma")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dumps_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def repo_root() -> Path:
    # This file lives at <repo>/src/resqai/rag/utils.py
    return Path(__file__).resolve().parents[3]


def resolve_from_repo(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def find_json_object(text: str) -> str:
    """Extract the first top-level JSON object from a string."""
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


def as_list_of_str(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return [x.strip() for x in value if x.strip()]
    raise ValueError("Expected a list of strings.")


def cleanup_json_text(text: str) -> str:
    """Best-effort cleanup for model outputs intended to be a single JSON object.

    - Strips leading/trailing whitespace
    - If extra text surrounds JSON, keeps only from first '{' to last '}' (inclusive)
    """
    s = (text or "").strip()
    if not s:
        return s
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return s[first : last + 1].strip()
    return s


def count_sentences(text: str) -> int:
    # Simple heuristic; good enough for enforcing "under 2 sentences".
    t = (text or "").strip()
    if not t:
        return 0
    seps = [".", "!", "?", "।"]
    count = 0
    for ch in t:
        if ch in seps:
            count += 1
    return max(1, count)
