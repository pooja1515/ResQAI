from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger("resqai.voice_intelligence")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Audio path is not a file: {path}")


def dumps_json(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

