from __future__ import annotations

import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import numpy as np
import torch

import logging

logger = logging.getLogger("resqai.flood_severity")


def get_repo_root() -> "Path":
    # This file lives at: <repo>/src/resqai/pipelines/flood_severity/utils.py
    # parents: flood_severity -> pipelines -> resqai -> src -> <repo>
    from pathlib import Path

    return Path(__file__).resolve().parents[4]


def resolve_from_repo(path: "Path") -> "Path":
    from pathlib import Path

    if path.is_absolute():
        return path
    return (get_repo_root() / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def autocast_if_available(device: torch.device, enabled: bool = True):
    # AMP is only reliable on CUDA in most setups; keep it conservative.
    if enabled and device.type == "cuda":
        with torch.cuda.amp.autocast():
            yield
    else:
        with nullcontext():
            yield


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())


def save_checkpoint(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    # PyTorch 2.6 changed `torch.load` default to `weights_only=True`,
    # which breaks loading full (trusted) checkpoint dicts like ours.
    # We explicitly request full loading while keeping backward-compat.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions don't support `weights_only`.
        return torch.load(path, map_location=device)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load checkpoint: %s", path)
        raise RuntimeError(
            f"Failed to load checkpoint at '{path}'. "
            "If this file came from an untrusted source, do NOT set weights_only=False. "
            "Re-train to generate a fresh checkpoint."
        ) from exc
