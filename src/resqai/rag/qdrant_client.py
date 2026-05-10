from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QdrantConfig:
    url: str
    api_key: str | None = None
    collection: str = "resqai_knowledge"


def get_qdrant_client(*_args: object, **_kwargs: object) -> None:
    """Placeholder for constructing a Qdrant client instance."""
    return None

