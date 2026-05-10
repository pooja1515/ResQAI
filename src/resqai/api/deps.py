from __future__ import annotations

from collections.abc import Generator

from resqai.configs.settings import Settings, get_settings


def settings_dep() -> Settings:
    return get_settings()


def request_id_dep() -> Generator[str, None, None]:
    # Placeholder dependency hook (e.g., from header / middleware).
    yield "request-id-placeholder"

