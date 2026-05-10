from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RESQAI_",
        case_sensitive=False,
        extra="ignore",
    )

    env: str = Field(default="local", description="Environment name (local/staging/prod).")
    log_level: str = Field(default="INFO", description="Logging level.")

    api_host: str = Field(default="0.0.0.0", description="API bind host.")
    api_port: int = Field(default=8000, description="API bind port.")

    device: str = Field(default="cpu", description="cpu/cuda, etc.")
    model_cache_dir: str = Field(default="./artifacts/model_cache")


@lru_cache
def get_settings() -> Settings:
    return Settings()

