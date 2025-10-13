"""Configuration settings for the inventory service."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Runtime settings sourced from environment variables."""

    api_host: str = Field("0.0.0.0", env="INVENTORY_SERVICE_HOST")
    api_port: int = Field(8001, env="INVENTORY_SERVICE_PORT")
    database_url: str = Field("postgresql://user:pass@localhost:5432/erp_inventory", env="INVENTORY_DATABASE_URL")
    enable_health_route: bool = Field(True, env="INVENTORY_ENABLE_HEALTH_ROUTE")
    log_level: str = Field("INFO", env="INVENTORY_LOG_LEVEL")

    class Config:
        env_file = (Path(__file__).resolve().parents[2] / ".env")
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


settings = get_settings()
