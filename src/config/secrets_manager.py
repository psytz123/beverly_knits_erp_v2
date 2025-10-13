"""Centralized secrets management utilities."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_secret_file() -> Dict[str, Any]:
    path = os.getenv("SECRETS_FILE")
    if not path:
        return {}

    secret_path = Path(path).expanduser().resolve()
    if not secret_path.exists():
        logger.warning("Secrets file specified but not found: %s", secret_path)
        return {}

    try:
        with secret_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to load secrets file %s: %s", secret_path, exc)
        return {}


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve a secret from environment variables or secrets file."""
    if value := os.getenv(name):
        return value

    secrets = _load_secret_file()
    if name in secrets:
        return secrets[name]

    return default


__all__ = ["get_secret"]
