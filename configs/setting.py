from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _dict_to_namespace(value: Any):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_dict_to_namespace(v) for v in value]
    return value


def _apply_env_overrides(data: dict) -> dict:
    """Một vài override tiện dụng khi demo/deploy."""
    if os.getenv("EMBEDDING_MODEL"):
        data.setdefault("embedding", {})["model_name"] = os.getenv("EMBEDDING_MODEL")
    if os.getenv("EMBEDDING_DEVICE"):
        data.setdefault("embedding", {})["device"] = os.getenv("EMBEDDING_DEVICE")
    if os.getenv("LLM_MODEL"):
        data.setdefault("llm", {})["model"] = os.getenv("LLM_MODEL")
    return data


def load_config(path: str | None = None):
    config_path = Path(path) if path else Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data = _apply_env_overrides(data)
    return _dict_to_namespace(data)


config = load_config()
