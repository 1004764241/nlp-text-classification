"""Project path and config loading helpers."""

from __future__ import annotations

from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Return repository root (parent of code/)."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> dict:
    """Load YAML config; relative paths are resolved from project root."""
    root = get_project_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
