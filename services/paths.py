"""Canonical filesystem locations for the app, resolved relative to the repo.

Keeping these in one place means "clone from GitHub and run" works without any
absolute paths baked into the code.
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent  # services/
REPO_ROOT = PACKAGE_DIR.parent


def repo_root() -> Path:
    return REPO_ROOT


def config_dir() -> Path:
    return REPO_ROOT / "config"


def default_models_root() -> Path:
    """Where downloaded llama.cpp GGUF weights live by default."""
    return REPO_ROOT / "models"


def setup_state_path() -> Path:
    """Marker file recording that first-run setup has been completed."""
    return config_dir() / "setup_state.json"


def resolve_models_root(models_root: str | None) -> Path:
    value = (models_root or "").strip()
    return Path(value).expanduser() if value else default_models_root()
