"""Reading and writing the persisted settings file.

The schema itself lives in :mod:`services.settings_schema`; this module only
handles file I/O and re-exports :class:`AppSettings` for existing callers.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .settings_schema import AppSettings, coerce_settings, settings_to_dict

__all__ = ["AppSettings", "ConfigStore"]

logger = logging.getLogger(__name__)


class ConfigStore:
    def __init__(self, config_path: Path):
        self.config_path = config_path

    def load(self) -> AppSettings:
        """Return the stored settings, falling back to defaults on any problem."""
        return self.load_with_warnings()[0]

    def load_with_warnings(self) -> tuple[AppSettings, list[str]]:
        """Return the stored settings alongside any recovery warnings.

        A missing, unreadable, or malformed file yields defaults rather than
        raising, so a bad config can never block the GUI or CLI from starting.
        """
        if not self.config_path.exists():
            return AppSettings(), []
        try:
            raw = self.config_path.read_text(encoding="utf-8")
        except OSError as exc:
            warning = f"Could not read {self.config_path}: {exc}. Using default settings."
            logger.warning(warning)
            return AppSettings(), [warning]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            warning = f"{self.config_path} is not valid JSON ({exc}). Using default settings."
            logger.warning(warning)
            return AppSettings(), [warning]

        settings, warnings = coerce_settings(data)
        for warning in warnings:
            logger.warning("%s: %s", self.config_path, warning)
        return settings, warnings

    def save(self, settings: AppSettings) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(settings_to_dict(settings), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
