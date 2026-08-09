"""Settings schema: the dataclass, its allowed values, and safe coercion.

This module is the single source of truth for what a valid Shapearator setting
looks like. The CLI imports the choice tuples for ``argparse``; ``ConfigStore``
imports :func:`coerce_settings` so a hand-edited or newer-version config file
degrades to defaults per field instead of crashing startup.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Callable, Mapping, NamedTuple


APPEARANCES = ("light", "dark")
PROVIDERS = ("geometry", "ollama", "llamacpp", "directory")
CANVAS_MODES = ("original", "uniform_to_largest", "individual_fit")
BITMAP_EXPORT_MODES = ("keep_background", "transparent_preserve_interior")
FORMATS = ("png", "jpg", "tiff", "svg")


@dataclass
class AppSettings:
    appearance: str = "light"
    provider: str = "geometry"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5vl:3b"
    llamacpp_url: str = "http://127.0.0.1:8080"
    llamacpp_model: str = ""
    models_root: str = ""  # where downloaded llama.cpp GGUF weights live; blank = <repo>/models
    local_model_root: str = ""  # optional directory-provider catalog root
    local_model_name: str = ""
    semantic_naming: bool = False
    default_formats: list[str] = field(default_factory=lambda: ["png", "svg"])
    output_width: int = 512
    output_height: int = 512
    canvas_mode: str = "uniform_to_largest"
    bitmap_export_mode: str = "transparent_preserve_interior"
    padding: int = 12
    min_area: int = 200
    merge_gap: int = 13
    last_input_path: str = ""
    last_output_dir: str = ""


CHOICES: dict[str, tuple[str, ...]] = {
    "appearance": APPEARANCES,
    "provider": PROVIDERS,
    "canvas_mode": CANVAS_MODES,
    "bitmap_export_mode": BITMAP_EXPORT_MODES,
}


class CoercionResult(NamedTuple):
    """A validated settings object plus the human-readable problems found."""

    settings: AppSettings
    warnings: list[str]


class _Invalid(Exception):
    """Raised by a field coercer when the value cannot be salvaged."""


# --- field coercers -------------------------------------------------------
# Each returns a clean value or raises _Invalid; the caller substitutes the
# dataclass default and records a warning.

def _coerce_enum(value: Any, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise _Invalid(f"expected one of {', '.join(allowed)}")
    return value


def _coerce_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise _Invalid("expected true or false")
    return value


def _coerce_int(value: Any, minimum: int) -> int:
    # bool is a subclass of int; a JSON `true` for a pixel count is a mistake.
    if isinstance(value, bool):
        raise _Invalid("expected a number")
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            raise _Invalid("expected a number") from None
    if not isinstance(value, int):
        raise _Invalid("expected a number")
    if value < minimum:
        raise _Invalid(f"expected a number >= {minimum}")
    return value


def _coerce_text(value: Any) -> str:
    if not isinstance(value, str):
        raise _Invalid("expected text")
    return value


def _coerce_nonblank_text(value: Any) -> str:
    text = _coerce_text(value).strip()
    if not text:
        raise _Invalid("expected a non-empty value")
    return text


def _coerce_formats(value: Any, warnings: list[str]) -> list[str]:
    if not isinstance(value, list):
        raise _Invalid(f"expected a list of {', '.join(FORMATS)}")
    kept: list[str] = []
    dropped: list[str] = []
    for entry in value:
        if isinstance(entry, str) and entry in FORMATS:
            if entry not in kept:
                kept.append(entry)
        else:
            dropped.append(repr(entry))
    if dropped:
        warnings.append(f"default_formats: ignored unsupported {', '.join(dropped)}.")
    if not kept:
        raise _Invalid(f"expected at least one of {', '.join(FORMATS)}")
    return kept


_TEXT_FIELDS = frozenset(
    {
        "ollama_model",
        "llamacpp_model",
        "models_root",
        "local_model_root",
        "local_model_name",
        "last_input_path",
        "last_output_dir",
    }
)

_URL_FIELDS = frozenset({"ollama_url", "llamacpp_url"})

_INT_MINIMUMS = {
    "output_width": 1,
    "output_height": 1,
    "padding": 0,
    "min_area": 1,
    "merge_gap": 1,
}


def _coercer_for(name: str, warnings: list[str]) -> Callable[[Any], Any]:
    if name in CHOICES:
        allowed = CHOICES[name]
        return lambda value: _coerce_enum(value, allowed)
    if name in _INT_MINIMUMS:
        minimum = _INT_MINIMUMS[name]
        return lambda value: _coerce_int(value, minimum)
    if name in _URL_FIELDS:
        return _coerce_nonblank_text
    if name in _TEXT_FIELDS:
        return _coerce_text
    if name == "semantic_naming":
        return _coerce_bool
    if name == "default_formats":
        return lambda value: _coerce_formats(value, warnings)
    raise KeyError(f"No coercer registered for settings field {name!r}")


def coerce_settings(data: Any) -> CoercionResult:
    """Build an :class:`AppSettings` from arbitrary decoded JSON.

    Unknown keys are dropped and invalid values fall back to the dataclass
    default, one field at a time, so a single bad entry never costs the user
    their other saved preferences. The input mapping is never mutated.
    """
    defaults = AppSettings()
    if not isinstance(data, Mapping):
        return CoercionResult(defaults, [f"Settings must be a JSON object; got {type(data).__name__}."])

    known = {spec.name for spec in fields(AppSettings)}
    warnings: list[str] = []
    accepted: dict[str, Any] = {}

    for key, value in data.items():
        if key not in known:
            warnings.append(f"Ignored unknown setting {key!r}.")
            continue
        try:
            accepted[key] = _coercer_for(key, warnings)(value)
        except _Invalid as exc:
            warnings.append(f"{key}: {exc}. Using default {getattr(defaults, key)!r}.")

    return CoercionResult(replace(defaults, **accepted), warnings)


def settings_to_dict(settings: AppSettings) -> dict[str, Any]:
    """Return a JSON-serializable copy of ``settings``."""
    return asdict(settings)
