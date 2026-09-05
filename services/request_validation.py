"""Pre-flight checks on an extraction request, shared by both interfaces.

The CLI and the GUI each used to carry their own copy of these rules, worded
differently and — as it turned out — not actually the same set: the CLI
rejected an input whose suffix was neither `.png` nor `.svg`, and the GUI did
not, so a path typed into the GUI's entry box could reach the extractor and
fail later with a murkier error. One list of rules now serves both.

Returning the problem rather than raising lets each caller present it in its
own idiom: the CLI exits with the message, the GUI shows a dialog titled with
`title`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from services.settings_schema import AppSettings
from services.vision import is_local_url

SUPPORTED_SUFFIXES = (".png", ".svg")


@dataclass(frozen=True)
class ValidationIssue:
    """One reason a request cannot run, ready to show to a person."""

    title: str
    message: str


def validate_extraction_request(
    settings: AppSettings,
    input_path: Path,
    formats: Iterable[str],
) -> ValidationIssue | None:
    """Return the first problem with this request, or None if it can run.

    Ordered cheapest and most fundamental first, so the message a user sees
    names the thing they most likely need to fix.
    """
    selected = set(formats)

    if not input_path.exists():
        return ValidationIssue("Missing Input", f"Input file not found: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return ValidationIssue("Unsupported Input", "Supported inputs are .png and .svg")

    if not selected:
        return ValidationIssue(
            "No Export Format", "At least one export format must be selected."
        )

    if settings.output_width <= 0 or settings.output_height <= 0:
        return ValidationIssue(
            "Canvas Size", "Canvas width and height must be positive integers."
        )

    if settings.provider == "ollama" and not is_local_url(settings.ollama_url):
        return ValidationIssue(
            "Local Only",
            "Ollama provider requires a local endpoint such as http://127.0.0.1:11434.",
        )

    if settings.provider == "llamacpp" and not is_local_url(settings.llamacpp_url):
        return ValidationIssue(
            "Local Only",
            "llama.cpp provider requires a local endpoint such as http://127.0.0.1:8080.",
        )

    return None
