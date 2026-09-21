"""Human-readable descriptions of settings, results and icons.

Text an interface displays, kept out of the widget classes so it can be
tested without building a window — and so a native front end gets the same
wording as the tkinter one.

This is display text about a *pending or finished* run. It is deliberately
not the `provider_summary` recorded on an `ExtractionResult`, which is part of
the exported record and must not drift to follow the wording here.
"""
from __future__ import annotations

from typing import Sequence

from pathlib import Path

from services.settings_schema import AppSettings
from services.sheets import find_sheets

STATUS_SEPARATOR = "  |  "


def format_size(size: Sequence[int]) -> str:
    """A width/height pair as it appears in the results table."""
    return f"{size[0]} x {size[1]}"


def describe_input(path: Path) -> str:
    """What the app sees at the chosen input, or "" when there is nothing to add.

    Only a folder needs explaining. Choosing the parent of the artwork instead
    of the artwork is the easy mistake, and it costs a whole run to discover,
    so the count is shown before the run rather than after it.
    """
    if not path.exists() or not path.is_dir():
        return ""
    count = len(find_sheets(path))
    if count == 0:
        return "Folder selected, but there are no .png or .svg sheets directly inside it."
    sheets = "sheet" if count == 1 else "sheets"
    return f"Folder selected: {count} {sheets}, each extracted into its own subfolder."


def provider_summary(settings: AppSettings) -> str:
    """One line naming the backend a run would use, for a header label."""
    if settings.provider == "ollama":
        return f"Active provider: Ollama local ({settings.ollama_model})"
    if settings.provider == "llamacpp":
        model = settings.llamacpp_model or "loaded model"
        return f"Active provider: llama.cpp local ({model})"
    if settings.provider == "directory":
        model = settings.local_model_name or "directory catalog"
        return f"Active provider: Local directory ({model})"
    return "Active provider: Geometry-only local extraction"


def describe_icon(icon) -> str:
    """The caption above the preview image."""
    return STATUS_SEPARATOR.join(
        [
            icon.stem,
            f"source {format_size(icon.source_size)}",
            f"canvas {format_size(icon.canvas_size)}",
        ]
    )


def describe_result(result) -> str:
    """The status line for a finished run.

    Mentions replaced files only when a previous export was actually
    overwritten, and naming only when naming was asked for, so a plain
    geometry run into a fresh folder reads as one clean sentence.
    """
    parts = [f"Extracted {len(result.icons)} icons to {result.output_dir}"]

    commit = getattr(result, "commit", None)
    if commit is not None and commit.replaced:
        parts.append(f"replaced {commit.replaced} files from the previous run")

    if result.naming.requested:
        parts.append(result.naming.describe())

    return STATUS_SEPARATOR.join(parts)


def describe_progress_completion(icon_count: int) -> str:
    """The progress label once a run has finished."""
    return f"Done. Exported {icon_count} icons."
