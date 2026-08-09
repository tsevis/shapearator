"""Semantic (model-driven) icon naming, and the gate that guards it.

The rule this module exists to enforce: a run never claims a model named an
icon unless that model actually did. Backend readiness is checked once, before
any pixel is written, and each icon carries its own naming outcome afterwards.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Callable, NamedTuple

from .extraction_types import (
    NAMING_FAILED,
    NAMING_NAMED,
    NAMING_NOT_REQUESTED,
    ExtractedIcon,
    ExtractionProgress,
    NamingSummary,
)
from .vision import (
    PreflightResult,
    active_vision_model,
    build_vision_client,
    preflight,
    semantic_naming_enabled,
)

ProgressCallback = Callable[[ExtractionProgress], None] | None


class SemanticPreflightError(RuntimeError):
    """The configured vision backend is not usable, so the run cannot name icons."""

    def __init__(self, result: PreflightResult):
        super().__init__(result.message)
        self.result = result


def slugify(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", normalized)


def naming_requested(settings) -> bool:
    """True when the settings ask for semantic naming and name a model to use."""
    return bool(semantic_naming_enabled(settings) and active_vision_model(settings))


class BackendReadiness(NamedTuple):
    proceed: bool
    warnings: tuple[str, ...]
    result: PreflightResult | None  # None when naming was never requested


def check_backend_ready(settings, allow_unnamed: bool) -> BackendReadiness:
    """Decide whether naming may proceed, before the run writes anything.

    A failed check raises unless the caller opted into ``allow_unnamed``, in
    which case the run continues as a plain geometry export and says so.
    """
    if not naming_requested(settings):
        return BackendReadiness(False, (), None)

    result = preflight(settings)
    if result.ok:
        warnings = () if result.vision_capable is not False else (result.message,)
        return BackendReadiness(True, warnings, result)
    if not allow_unnamed:
        raise SemanticPreflightError(result)
    return BackendReadiness(False, (f"Semantic naming skipped: {result.message}",), result)


def apply_semantic_names(
    settings,
    icons: list[ExtractedIcon],
    progress_callback: ProgressCallback = None,
) -> tuple[list[ExtractedIcon], NamingSummary, tuple[str, ...]]:
    """Label and rename each icon, returning new icons plus what happened.

    A per-icon failure is not fatal: that icon keeps its generic name and
    records why, so a flaky model call costs one filename rather than the run.
    """
    client = build_vision_client(settings)
    model = active_vision_model(settings)
    used_names: dict[str, int] = {}
    renamed: list[ExtractedIcon] = []
    errors: list[str] = []

    for position, icon in enumerate(icons, start=1):
        _emit(progress_callback, position, len(icons))
        semantic, error = _label_icon(client, model, icon)
        if error is not None:
            errors.append(error)
            renamed.append(replace(icon, naming_status=NAMING_FAILED, naming_error=error))
            continue
        renamed.append(_rename_icon(icon, semantic, used_names))

    named = sum(1 for icon in renamed if icon.naming_status == NAMING_NAMED)
    failed = len(renamed) - named
    summary = NamingSummary(
        requested=True,
        provider=settings.provider,
        model=model,
        named=named,
        failed=failed,
        errors=tuple(dict.fromkeys(errors)),
    )
    warnings = ()
    if failed:
        warnings = (
            f"{failed} of {len(renamed)} icons could not be named and kept generic filenames "
            f"({'; '.join(summary.errors)}).",
        )
    return renamed, summary, warnings


def not_requested_summary() -> NamingSummary:
    return NamingSummary()


def mark_all_unnamed(icons: list[ExtractedIcon]) -> list[ExtractedIcon]:
    return [replace(icon, naming_status=NAMING_NOT_REQUESTED) for icon in icons]


# --- internals ------------------------------------------------------------

def _emit(callback: ProgressCallback, current: int, total: int) -> None:
    if callback is not None:
        callback(ExtractionProgress("naming", current, total, f"Naming icon {current} of {total}"))


def _label_icon(client, model: str, icon: ExtractedIcon) -> tuple[dict, str | None]:
    if icon.preview_path is None or not icon.preview_path.exists():
        return {}, "no bitmap preview was available to label"
    try:
        return client.identify_icon(model, icon.preview_path), None
    except Exception as exc:  # any backend failure is per-icon, never fatal
        return {}, str(exc) or exc.__class__.__name__


def _rename_icon(icon: ExtractedIcon, semantic: dict, used_names: dict[str, int]) -> ExtractedIcon:
    stem = slugify(str(semantic.get("label", ""))) or f"icon-{icon.index:03d}"
    if stem in used_names:
        used_names[stem] += 1
        stem = f"{stem}-{used_names[stem]:02d}"
    else:
        used_names[stem] = 1

    try:
        outputs = _rename_outputs(icon.outputs, stem)
    except OSError as exc:
        return replace(icon, naming_status=NAMING_FAILED, naming_error=f"could not rename files: {exc}")

    tags = semantic.get("tags")
    return replace(
        icon,
        stem=stem,
        outputs=outputs,
        preview_path=outputs.get("png") or outputs.get("jpg") or outputs.get("tiff") or icon.preview_path,
        semantic_label=stem,
        semantic_tags=list(tags) if isinstance(tags, list) else [],
        semantic_confidence=semantic.get("confidence"),
        naming_status=NAMING_NAMED,
        naming_error=None,
    )


def _rename_outputs(outputs: dict[str, Path], stem: str) -> dict[str, Path]:
    """Move every export to ``stem``, rolling back if one of them fails.

    Renaming is not atomic across formats, so a partial failure would leave
    the icon's files split across two names with no record of either.
    """
    done: list[tuple[Path, Path]] = []
    renamed: dict[str, Path] = {}
    try:
        for fmt, old_path in outputs.items():
            new_path = old_path.with_name(f"{stem}{old_path.suffix.lower()}")
            if new_path != old_path:
                old_path.rename(new_path)
                done.append((old_path, new_path))
            renamed[fmt] = new_path
    except OSError:
        for old_path, new_path in reversed(done):
            new_path.rename(old_path)
        raise
    return renamed
