"""Finding a bitmap to show for an extracted icon.

An icon exported as a bitmap already has one. An SVG-only icon does not, so a
preview is rendered once and cached under a scratch directory the caller owns.
The renderer is a parameter so this can be tested without Inkscape.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from services.svg_ops import export_svg_to_png

Renderer = Callable[[Path, Path], object]


def preview_cache_name(stem: str) -> str:
    """The cache filename for an icon's rendered preview."""
    return f"{stem}_preview.png"


def resolve_preview_path(
    icon,
    cache_dir: Path,
    render: Renderer = export_svg_to_png,
) -> Path | None:
    """Return a bitmap to display for `icon`, rendering one if needed.

    None means there is nothing to show: no bitmap export, no SVG to render
    from, or a render that failed. A failed render is not fatal — the preview
    pane is a convenience, and the export itself already succeeded.
    """
    existing = getattr(icon, "preview_path", None)
    if existing is not None and existing.exists():
        return existing

    svg_path = icon.outputs.get("svg")
    if svg_path is None or not svg_path.exists():
        return None

    preview_path = cache_dir / preview_cache_name(icon.stem)
    if preview_path.exists():
        return preview_path

    try:
        render(svg_path, preview_path)
    except Exception:
        return None

    return preview_path if preview_path.exists() else None
