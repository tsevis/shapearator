"""Turning extracted icons into the layers of one Photoshop file.

:mod:`services.psd_writer` knows the file format and nothing about icons. This
module is the other half: it decides what a layer *is* and where it sits, and
the two questions are independent.

**Where** — ``psd_layout``. ``sheet`` rebuilds the artwork: the document is the
size of the source sheet and every shape keeps the position it had, so the file
opens looking like the original with one layer per shape. ``canvas`` places
each shape the way its single-file export places it, centred on the export
canvas, which stacks every shape in the middle of the document.

**What** — ``psd_layers``. Only ``bitmap`` is implemented here; the other two
add vector data on top of the same pixels and live in
:mod:`services.psd_vector`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from services.psd_writer import PsdLayer


@dataclass(frozen=True)
class Placement:
    """A trimmed layer and where its top-left corner belongs."""

    rgba: np.ndarray
    top: int
    left: int

    @property
    def is_empty(self) -> bool:
        return self.rgba.shape[0] == 0 or self.rgba.shape[1] == 0


def load_rgba(path: Path) -> np.ndarray:
    """A rendered icon as (h, w, 4) uint8."""
    with Image.open(path) as image:
        return np.array(image.convert("RGBA"))


def trim(rgba: np.ndarray, top: int = 0, left: int = 0) -> Placement:
    """Drop fully transparent margins, keeping track of what was dropped.

    A layer the size of the canvas per shape is how a PSD of four hundred
    shapes reaches a gigabyte; the margin carries no information, and its
    offset is recoverable, so it is not stored.
    """
    if rgba.size == 0:
        return Placement(rgba, top, left)
    filled = np.where(rgba[:, :, 3] > 0)
    if len(filled[0]) == 0:
        return Placement(np.zeros((0, 0, 4), dtype=np.uint8), top, left)
    y0, y1 = int(filled[0].min()), int(filled[0].max()) + 1
    x0, x1 = int(filled[1].min()), int(filled[1].max()) + 1
    return Placement(rgba[y0:y1, x0:x1], top + y0, left + x0)


def layer_from_render(
    stem: str, rendered: Path, top: int, left: int, scale_to: tuple[int, int] | None = None
) -> PsdLayer | None:
    """One layer from one rendered icon, or None when nothing was drawn.

    ``scale_to`` resizes the render before trimming. Inkscape rasterises an
    SVG at whatever size the document declares, which for a fragment is its
    own bounds in user units; the sheet layout needs it at the sheet's pixel
    scale instead.
    """
    rgba = load_rgba(rendered)
    if scale_to is not None and (rgba.shape[1], rgba.shape[0]) != scale_to:
        width, height = max(1, scale_to[0]), max(1, scale_to[1])
        with Image.fromarray(rgba) as image:
            rgba = np.array(image.resize((width, height), Image.LANCZOS))
    placement = trim(rgba, top, left)
    if placement.is_empty:
        return None
    return PsdLayer(name=stem, rgba=placement.rgba,
                    top=placement.top, left=placement.left)


def document_transform(
    layout: str,
    source_bounds: tuple[int, int, int, int],
    source_size: tuple[int, int],
    canvas_size: tuple[int, int],
    canvas_mode: str,
    uniform_scale: float,
) -> tuple[float, tuple[float, float]]:
    """How to move a shape's own geometry into the document's coordinates.

    In the sheet layout the artwork's coordinates *are* the document's, so
    nothing moves. In the canvas layout the shape is scaled and centred
    exactly as its single-file export is, and this reproduces that placement
    from the same numbers rather than guessing at it.
    """
    if layout == "sheet":
        return 1.0, (0.0, 0.0)

    target_w, target_h = canvas_size
    source_w, source_h = max(1, source_size[0]), max(1, source_size[1])
    if canvas_mode == "uniform_to_largest":
        scale = uniform_scale
    elif canvas_mode == "individual_fit":
        scale = min(target_w / source_w, target_h / source_h)
    else:
        scale = 1.0

    offset_x = (target_w - source_w * scale) / 2.0
    offset_y = (target_h - source_h * scale) / 2.0
    # The fragment's origin is the padded box corner, not the sheet's.
    return scale, (offset_x - source_bounds[0] * scale, offset_y - source_bounds[1] * scale)


def document_size(layout: str, sheet_size: tuple[int, int], canvas_size: tuple[int, int]) -> tuple[int, int]:
    """How big the PSD is, which the layout decides rather than the settings.

    In ``sheet`` layout the export canvas does not apply: the document is the
    artwork, and forcing it onto a square canvas would move every shape off
    the position the layout exists to preserve.
    """
    return sheet_size if layout == "sheet" else canvas_size
