"""Deciding what a PSD layer is and where it sits.

The file format is covered by `test_psd_writer.py`. These cover the half that
turns icons into layers: trimming, placement, and which of the two layouts
decides the document's size.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from services import psd_export


def rgba(height: int, width: int, alpha: int = 255) -> np.ndarray:
    patch = np.zeros((height, width, 4), dtype=np.uint8)
    patch[:, :, 3] = alpha
    return patch


def framed(outer: int, inner: int) -> np.ndarray:
    """An `inner` square of opaque pixels centred in a transparent square."""
    patch = np.zeros((outer, outer, 4), dtype=np.uint8)
    start = (outer - inner) // 2
    patch[start:start + inner, start:start + inner, 3] = 255
    return patch


# --- trimming the margin ---------------------------------------------------

def test_a_transparent_margin_is_dropped():
    placement = psd_export.trim(framed(100, 20))
    assert placement.rgba.shape[:2] == (20, 20)


def test_what_was_dropped_is_added_to_the_offset():
    """Trimming without moving the layer would shift every shape up and left."""
    placement = psd_export.trim(framed(100, 20), top=300, left=500)
    assert (placement.top, placement.left) == (300 + 40, 500 + 40)


def test_a_layer_of_nothing_trims_to_nothing():
    placement = psd_export.trim(rgba(50, 50, alpha=0), top=7, left=9)
    assert placement.is_empty
    assert (placement.top, placement.left) == (7, 9), "an empty layer keeps its place"


def test_an_already_tight_layer_is_left_alone():
    placement = psd_export.trim(rgba(12, 34), top=1, left=2)
    assert placement.rgba.shape[:2] == (12, 34)
    assert (placement.top, placement.left) == (1, 2)


# --- one layer from one render --------------------------------------------

@pytest.fixture
def render(tmp_path):
    def write(patch: np.ndarray, name: str = "icon_001.png"):
        path = tmp_path / name
        Image.fromarray(patch).save(path)
        return path

    return write


def test_a_render_becomes_a_placed_layer(render):
    layer = psd_export.layer_from_render("icon_007", render(framed(80, 16)), top=10, left=20)
    assert layer is not None
    assert layer.name == "icon_007"
    assert (layer.height, layer.width) == (16, 16)
    assert (layer.top, layer.left) == (10 + 32, 20 + 32)


def test_a_blank_render_produces_no_layer(render):
    """An icon that rendered to nothing must not become an empty layer."""
    assert psd_export.layer_from_render("blank", render(rgba(30, 30, alpha=0)), 0, 0) is None


def test_a_render_is_resized_to_the_size_asked_for(render):
    """Inkscape rasterises a fragment at its own user units, not the sheet's."""
    layer = psd_export.layer_from_render(
        "icon_001", render(rgba(40, 40)), top=0, left=0, scale_to=(120, 90))
    assert (layer.width, layer.height) == (120, 90)


def test_a_render_already_the_right_size_is_not_resampled(render):
    layer = psd_export.layer_from_render(
        "icon_001", render(rgba(64, 64)), top=0, left=0, scale_to=(64, 64))
    assert (layer.width, layer.height) == (64, 64)


# --- which layout decides the document ------------------------------------

def test_the_sheet_layout_uses_the_sheet_s_own_size():
    """Forcing the artwork onto a square canvas would move every shape."""
    assert psd_export.document_size("sheet", (1600, 900), (1024, 1024)) == (1600, 900)


def test_the_canvas_layout_uses_the_export_canvas():
    assert psd_export.document_size("canvas", (1600, 900), (1024, 1024)) == (1024, 1024)
