"""The extractor's SVG pipeline, with only Inkscape faked.

`_extract_from_svg` and `_export_svg_icon` are the two methods that thread the
external tools together. Rather than stubbing every seam and asserting a call
order — which pins the implementation's shape instead of its behaviour — these
tests fake exactly three functions, the Inkscape entry points:

    query_svg_boxes     measuring elements on the sheet
    render_svg_to_png   rasterizing the whole sheet for grouping
    export_svg_to_png   rendering one icon's fragment

Everything else runs for real: id assignment, icon discovery, grouped-versus-
loose detection, fragment building, canvas normalization and the bitmap
conversions. The SVG files these tests inspect are the ones the engine actually
writes.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import services.extractor as ex
from services.extractor import IconExtractor
from services.geometry import Box
from services.settings_schema import AppSettings

GROUPED_SHEET = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <g id="i1"><rect x="10" y="10" width="10" height="10"/><rect x="22" y="10" width="6" height="6"/></g>
  <g id="i2"><rect x="60" y="60" width="10" height="10"/></g>
</svg>"""

LOOSE_SHEET = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="10" y="10" width="10" height="10"/>
  <rect x="22" y="10" width="6" height="6"/>
  <rect x="60" y="60" width="10" height="10"/>
</svg>"""

GROUPED_BOXES = {"i1": Box(10, 10, 18, 10), "i2": Box(60, 60, 10, 10)}
LOOSE_BOXES = {
    "shape_0001": Box(10, 10, 10, 10),
    "shape_0002": Box(22, 10, 6, 6),
    "shape_0003": Box(60, 60, 10, 10),
}


def extractor(**overrides) -> IconExtractor:
    settings = AppSettings()
    for field, value in overrides.items():
        setattr(settings, field, value)
    return IconExtractor(settings)


@pytest.fixture
def sheet(tmp_path):
    def write(markup: str) -> Path:
        path = tmp_path / "sheet.svg"
        path.write_text(markup)
        return path

    return write


@pytest.fixture
def fake_inkscape(monkeypatch):
    """Replace the three Inkscape calls; leave the rest of the engine alone."""
    calls = {"query": 0, "render": 0, "export": []}

    def install(boxes):
        def fake_query(_path):
            calls["query"] += 1
            return dict(boxes)

        def fake_render(_src, dst):
            # A white sheet with two dark blobs, matching the loose fixture's
            # two visual clusters at 1:1 with the viewBox.
            calls["render"] += 1
            proof = np.full((100, 100), 255, dtype=np.uint8)
            proof[8:30, 8:30] = 0
            proof[58:72, 58:72] = 0
            Image.fromarray(proof).save(dst)

        def fake_export(src, dst):
            calls["export"].append((Path(src).name, Path(dst).name))
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGBA", (64, 64), (0, 0, 0, 255)).save(dst)

        monkeypatch.setattr(ex, "query_svg_boxes", fake_query)
        monkeypatch.setattr(ex, "render_svg_to_png", fake_render)
        monkeypatch.setattr(ex, "export_svg_to_png", fake_export)
        return calls

    return install


# --- grouped artwork is taken as authored ---------------------------------

def test_each_authored_group_becomes_one_icon(sheet, tmp_path, fake_inkscape):
    calls = fake_inkscape(GROUPED_BOXES)

    icons = extractor()._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert [i.stem for i in icons] == ["icon_001", "icon_002"]
    assert calls["render"] == 0, "grouped artwork needs no raster detection pass"


def test_a_multi_shape_group_stays_a_single_icon(sheet, tmp_path, fake_inkscape):
    """The drum-kit case: nineteen paths authored as one icon stay one icon."""
    fake_inkscape(GROUPED_BOXES)

    icons = extractor()._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    first = ET.parse(icons[0].outputs["svg"]).getroot()
    assert sum(1 for _ in first.iter("{http://www.w3.org/2000/svg}rect")) == 2


def test_detection_settings_do_not_touch_grouped_artwork(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)

    icons = extractor(min_area=99999, merge_gap=99)._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert len(icons) == 2


# --- loose artwork is clustered visually ----------------------------------

def test_loose_shapes_are_clustered_by_the_raster_pass(sheet, tmp_path, fake_inkscape):
    calls = fake_inkscape(LOOSE_BOXES)

    icons = extractor(min_area=50, merge_gap=9)._extract_from_svg(
        sheet(LOOSE_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert calls["render"] == 1, "loose artwork needs the raster proof"
    assert len(icons) == 2, "two visual clusters from three shapes"


def test_nearby_strokes_end_up_in_the_same_icon(sheet, tmp_path, fake_inkscape):
    fake_inkscape(LOOSE_BOXES)

    icons = extractor(min_area=50, merge_gap=9)._extract_from_svg(
        sheet(LOOSE_SHEET), tmp_path / "out", {"svg"}, None
    )

    first = ET.parse(icons[0].outputs["svg"]).getroot()
    assert sum(1 for _ in first.iter("{http://www.w3.org/2000/svg}rect")) == 2


# --- what each format selection produces ----------------------------------

def test_an_svg_only_run_writes_vectors_and_no_bitmap(sheet, tmp_path, fake_inkscape):
    calls = fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    icons = extractor()._extract_from_svg(sheet(GROUPED_SHEET), out, {"svg"}, None)

    assert set(icons[0].outputs) == {"svg"}
    assert icons[0].outputs["svg"].exists()
    assert calls["export"] == [], "no icon should have been rasterized"
    assert icons[0].vector_mode == "vector-native-grouped"


def test_a_bitmap_only_run_writes_no_vector(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    icons = extractor()._extract_from_svg(sheet(GROUPED_SHEET), out, {"png"}, None)

    assert set(icons[0].outputs) == {"png"}
    assert icons[0].outputs["png"].exists()
    assert icons[0].vector_mode is None


def test_both_formats_are_written_from_one_run(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)

    icons = extractor()._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"png", "svg"}, None
    )

    assert set(icons[0].outputs) == {"png", "svg"}
    assert icons[0].preview_path == icons[0].outputs["png"]


def test_the_working_svg_is_cleaned_up_when_no_vector_was_asked_for(
    sheet, tmp_path, fake_inkscape
):
    """The intermediate fragment is scaffolding, not an export."""
    fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    extractor()._extract_from_svg(sheet(GROUPED_SHEET), out, {"png"}, None)

    assert not (out / "_work_svg").exists()
    assert not (out / "_work_png").exists()


def test_a_format_directory_is_not_left_behind_empty(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    extractor()._extract_from_svg(sheet(GROUPED_SHEET), out, {"png"}, None)

    assert not (out / "svg").exists()


# --- the geometry recorded for each icon ----------------------------------

def test_padding_widens_the_recorded_source_size(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)

    icons = extractor(padding=5)._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    box = GROUPED_BOXES["i1"]
    assert icons[0].source_size == (box.w + 10, box.h + 10)


def test_the_recorded_bounds_start_at_the_padded_corner(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)

    icons = extractor(padding=5)._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    box = GROUPED_BOXES["i1"]
    assert icons[0].source_bounds[:2] == (box.x - 5, box.y - 5)


def test_every_icon_lands_on_the_configured_canvas(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)

    icons = extractor(output_width=256, output_height=200)._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert all(i.canvas_size == (256, 200) for i in icons)


def test_icons_are_numbered_from_one_with_padded_stems(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)

    icons = extractor()._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert [i.index for i in icons] == [1, 2]
    assert [i.stem for i in icons] == ["icon_001", "icon_002"]


# --- progress reporting ---------------------------------------------------

def test_progress_covers_detection_and_every_icon(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)
    seen = []

    extractor()._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, seen.append
    )

    phases = [p.phase for p in seen]
    assert phases == ["detect", "export", "export"]
    assert seen[0].message == "Detected 2 icons"
    assert seen[-1].message == "Exporting icon 2 of 2"


def test_a_run_without_a_progress_callback_is_fine(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)

    icons = extractor()._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert len(icons) == 2
