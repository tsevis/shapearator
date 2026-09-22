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

    def install(boxes, proof=None):
        def fake_query(_path):
            calls["query"] += 1
            return dict(boxes)

        def fake_render(_src, dst):
            # A white sheet with two dark blobs, matching the loose fixture's
            # two visual clusters at 1:1 with the viewBox.
            calls["render"] += 1
            sheet_proof = proof
            if sheet_proof is None:
                sheet_proof = np.full((100, 100), 255, dtype=np.uint8)
                sheet_proof[8:30, 8:30] = 0
                sheet_proof[58:72, 58:72] = 0
            Image.fromarray(sheet_proof).save(dst)

        def fake_export(src, dst):
            calls["export"].append((Path(src).name, Path(dst).name))
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            blank = any(stem in Path(src).name for stem in calls.get("blank", ()))
            fill = (0, 0, 0, 0) if blank else (0, 0, 0, 255)
            Image.new("RGBA", (64, 64), fill).save(dst)

        monkeypatch.setattr(ex, "query_svg_boxes", fake_query)
        monkeypatch.setattr(ex, "render_svg_to_png", fake_render)
        monkeypatch.setattr(ex, "export_svg_to_png", fake_export)
        return calls

    return install


# --- grouped artwork is taken as authored ---------------------------------

def test_each_authored_group_becomes_one_icon(sheet, tmp_path, fake_inkscape):
    calls = fake_inkscape(GROUPED_BOXES)

    icons, _warnings = extractor()._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert [i.stem for i in icons] == ["icon_001", "icon_002"]
    assert calls["render"] == 0, "grouped artwork needs no raster detection pass"


def test_a_multi_shape_group_stays_a_single_icon(sheet, tmp_path, fake_inkscape):
    """The drum-kit case: nineteen paths authored as one icon stay one icon."""
    fake_inkscape(GROUPED_BOXES)

    icons, _warnings = extractor()._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    first = ET.parse(icons[0].outputs["svg"]).getroot()
    assert sum(1 for _ in first.iter("{http://www.w3.org/2000/svg}rect")) == 2


def test_detection_settings_do_not_touch_grouped_artwork(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)

    icons, _warnings = extractor(min_area=99999, merge_gap=99)._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert len(icons) == 2


# --- loose artwork is clustered visually ----------------------------------

def test_loose_shapes_are_clustered_by_the_raster_pass(sheet, tmp_path, fake_inkscape):
    calls = fake_inkscape(LOOSE_BOXES)

    icons, _warnings = extractor(min_area=50, merge_gap=9)._extract_from_svg(
        sheet(LOOSE_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert calls["render"] == 1, "loose artwork needs the raster proof"
    assert len(icons) == 2, "two visual clusters from three shapes"


def test_nearby_strokes_end_up_in_the_same_icon(sheet, tmp_path, fake_inkscape):
    fake_inkscape(LOOSE_BOXES)

    icons, _warnings = extractor(min_area=50, merge_gap=9)._extract_from_svg(
        sheet(LOOSE_SHEET), tmp_path / "out", {"svg"}, None
    )

    first = ET.parse(icons[0].outputs["svg"]).getroot()
    assert sum(1 for _ in first.iter("{http://www.w3.org/2000/svg}rect")) == 2


# --- what each format selection produces ----------------------------------

def test_an_svg_only_run_writes_vectors_and_no_bitmap(sheet, tmp_path, fake_inkscape):
    calls = fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    icons, _warnings = extractor()._extract_from_svg(sheet(GROUPED_SHEET), out, {"svg"}, None)

    assert set(icons[0].outputs) == {"svg"}
    assert icons[0].outputs["svg"].exists()
    assert calls["export"] == [], "no icon should have been rasterized"
    assert icons[0].vector_mode == "vector-native-grouped"


def test_a_bitmap_only_run_writes_no_vector(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    icons, _warnings = extractor()._extract_from_svg(sheet(GROUPED_SHEET), out, {"png"}, None)

    assert set(icons[0].outputs) == {"png"}
    assert icons[0].outputs["png"].exists()
    assert icons[0].vector_mode is None


def test_both_formats_are_written_from_one_run(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)

    icons, _warnings = extractor()._extract_from_svg(
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

    icons, _warnings = extractor(padding=5)._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    box = GROUPED_BOXES["i1"]
    assert icons[0].source_size == (box.w + 10, box.h + 10)


def test_the_recorded_bounds_start_at_the_padded_corner(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)

    icons, _warnings = extractor(padding=5)._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    box = GROUPED_BOXES["i1"]
    assert icons[0].source_bounds[:2] == (box.x - 5, box.y - 5)


def test_every_icon_lands_on_the_configured_canvas(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)

    icons, _warnings = extractor(output_width=256, output_height=200)._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert all(i.canvas_size == (256, 200) for i in icons)


def test_icons_are_numbered_from_one_with_padded_stems(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)

    icons, _warnings = extractor()._extract_from_svg(
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

    icons, _warnings = extractor()._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert len(icons) == 2


# --- the split mode the user chose ----------------------------------------

#: A mosaic proof: every tile touches its neighbour, so the whole sheet is one
#: connected blob and clustering can only ever return a single icon.
def _one_blob_proof():
    proof = np.full((100, 100), 255, dtype=np.uint8)
    proof[5:95, 5:95] = 0
    return proof


def test_shape_split_gives_every_loose_shape_its_own_icon(sheet, tmp_path, fake_inkscape):
    """The mosaic case: 417 touching tiles must not become one icon."""
    calls = fake_inkscape(LOOSE_BOXES)

    icons, _warnings = extractor(svg_split="shape")._extract_from_svg(
        sheet(LOOSE_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert len(icons) == 3, "one icon per shape, whatever the pixels say"
    assert calls["render"] == 0, "structure was asked for; no raster proof is needed"


def test_shape_split_writes_one_shape_per_file(sheet, tmp_path, fake_inkscape):
    fake_inkscape(LOOSE_BOXES)

    icons, _warnings = extractor(svg_split="shape")._extract_from_svg(
        sheet(LOOSE_SHEET), tmp_path / "out", {"svg"}, None
    )

    counts = [
        sum(1 for _ in ET.parse(i.outputs["svg"]).getroot().iter("{http://www.w3.org/2000/svg}rect"))
        for i in icons
    ]
    assert counts == [1, 1, 1]


def test_cluster_split_overrides_authored_groups(sheet, tmp_path, fake_inkscape):
    """The escape hatch the other way: ignore the <g>s and go by pixels."""
    calls = fake_inkscape(GROUPED_BOXES)

    icons, _warnings = extractor(svg_split="cluster", min_area=50, merge_gap=9)._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert calls["render"] == 1, "clustering was asked for; the raster proof is needed"
    assert len(icons) == 2


# --- telling the user when clustering swallowed the sheet -----------------

def test_a_collapsed_auto_run_says_so(sheet, tmp_path, fake_inkscape):
    """Silence here is the original bug: one file, and no idea why."""
    fake_inkscape(LOOSE_BOXES, proof=_one_blob_proof())

    icons, warnings = extractor(min_area=50, merge_gap=9)._extract_from_svg(
        sheet(LOOSE_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert len(icons) == 1
    assert len(warnings) == 1
    assert "3 shapes" in warnings[0] and "Every shape" in warnings[0]


def test_an_ordinary_cluster_run_stays_quiet(sheet, tmp_path, fake_inkscape):
    fake_inkscape(LOOSE_BOXES)

    _icons, warnings = extractor(min_area=50, merge_gap=9)._extract_from_svg(
        sheet(LOOSE_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert warnings == ()


def test_a_chosen_shape_split_never_warns_about_collapsing(sheet, tmp_path, fake_inkscape):
    """The warning names a mistake; obeying the user is not one."""
    fake_inkscape(LOOSE_BOXES, proof=_one_blob_proof())

    _icons, warnings = extractor(svg_split="shape")._extract_from_svg(
        sheet(LOOSE_SHEET), tmp_path / "out", {"svg"}, None
    )

    assert warnings == ()


# --- one layered Photoshop file per sheet ---------------------------------

def _psd_layers(path):
    from psd_tools import PSDImage
    return [(l.name, l.offset, l.size) for l in PSDImage.open(path)]


def test_a_psd_run_writes_one_file_holding_every_icon(sheet, tmp_path, fake_inkscape):
    """The point of the format: forty files become one, forty layers deep."""
    fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    icons, _warnings = extractor()._extract_from_svg(
        sheet(GROUPED_SHEET), out, {"psd"}, None)

    written = out / "psd" / "sheet.psd"
    assert written.exists(), sorted(p.name for p in out.iterdir())
    assert len(_psd_layers(written)) == len(icons) == 2


def test_the_sheet_layout_keeps_each_shape_where_it_was(sheet, tmp_path, fake_inkscape):
    """Stacked in the middle you would have to move every layer by hand."""
    fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    extractor(psd_layout="sheet", padding=0)._extract_from_svg(
        sheet(GROUPED_SHEET), out, {"psd"}, None)

    offsets = {name: offset for name, offset, _size in _psd_layers(out / "psd" / "sheet.psd")}
    assert offsets["icon_001"] == (GROUPED_BOXES["i1"].x, GROUPED_BOXES["i1"].y)
    assert offsets["icon_002"] == (GROUPED_BOXES["i2"].x, GROUPED_BOXES["i2"].y)


def test_the_sheet_layout_makes_a_document_the_size_of_the_artwork(sheet, tmp_path, fake_inkscape):
    from psd_tools import PSDImage
    fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    extractor(psd_layout="sheet", output_width=999, output_height=777)._extract_from_svg(
        sheet(GROUPED_SHEET), out, {"psd"}, None)

    assert PSDImage.open(out / "psd" / "sheet.psd").size == (100, 100), "the sheet's viewBox"


def test_the_canvas_layout_makes_a_document_the_size_of_the_export_canvas(sheet, tmp_path, fake_inkscape):
    from psd_tools import PSDImage
    fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    extractor(psd_layout="canvas", output_width=300, output_height=200)._extract_from_svg(
        sheet(GROUPED_SHEET), out, {"psd"}, None)

    assert PSDImage.open(out / "psd" / "sheet.psd").size == (300, 200)


def test_a_run_without_psd_writes_no_psd_folder(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    extractor()._extract_from_svg(sheet(GROUPED_SHEET), out, {"svg"}, None)

    assert not (out / "psd").exists()


def test_psd_and_svg_can_be_asked_for_together(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    icons, _warnings = extractor()._extract_from_svg(
        sheet(GROUPED_SHEET), out, {"psd", "svg"}, None)

    assert (out / "psd" / "sheet.psd").exists()
    assert all(icon.outputs["svg"].exists() for icon in icons)


def test_a_shape_that_rasterises_to_nothing_is_reported_not_dropped(sheet, tmp_path, fake_inkscape):
    """A degenerate 1pt path is a real thing in real artwork.

    It cannot become a layer -- PSD has no zero-area layer -- but a sheet that
    exported 36 files and produced 35 layers, saying nothing, is the silent
    loss this whole format is supposed to avoid.
    """
    calls = fake_inkscape(GROUPED_BOXES)
    calls["blank"] = ("icon_002",)
    out = tmp_path / "out"

    icons, warnings = extractor()._extract_from_svg(
        sheet(GROUPED_SHEET), out, {"psd"}, None)

    assert len(icons) == 2
    assert len(_psd_layers(out / "psd" / "sheet.psd")) == 1
    assert any("icon_002" in warning for warning in warnings), warnings


def test_a_psd_where_everything_drew_says_nothing(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)

    _icons, warnings = extractor()._extract_from_svg(
        sheet(GROUPED_SHEET), tmp_path / "out", {"psd"}, None)

    assert warnings == ()


def _path_names(psd_path):
    from psd_tools import PSDImage
    resources = PSDImage.open(psd_path).image_resources
    return {resources[key].name for key in resources
            if 2000 <= int(key) <= 2997}


def test_bitmap_paths_puts_every_shape_in_the_paths_panel(sheet, tmp_path, fake_inkscape):
    fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    extractor(psd_layers="bitmap_paths")._extract_from_svg(
        sheet(GROUPED_SHEET), out, {"psd"}, None)

    assert _path_names(out / "psd" / "sheet.psd") == {"icon_001", "icon_002"}


def test_plain_bitmap_layers_carry_no_paths(sheet, tmp_path, fake_inkscape):
    """Paths cost geometry and file size; only the mode that asked pays."""
    fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    extractor(psd_layers="bitmap")._extract_from_svg(
        sheet(GROUPED_SHEET), out, {"psd"}, None)

    assert _path_names(out / "psd" / "sheet.psd") == set()


def test_a_path_lands_where_the_shape_is(sheet, tmp_path, fake_inkscape):
    """A rect at 10,10 in a 100pt sheet is a tenth of the way in."""
    from psd_tools import PSDImage
    from services import psd_vector
    fake_inkscape(GROUPED_BOXES)
    out = tmp_path / "out"

    extractor(psd_layers="bitmap_paths", psd_layout="sheet")._extract_from_svg(
        sheet(GROUPED_SHEET), out, {"psd"}, None)

    resources = PSDImage.open(out / "psd" / "sheet.psd").image_resources
    blob = next(resources.get_data(int(k)) for k in resources
                if 2000 <= int(k) <= 2997 and resources[int(k)].name == "icon_001")
    anchors = [psd_vector.read_knot(blob, i).anchor for i in range(1, 5)]
    assert min(v for v, _h in anchors) == pytest.approx(0.10, abs=0.01)
    assert min(h for _v, h in anchors) == pytest.approx(0.10, abs=0.01)
