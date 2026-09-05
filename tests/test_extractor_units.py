"""The extractor's own logic, separated from the Inkscape/potrace orchestration.

`IconExtractor` is mostly a pipeline over `svg_ops` and `raster_ops`, both
covered by their own suites. What lives *here* is the grouping algorithm, the
canvas and vector-mode decisions, and the cleanup afterwards — all reachable
without rendering anything.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from services.extraction_types import ExtractedIcon
from services.extractor import IconExtractor, extract_icon_palette
from services.geometry import Box
from services.settings_schema import AppSettings

VIEW_BOX = (0.0, 0.0, 100.0, 100.0)
RASTER_SHAPE = (100, 100)          # 1:1 with the viewBox, so boxes map straight across


def extractor(**overrides) -> IconExtractor:
    settings = AppSettings()
    for field, value in overrides.items():
        setattr(settings, field, value)
    return IconExtractor(settings)


def child(element_id: str) -> ET.Element:
    return ET.Element("path", {"id": element_id})


def grouping(children_boxes: dict[str, Box], raster_groups: list[Box]):
    """Run the grouping with one drawable per entry in `children_boxes`."""
    children = [child(name) for name in children_boxes]
    return extractor()._group_svg_children(
        children, children_boxes, raster_groups, RASTER_SHAPE, VIEW_BOX
    )


def ids(grouped) -> list[list[str]]:
    return [[c.attrib["id"] for c in members] for _box, members in grouped]


# --- grouping drawables into icons ----------------------------------------

def test_strokes_inside_one_raster_group_become_a_single_icon():
    """A multi-stroke drawing has to stay one asset."""
    grouped = grouping(
        {"a": Box(10, 10, 5, 5), "b": Box(20, 20, 5, 5)},
        [Box(0, 0, 40, 40)],
    )

    assert ids(grouped) == [["a", "b"]]


def test_separate_raster_groups_become_separate_icons():
    grouped = grouping(
        {"a": Box(10, 10, 5, 5), "b": Box(70, 70, 5, 5)},
        [Box(0, 0, 40, 40), Box(60, 60, 30, 30)],
    )

    assert ids(grouped) == [["a"], ["b"]]


def test_an_icon_box_is_the_union_of_its_strokes_not_the_raster_group():
    """The exported crop follows the artwork, not the detector's bounding box."""
    grouped = grouping(
        {"a": Box(10, 10, 5, 5), "b": Box(20, 20, 5, 5)},
        [Box(0, 0, 90, 90)],
    )

    assert grouped[0][0] == Box(10, 10, 15, 15)


def test_a_drawable_belongs_to_only_one_icon():
    """Overlapping raster groups must not export the same stroke twice."""
    grouped = grouping(
        {"a": Box(10, 10, 5, 5)},
        [Box(0, 0, 40, 40), Box(0, 0, 50, 50)],
    )

    assert ids(grouped) == [["a"]]


def test_a_stroke_the_raster_pass_missed_still_becomes_an_icon():
    """Losing artwork silently would be worse than an extra icon."""
    grouped = grouping(
        {"seen": Box(10, 10, 5, 5), "missed": Box(80, 80, 5, 5)},
        [Box(0, 0, 40, 40)],
    )

    assert ids(grouped) == [["seen"], ["missed"]]


def test_a_raster_group_containing_nothing_produces_no_icon():
    grouped = grouping({"a": Box(10, 10, 5, 5)}, [Box(0, 0, 40, 40), Box(60, 60, 20, 20)])
    assert ids(grouped) == [["a"]]


def test_icons_come_back_in_reading_order():
    grouped = grouping(
        {
            "bottom": Box(10, 80, 5, 5),
            "top_right": Box(80, 10, 5, 5),
            "top_left": Box(10, 10, 5, 5),
        },
        [Box(5, 75, 15, 15), Box(75, 5, 15, 15), Box(5, 5, 15, 15)],
    )

    assert ids(grouped) == [["top_left"], ["top_right"], ["bottom"]]


def test_membership_is_decided_by_the_stroke_centre():
    """A stroke poking out of the detected group still belongs to it."""
    grouped = grouping({"wide": Box(30, 10, 40, 5)}, [Box(0, 0, 55, 40)])
    assert ids(grouped) == [["wide"]]


def test_nothing_drawable_produces_no_icons():
    assert grouping({}, [Box(0, 0, 40, 40)]) == []


# --- the output canvas ----------------------------------------------------

def test_the_canvas_comes_from_the_configured_size():
    assert extractor(output_width=800, output_height=600)._canvas_size() == (800, 600)


@pytest.mark.parametrize("width,height", [(0, 512), (512, 0), (-10, -10)])
def test_a_non_positive_canvas_is_clamped_rather_than_crashing(width, height):
    """Validation rejects these, but the engine must not divide by zero."""
    canvas = extractor(output_width=width, output_height=height)._canvas_size()
    assert canvas[0] >= 1 and canvas[1] >= 1


# --- what vector mode gets recorded ---------------------------------------

def test_no_svg_export_means_no_vector_mode(tmp_path):
    assert extractor()._vector_mode_for_svg_source({"png": tmp_path / "a.png"}) is None
    assert extractor()._vector_mode_for_raster_crop({"png": tmp_path / "a.png"}, None) is None


def test_an_svg_from_svg_input_is_recorded_as_vector_native(tmp_path):
    assert extractor()._vector_mode_for_svg_source({"svg": tmp_path / "a.svg"}) == (
        "vector-native-grouped"
    )


def test_a_traced_crop_without_a_preview_is_assumed_monochrome(tmp_path):
    assert extractor()._vector_mode_for_raster_crop({"svg": tmp_path / "a.svg"}, None) == (
        "traced-monochrome"
    )


# --- cleaning up empty working directories --------------------------------

def make_dirs(root, *names):
    for name in names:
        (root / name).mkdir(parents=True)


def test_the_working_directories_are_removed_when_empty(tmp_path):
    make_dirs(tmp_path, "_work_png", "_work_svg")

    extractor()._prune_empty_dirs(tmp_path, {"png", "svg"})

    assert not (tmp_path / "_work_png").exists()
    assert not (tmp_path / "_work_svg").exists()


def test_a_format_directory_is_removed_when_that_format_was_not_requested(tmp_path):
    make_dirs(tmp_path, "png", "svg")

    extractor()._prune_empty_dirs(tmp_path, {"jpg"})

    assert not (tmp_path / "png").exists()
    assert not (tmp_path / "svg").exists()


def test_a_requested_format_directory_is_kept_even_when_empty(tmp_path):
    make_dirs(tmp_path, "png", "svg")

    extractor()._prune_empty_dirs(tmp_path, {"png", "svg"})

    assert (tmp_path / "png").exists()
    assert (tmp_path / "svg").exists()


def test_a_directory_holding_files_is_never_removed(tmp_path):
    """Pruning must not delete anyone's exports."""
    make_dirs(tmp_path, "_work_png")
    (tmp_path / "_work_png" / "leftover.png").write_bytes(b"data")

    extractor()._prune_empty_dirs(tmp_path, {"png"})

    assert (tmp_path / "_work_png" / "leftover.png").exists()


def test_pruning_a_folder_without_those_directories_is_harmless(tmp_path):
    extractor()._prune_empty_dirs(tmp_path, {"png"})


# --- palette sampling -----------------------------------------------------

def test_an_icon_with_neither_bitmap_nor_svg_has_no_palette():
    assert extract_icon_palette(None, None) == (None, [])


def test_a_missing_file_is_not_sampled(tmp_path):
    assert extract_icon_palette(tmp_path / "absent.png", tmp_path / "absent.svg") == (None, [])


def test_an_svg_that_cannot_be_rendered_yields_no_palette(monkeypatch, tmp_path):
    """A palette is metadata, not a reason to fail the export."""
    import services.extractor as ex

    svg = tmp_path / "icon.svg"
    svg.write_text("<svg/>")
    monkeypatch.setattr(
        ex, "export_svg_to_png", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no inkscape"))
    )

    assert extract_icon_palette(None, svg) == (None, [])


# --- relocating icons into the published export ---------------------------

def test_an_icon_keeps_its_identity_when_moved_out_of_staging(tmp_path):
    from services.extractor import _relocate_icon

    class FakeStaging:
        def final_path(self, path):
            return tmp_path / "published" / path.name

    icon = ExtractedIcon(
        index=1,
        stem="icon_001",
        outputs={"png": tmp_path / "staging" / "icon_001.png"},
        preview_path=tmp_path / "staging" / "icon_001.png",
        canvas_size=(512, 512),
        source_size=(44, 42),
        source_bounds=(0, 0, 44, 42),
    )

    moved = _relocate_icon(icon, FakeStaging())

    assert moved.stem == "icon_001"
    assert moved.outputs["png"] == tmp_path / "published" / "icon_001.png"
    assert icon.outputs["png"] == tmp_path / "staging" / "icon_001.png", "original untouched"


# --- asking Inkscape about the sheet --------------------------------------
#
# These two methods exist to marshal a parsed tree back onto disk for the
# external tools. The tools are mocked; what is checked is that a valid SVG
# reaches them and that the result is passed through.

def parsed_sheet() -> ET.ElementTree:
    return ET.ElementTree(
        ET.fromstring('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"/>')
    )


def test_element_bounds_are_queried_from_a_written_out_copy(monkeypatch):
    import services.extractor as ex

    seen = {}

    def fake_query(path):
        seen["existed"] = path.exists()
        seen["parsed"] = ET.parse(path).getroot().tag
        return {"a": Box(1, 2, 3, 4)}

    monkeypatch.setattr(ex, "query_svg_boxes", fake_query)

    boxes = extractor()._query_element_boxes(parsed_sheet())

    assert boxes == {"a": Box(1, 2, 3, 4)}
    assert seen["existed"] is True
    assert seen["parsed"].endswith("svg")


def test_the_temporary_copy_does_not_outlive_the_query(monkeypatch):
    import services.extractor as ex

    captured = {}

    def fake_query(path):
        captured["path"] = path
        return {}

    monkeypatch.setattr(ex, "query_svg_boxes", fake_query)
    extractor()._query_element_boxes(parsed_sheet())

    assert not captured["path"].exists()


def test_raster_groups_are_detected_on_a_rendered_proof(monkeypatch):
    import numpy as np

    import services.extractor as ex

    # A white sheet with two dark blobs, far enough apart not to merge.
    proof = np.full((100, 100), 255, dtype=np.uint8)
    proof[10:30, 10:30] = 0
    proof[70:90, 70:90] = 0

    monkeypatch.setattr(ex, "render_svg_to_png", lambda _src, _dst: None)
    monkeypatch.setattr(ex.cv2, "imread", lambda *_a, **_k: proof)

    groups, shape = extractor(min_area=50, merge_gap=9)._detect_raster_groups(parsed_sheet())

    assert shape == (100, 100), "width and height, in that order"
    assert len(groups) == 2


def test_a_sheet_that_will_not_rasterize_is_reported(monkeypatch):
    """cv2 returns None rather than raising, so this must be caught explicitly."""
    import services.extractor as ex

    monkeypatch.setattr(ex, "render_svg_to_png", lambda _src, _dst: None)
    monkeypatch.setattr(ex.cv2, "imread", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="rasterize"):
        extractor()._detect_raster_groups(parsed_sheet())
