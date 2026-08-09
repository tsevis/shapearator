"""Characterization tests for SVG parsing, fragment building, and normalization."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from services.geometry import Box
from services.svg_ops import (
    SVG_NS,
    build_svg_fragment,
    ensure_element_ids,
    get_svg_canvas_size,
    inject_svg_metadata,
    normalize_svg_to_canvas,
    parse_svg_length,
    parse_viewbox,
    query_svg_boxes,
    require_binary,
    svg_box_from_raster_box,
)


def _root(**attrib) -> ET.Element:
    return ET.Element(f"{{{SVG_NS}}}svg", attrib)


def _child(parent: ET.Element, tag: str = "path", **attrib) -> ET.Element:
    return ET.SubElement(parent, f"{{{SVG_NS}}}{tag}", attrib)


def _tags(element: ET.Element) -> list[str]:
    return [child.tag.split("}")[-1] for child in element]


# --- length and viewBox parsing ------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [("120", 120.0), ("120px", 120.0), (" 12.5 ", 12.5), ("auto", 0.0), ("", 0.0)],
)
def test_parse_svg_length(raw, expected):
    assert parse_svg_length(raw) == expected


def test_parse_viewbox_accepts_comma_and_space_separators():
    assert parse_viewbox(_root(viewBox="0 0 100 200")) == (0.0, 0.0, 100.0, 200.0)
    assert parse_viewbox(_root(viewBox="0,0,100,200")) == (0.0, 0.0, 100.0, 200.0)


def test_parse_viewbox_requires_the_attribute():
    with pytest.raises(RuntimeError, match="viewBox"):
        parse_viewbox(_root())


def test_parse_viewbox_rejects_the_wrong_arity():
    with pytest.raises(RuntimeError, match="Unexpected viewBox"):
        parse_viewbox(_root(viewBox="0 0 100"))


def test_canvas_size_prefers_viewbox_over_width_height():
    assert get_svg_canvas_size(_root(viewBox="0 0 64 32", width="999", height="999")) == (64.0, 32.0)


def test_canvas_size_falls_back_to_width_and_height():
    assert get_svg_canvas_size(_root(width="64px", height="32px")) == (64.0, 32.0)


def test_canvas_size_never_returns_zero():
    assert get_svg_canvas_size(_root()) == (1.0, 1.0)


# --- ids ------------------------------------------------------------------

def test_ensure_element_ids_fills_only_the_missing_ones():
    root = _root()
    _child(root, id="keep-me")
    _child(root)
    ensure_element_ids(root)
    assert [child.attrib["id"] for child in root] == ["keep-me", "shape_0001"]


def test_ensure_element_ids_skips_defs():
    root = _root()
    _child(root, "defs")
    _child(root)
    ensure_element_ids(root)
    assert "id" not in root[0].attrib
    assert root[1].attrib["id"] == "shape_0001"


def test_ensure_element_ids_never_collides_with_an_existing_id():
    root = _root()
    _child(root)  # would naively become shape_0001
    _child(root, id="shape_0001")  # already taken further down the document
    ensure_element_ids(root)
    ids = [child.attrib["id"] for child in root]
    assert len(set(ids)) == len(ids)


# --- fragment building ----------------------------------------------------

def test_fragment_viewbox_covers_the_bounds_plus_padding(tmp_path):
    root = _root(viewBox="0 0 500 500")
    child = _child(root, id="a")
    out = tmp_path / "frag.svg"
    build_svg_fragment(root, [child], Box(100, 100, 40, 20), padding=10, output_path=out)
    fragment = ET.parse(out).getroot()
    assert fragment.attrib["viewBox"] == "0 0 60 40"


def test_fragment_translates_children_into_the_local_frame(tmp_path):
    root = _root(viewBox="0 0 500 500")
    child = _child(root, id="a")
    out = tmp_path / "frag.svg"
    build_svg_fragment(root, [child], Box(100, 50, 40, 20), padding=10, output_path=out)
    fragment = ET.parse(out).getroot()
    drawable = next(node for node in fragment if node.tag.endswith("path"))
    assert drawable.attrib["transform"] == "translate(-90 -40)"


def test_fragment_preserves_an_existing_child_transform(tmp_path):
    root = _root(viewBox="0 0 500 500")
    child = _child(root, id="a", transform="rotate(45)")
    out = tmp_path / "frag.svg"
    build_svg_fragment(root, [child], Box(0, 0, 10, 10), padding=0, output_path=out)
    drawable = next(node for node in ET.parse(out).getroot() if node.tag.endswith("path"))
    assert drawable.attrib["transform"] == "translate(0 0) rotate(45)"


def test_fragment_does_not_mutate_the_source_tree(tmp_path):
    root = _root(viewBox="0 0 500 500")
    child = _child(root, id="a")
    build_svg_fragment(root, [child], Box(100, 50, 40, 20), 10, tmp_path / "frag.svg")
    assert "transform" not in child.attrib


def test_fragment_carries_the_root_style_attribute(tmp_path):
    root = _root(viewBox="0 0 500 500", style="fill:none")
    child = _child(root, id="a")
    out = tmp_path / "frag.svg"
    build_svg_fragment(root, [child], Box(0, 0, 10, 10), 0, out)
    assert ET.parse(out).getroot().attrib["style"] == "fill:none"


# --- canvas normalization -------------------------------------------------

def test_normalize_centres_and_scales_into_the_target_canvas(tmp_path):
    source = tmp_path / "in.svg"
    root = _root(viewBox="0 0 100 100")
    _child(root, id="a")
    ET.ElementTree(root).write(source)
    out = tmp_path / "out.svg"
    normalize_svg_to_canvas(source, out, (200, 200), "individual_fit", uniform_scale=1.0)

    canvas = ET.parse(out).getroot()
    assert canvas.attrib["viewBox"] == "0 0 200 200"
    assert canvas.attrib["width"] == "200"
    group = next(node for node in canvas if node.tag.endswith("g"))
    assert "scale(2.000000)" in group.attrib["transform"]
    assert "translate(0.0000 0.0000)" in group.attrib["transform"]


def test_normalize_keeps_defs_outside_the_scaling_group(tmp_path):
    source = tmp_path / "in.svg"
    root = _root(viewBox="0 0 100 100")
    defs = _child(root, "defs")
    ET.SubElement(defs, f"{{{SVG_NS}}}linearGradient", {"id": "grad"})
    _child(root, id="a", fill="url(#grad)")
    ET.ElementTree(root).write(source)
    out = tmp_path / "out.svg"
    normalize_svg_to_canvas(source, out, (200, 200), "original", uniform_scale=1.0)

    canvas = ET.parse(out).getroot()
    assert _tags(canvas) == ["defs", "g"]
    gradients = canvas.findall(f".//{{{SVG_NS}}}linearGradient")
    assert len(gradients) == 1


def test_normalize_original_mode_leaves_the_scale_at_one(tmp_path):
    source = tmp_path / "in.svg"
    root = _root(viewBox="0 0 100 100")
    _child(root, id="a")
    ET.ElementTree(root).write(source)
    out = tmp_path / "out.svg"
    normalize_svg_to_canvas(source, out, (200, 200), "original", uniform_scale=4.0)
    group = next(node for node in ET.parse(out).getroot() if node.tag.endswith("g"))
    assert "scale(1.000000)" in group.attrib["transform"]


# --- metadata injection ---------------------------------------------------

def test_inject_metadata_writes_a_readable_payload(tmp_path):
    path = tmp_path / "icon.svg"
    root = _root(viewBox="0 0 10 10")
    _child(root, id="a")
    ET.ElementTree(root).write(path)

    inject_svg_metadata(path, {"stem": "heart", "tags": ["love"]})
    written = ET.parse(path).getroot()
    metadata = written.find(f"{{{SVG_NS}}}metadata")
    assert json.loads(metadata.text)["stem"] == "heart"
    assert _tags(written)[0] == "metadata"


def test_inject_metadata_replaces_a_previous_block(tmp_path):
    path = tmp_path / "icon.svg"
    root = _root(viewBox="0 0 10 10")
    ET.ElementTree(root).write(path)
    inject_svg_metadata(path, {"stem": "old"})
    inject_svg_metadata(path, {"stem": "new"})

    written = ET.parse(path).getroot()
    blocks = written.findall(f"{{{SVG_NS}}}metadata")
    assert len(blocks) == 1
    assert json.loads(blocks[0].text)["stem"] == "new"


def test_inject_metadata_ignores_unparseable_files(tmp_path):
    path = tmp_path / "broken.svg"
    path.write_text("not xml at all", encoding="utf-8")
    inject_svg_metadata(path, {"stem": "x"})  # must not raise
    assert path.read_text(encoding="utf-8") == "not xml at all"


# --- coordinate mapping ---------------------------------------------------

def test_raster_box_maps_into_viewbox_units():
    box = svg_box_from_raster_box(Box(100, 200, 50, 25), raster_w=1000, raster_h=1000, view_box=(0, 0, 100, 100))
    assert (box.x, box.y, box.w, box.h) == (10, 20, 5, 3)


def test_raster_box_mapping_honours_a_viewbox_offset():
    box = svg_box_from_raster_box(Box(0, 0, 10, 10), raster_w=100, raster_h=100, view_box=(50, 20, 100, 100))
    assert (box.x, box.y) == (50, 20)


# --- external tools -------------------------------------------------------

def test_require_binary_reports_the_missing_tool_by_name():
    with pytest.raises(RuntimeError, match="definitely-not-a-real-binary"):
        require_binary("definitely-not-a-real-binary")


def _fake_inkscape(stdout: str):
    """Patch out the binary lookup and the subprocess, returning ``stdout``."""
    return (
        patch("services.svg_ops.require_binary", return_value="inkscape"),
        patch("services.svg_ops.subprocess.run", return_value=SimpleNamespace(stdout=stdout)),
    )


def test_query_svg_boxes_parses_the_inkscape_csv(tmp_path):
    stdout = "shape_0001,10.4,20.9,30.1,40.2\nshape_0002,0,0,5,5\n"
    binary_patch, run_patch = _fake_inkscape(stdout)
    with binary_patch, run_patch:
        boxes = query_svg_boxes(tmp_path / "in.svg")
    # x/y floor, w/h ceil, so the box never under-covers the element.
    assert boxes["shape_0001"] == Box(10, 20, 31, 41)
    assert boxes["shape_0002"] == Box(0, 0, 5, 5)


def test_query_svg_boxes_skips_junk_lines(tmp_path):
    stdout = "shape_0001,1,1,2,2\nnot,enough\n\n"
    binary_patch, run_patch = _fake_inkscape(stdout)
    with binary_patch, run_patch:
        boxes = query_svg_boxes(tmp_path / "in.svg")
    assert list(boxes) == ["shape_0001"]


def test_query_svg_boxes_keeps_container_rows(tmp_path):
    """Layer and group rows are needed now: icons often live inside them."""
    stdout = "Layer_1,0,0,500,500\nshape_0001,1,1,2,2\n"
    binary_patch, run_patch = _fake_inkscape(stdout)
    with binary_patch, run_patch:
        boxes = query_svg_boxes(tmp_path / "in.svg")
    assert sorted(boxes) == ["Layer_1", "shape_0001"]


def test_query_svg_boxes_never_returns_a_zero_sized_box(tmp_path):
    binary_patch, run_patch = _fake_inkscape("shape_0001,5,5,0,0\n")
    with binary_patch, run_patch:
        boxes = query_svg_boxes(tmp_path / "in.svg")
    assert (boxes["shape_0001"].w, boxes["shape_0001"].h) == (1, 1)
