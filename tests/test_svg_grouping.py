"""Choosing which SVG elements are icons, using the artwork's own structure."""
from __future__ import annotations

import xml.etree.ElementTree as ET

from services.geometry import Box
from services.svg_ops import (
    SVG_NS,
    ancestor_transform,
    build_parent_map,
    build_svg_fragment,
    ensure_element_ids,
    find_icon_elements,
    is_grouped_artwork,
)


def _parse(markup: str) -> ET.Element:
    return ET.fromstring(markup.replace("@NS@", SVG_NS))


def _boxes(root: ET.Element) -> dict[str, Box]:
    """Every identified element gets a nominal box, as Inkscape would report."""
    return {e.attrib["id"]: Box(0, 0, 10, 10) for e in root.iter() if "id" in e.attrib}


def _ids(elements: list[ET.Element]) -> list[str]:
    return [e.attrib["id"] for e in elements]


# --- picking the icon level ----------------------------------------------

def test_flat_sheet_uses_root_children():
    root = _parse('<svg xmlns="@NS@"><path id="a"/><path id="b"/><path id="c"/></svg>')
    assert _ids(find_icon_elements(root, _boxes(root))) == ["a", "b", "c"]


def test_a_single_wrapping_layer_is_unwrapped():
    """Illustrator and Figma wrap everything in one layer group."""
    root = _parse("""
    <svg xmlns="@NS@">
      <g id="Layer_1"><g id="one"/><g id="two"/><g id="three"/></g>
    </svg>
    """)
    assert _ids(find_icon_elements(root, _boxes(root))) == ["one", "two", "three"]


def test_several_nested_wrappers_are_unwrapped():
    root = _parse("""
    <svg xmlns="@NS@">
      <g id="outer"><g id="inner"><g id="a"/><g id="b"/></g></g>
    </svg>
    """)
    assert _ids(find_icon_elements(root, _boxes(root))) == ["a", "b"]


def test_a_lone_icon_stays_the_icon():
    """Unwrapping must stop at the artwork, not descend into its pieces."""
    root = _parse('<svg xmlns="@NS@"><g id="only"><path id="p1"/></g></svg>')
    assert _ids(find_icon_elements(root, _boxes(root))) == ["p1"]


def test_elements_without_measured_bounds_are_ignored():
    root = _parse('<svg xmlns="@NS@"><path id="a"/><path id="ghost"/></svg>')
    boxes = {"a": Box(0, 0, 10, 10)}
    assert _ids(find_icon_elements(root, boxes)) == ["a"]


def test_definitions_are_never_icons():
    root = _parse("""
    <svg xmlns="@NS@">
      <defs id="d"><linearGradient id="grad"/></defs>
      <style id="s">.x{fill:red}</style>
      <path id="a"/><path id="b"/>
    </svg>
    """)
    assert _ids(find_icon_elements(root, _boxes(root))) == ["a", "b"]


# --- grouped vs loose artwork --------------------------------------------

def test_groups_are_treated_as_authored_icons():
    root = _parse('<svg xmlns="@NS@"><g id="a"/><g id="b"/></svg>')
    assert is_grouped_artwork(list(root)) is True


def test_bare_shapes_are_not_treated_as_authored_icons():
    """A hand-drawn icon is several strokes; structure would shatter it."""
    root = _parse('<svg xmlns="@NS@"><path id="a"/><path id="b"/><path id="c"/></svg>')
    assert is_grouped_artwork(list(root)) is False


def test_use_elements_count_as_containers():
    root = _parse('<svg xmlns="@NS@"><use id="a"/><use id="b"/></svg>')
    assert is_grouped_artwork(list(root)) is True


def test_a_mixed_sheet_follows_the_majority():
    mostly_groups = _parse('<svg xmlns="@NS@"><g id="a"/><g id="b"/><path id="c"/></svg>')
    mostly_paths = _parse('<svg xmlns="@NS@"><g id="a"/><path id="b"/><path id="c"/></svg>')
    assert is_grouped_artwork(list(mostly_groups)) is True
    assert is_grouped_artwork(list(mostly_paths)) is False


def test_an_empty_level_is_not_grouped_artwork():
    assert is_grouped_artwork([]) is False


# --- ids at every depth ---------------------------------------------------

def test_ids_are_assigned_at_every_depth():
    root = _parse('<svg xmlns="@NS@"><g><g><path/></g></g></svg>')
    ensure_element_ids(root)
    assert all("id" in e.attrib for e in root.iter() if e is not root)


def test_deep_id_assignment_avoids_collisions():
    root = _parse('<svg xmlns="@NS@"><g><path/></g><path id="shape_0001"/></svg>')
    ensure_element_ids(root)
    ids = [e.attrib["id"] for e in root.iter() if "id" in e.attrib]
    assert len(set(ids)) == len(ids)


def test_definitions_are_left_unidentified():
    root = _parse('<svg xmlns="@NS@"><defs><linearGradient/></defs><path/></svg>')
    ensure_element_ids(root)
    defs = root.find(f"{{{SVG_NS}}}defs")
    assert "id" not in defs.attrib


# --- ancestor transforms --------------------------------------------------

def test_ancestor_transform_is_empty_at_the_root():
    root = _parse('<svg xmlns="@NS@"><path id="a"/></svg>')
    parents = build_parent_map(root)
    assert ancestor_transform(root[0], parents) == ""


def test_ancestor_transform_collects_the_chain_outermost_first():
    root = _parse("""
    <svg xmlns="@NS@">
      <g transform="translate(10 20)"><g transform="scale(2)"><path id="a"/></g></g>
    </svg>
    """)
    parents = build_parent_map(root)
    target = next(e for e in root.iter() if e.attrib.get("id") == "a")
    assert ancestor_transform(target, parents) == "translate(10 20) scale(2)"


def test_a_lifted_icon_keeps_its_layer_transform(tmp_path):
    """Bounds are measured in document space, so the transform must come along."""
    root = _parse("""
    <svg xmlns="@NS@" viewBox="0 0 100 100">
      <g id="layer" transform="translate(40 25) scale(0.8)"><g id="icon"><path id="p"/></g></g>
    </svg>
    """)
    parents = build_parent_map(root)
    icon = next(e for e in root.iter() if e.attrib.get("id") == "icon")
    out = tmp_path / "fragment.svg"
    build_svg_fragment(root, [icon], Box(0, 0, 10, 10), 0, out, parents)

    node = next(n for n in ET.parse(out).getroot() if n.attrib.get("id") == "icon")
    assert node.attrib["transform"] == "translate(0 0) translate(40 25) scale(0.8)"


def test_an_icon_at_the_root_gains_no_extra_transform(tmp_path):
    root = _parse('<svg xmlns="@NS@" viewBox="0 0 100 100"><path id="a" transform="rotate(45)"/></svg>')
    parents = build_parent_map(root)
    out = tmp_path / "fragment.svg"
    build_svg_fragment(root, [root[0]], Box(0, 0, 10, 10), 0, out, parents)

    node = next(n for n in ET.parse(out).getroot() if n.attrib.get("id") == "a")
    assert node.attrib["transform"] == "translate(0 0) rotate(45)"
