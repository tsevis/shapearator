"""Extracted fragments must keep the definitions their shapes reference."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from services.geometry import Box
from services.svg_ops import (
    SVG_NS,
    XLINK_NS,
    build_svg_fragment,
    collect_referenced_ids,
    normalize_svg_to_canvas,
    resolve_definitions,
)


def _parse(markup: str) -> ET.Element:
    return ET.fromstring(markup.replace("@NS@", SVG_NS).replace("@XLINK@", XLINK_NS))


def _fragment(root: ET.Element, child_ids: list[str], tmp_path, padding: int = 0) -> ET.Element:
    children = [node for node in root if node.attrib.get("id") in child_ids]
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "fragment.svg"
    build_svg_fragment(root, children, Box(0, 0, 100, 100), padding, out)
    return ET.parse(out).getroot()


def _ids_in_defs(fragment: ET.Element) -> set[str]:
    return {
        node.attrib["id"]
        for defs in fragment.findall(f"{{{SVG_NS}}}defs")
        for node in defs.iter()
        if "id" in node.attrib
    }


def _tags(element: ET.Element) -> list[str]:
    return [child.tag.split("}")[-1] for child in element]


# --- reference scanning ---------------------------------------------------

@pytest.mark.parametrize(
    "attrib,expected",
    [
        ('fill="url(#grad)"', {"grad"}),
        ("fill=\"url('#grad')\"", {"grad"}),
        ('clip-path="url(#clip)"', {"clip"}),
        ('mask="url(#m)"', {"m"}),
        ('filter="url(#f)"', {"f"}),
        ('marker-end="url(#arrow)"', {"arrow"}),
        ('style="fill:url(#grad);stroke:url(#s)"', {"grad", "s"}),
        ('fill="#ff0000"', set()),  # a colour, not a reference
        ('fill="none"', set()),
    ],
)
def test_reference_scanning_finds_url_targets(attrib, expected):
    element = _parse(f'<path xmlns="@NS@" {attrib}/>')
    assert collect_referenced_ids([element]) == expected


def test_reference_scanning_finds_href_targets():
    element = _parse('<use xmlns="@NS@" href="#symbol-a"/>')
    assert collect_referenced_ids([element]) == {"symbol-a"}


def test_reference_scanning_finds_xlink_href_targets():
    element = _parse('<use xmlns="@NS@" xmlns:xlink="@XLINK@" xlink:href="#symbol-a"/>')
    assert collect_referenced_ids([element]) == {"symbol-a"}


def test_reference_scanning_ignores_external_urls():
    element = _parse('<image xmlns="@NS@" href="https://example.com/a.png"/>')
    assert collect_referenced_ids([element]) == set()


def test_reference_scanning_descends_into_children():
    group = _parse('<g xmlns="@NS@"><path fill="url(#deep)"/></g>')
    assert collect_referenced_ids([group]) == {"deep"}


# --- definition resolution ------------------------------------------------

BASIC = """
<svg xmlns="@NS@" viewBox="0 0 100 100">
  <defs>
    <linearGradient id="used"/>
    <linearGradient id="unused"/>
  </defs>
  <path id="a" fill="url(#used)"/>
  <path id="b" fill="#000"/>
</svg>
"""


def test_only_referenced_definitions_are_pulled_in():
    root = _parse(BASIC)
    selected = [node for node in root if node.attrib.get("id") == "a"]
    resolved = resolve_definitions(root, selected)
    assert [node.attrib["id"] for node in resolved] == ["used"]


def test_a_shape_referencing_nothing_pulls_nothing():
    root = _parse(BASIC)
    selected = [node for node in root if node.attrib.get("id") == "b"]
    assert resolve_definitions(root, selected) == []


def test_definitions_resolve_transitively():
    root = _parse("""
    <svg xmlns="@NS@" xmlns:xlink="@XLINK@" viewBox="0 0 100 100">
      <defs>
        <linearGradient id="base"/>
        <linearGradient id="derived" xlink:href="#base"/>
        <linearGradient id="orphan"/>
      </defs>
      <path id="a" fill="url(#derived)"/>
    </svg>
    """)
    selected = [node for node in root if node.attrib.get("id") == "a"]
    assert {node.attrib["id"] for node in resolve_definitions(root, selected)} == {"derived", "base"}


def test_definitions_referenced_from_inside_a_definition_are_included():
    root = _parse("""
    <svg xmlns="@NS@" viewBox="0 0 100 100">
      <defs>
        <linearGradient id="grad"/>
        <clipPath id="clip"><rect fill="url(#grad)"/></clipPath>
      </defs>
      <path id="a" clip-path="url(#clip)"/>
    </svg>
    """)
    selected = [node for node in root if node.attrib.get("id") == "a"]
    assert {node.attrib["id"] for node in resolve_definitions(root, selected)} == {"clip", "grad"}


def test_definitions_outside_a_defs_wrapper_are_found():
    # Plenty of exporters emit gradients as bare root children.
    root = _parse("""
    <svg xmlns="@NS@" viewBox="0 0 100 100">
      <linearGradient id="loose"/>
      <path id="a" fill="url(#loose)"/>
    </svg>
    """)
    selected = [node for node in root if node.attrib.get("id") == "a"]
    assert [node.attrib["id"] for node in resolve_definitions(root, selected)] == ["loose"]


def test_definitions_nested_below_the_root_are_found():
    # Illustrator and Figma both emit <defs> inside a wrapping <g>.
    root = _parse("""
    <svg xmlns="@NS@" viewBox="0 0 100 100">
      <g id="layer"><defs><linearGradient id="deep"/></defs></g>
      <path id="a" fill="url(#deep)"/>
    </svg>
    """)
    selected = [node for node in root if node.attrib.get("id") == "a"]
    assert [node.attrib["id"] for node in resolve_definitions(root, selected)] == ["deep"]


def test_a_selected_drawable_is_not_duplicated_into_defs():
    root = _parse("""
    <svg xmlns="@NS@" viewBox="0 0 100 100">
      <path id="a"/>
      <use id="b" href="#a"/>
    </svg>
    """)
    selected = [node for node in root if node.attrib.get("id") in {"a", "b"}]
    assert resolve_definitions(root, selected) == []


def test_a_use_target_outside_the_selection_is_pulled_in():
    root = _parse("""
    <svg xmlns="@NS@" viewBox="0 0 100 100">
      <defs><symbol id="star"><path/></symbol></defs>
      <use id="b" href="#star"/>
    </svg>
    """)
    selected = [node for node in root if node.attrib.get("id") == "b"]
    assert [node.attrib["id"] for node in resolve_definitions(root, selected)] == ["star"]


def test_a_dangling_reference_is_ignored():
    root = _parse("""
    <svg xmlns="@NS@" viewBox="0 0 100 100">
      <path id="a" fill="url(#missing)"/>
    </svg>
    """)
    selected = [node for node in root if node.attrib.get("id") == "a"]
    assert resolve_definitions(root, selected) == []  # must not raise


def test_a_self_referencing_definition_terminates():
    root = _parse("""
    <svg xmlns="@NS@" xmlns:xlink="@XLINK@" viewBox="0 0 100 100">
      <defs>
        <linearGradient id="loop" xlink:href="#loop"/>
        <linearGradient id="ping" xlink:href="#pong"/>
        <linearGradient id="pong" xlink:href="#ping"/>
      </defs>
      <path id="a" fill="url(#loop)" stroke="url(#ping)"/>
    </svg>
    """)
    selected = [node for node in root if node.attrib.get("id") == "a"]
    assert {node.attrib["id"] for node in resolve_definitions(root, selected)} == {"loop", "ping", "pong"}


# --- what actually lands in the fragment ---------------------------------

def test_fragment_carries_the_referenced_gradient(tmp_path):
    fragment = _fragment(_parse(BASIC), ["a"], tmp_path)
    assert _ids_in_defs(fragment) == {"used"}
    assert _tags(fragment)[0] == "defs"  # definitions precede the shapes


def test_fragment_omits_definitions_it_does_not_use(tmp_path):
    fragment = _fragment(_parse(BASIC), ["b"], tmp_path)
    assert fragment.find(f"{{{SVG_NS}}}defs") is None


def test_fragment_keeps_stylesheet_rules(tmp_path):
    root = _parse("""
    <svg xmlns="@NS@" viewBox="0 0 100 100">
      <style>.ink { fill: #123456; }</style>
      <path id="a" class="ink"/>
    </svg>
    """)
    fragment = _fragment(root, ["a"], tmp_path)
    style = fragment.find(f"{{{SVG_NS}}}style")
    assert style is not None and "#123456" in style.text


def test_stylesheets_are_kept_even_when_no_selector_matches(tmp_path):
    # CSS selectors cannot be resolved without a full cascade; keeping the
    # sheet is the only safe option.
    root = _parse("""
    <svg xmlns="@NS@" viewBox="0 0 100 100">
      <style>.other { fill: red; }</style>
      <path id="a"/>
    </svg>
    """)
    assert _fragment(root, ["a"], tmp_path).find(f"{{{SVG_NS}}}style") is not None


def test_definitions_referenced_only_from_a_stylesheet_are_kept(tmp_path):
    root = _parse("""
    <svg xmlns="@NS@" viewBox="0 0 100 100">
      <style>.ink { fill: url(#grad); }</style>
      <defs><linearGradient id="grad"/></defs>
      <path id="a" class="ink"/>
    </svg>
    """)
    assert _ids_in_defs(_fragment(root, ["a"], tmp_path)) == {"grad"}


def test_fragment_definitions_are_not_translated(tmp_path):
    """Only the shapes move; a userSpaceOnUse gradient follows its shape."""
    root = _parse(BASIC)
    children = [node for node in root if node.attrib.get("id") == "a"]
    out = tmp_path / "fragment.svg"
    build_svg_fragment(root, children, Box(40, 30, 20, 20), padding=5, output_path=out)
    fragment = ET.parse(out).getroot()

    defs = fragment.find(f"{{{SVG_NS}}}defs")
    assert "transform" not in defs.attrib
    assert all("transform" not in node.attrib for node in defs.iter() if node is not defs)
    shape = next(node for node in fragment if node.tag.endswith("path"))
    assert shape.attrib["transform"] == "translate(-35 -25)"


def test_fragment_does_not_mutate_the_source_definitions(tmp_path):
    root = _parse(BASIC)
    _fragment(root, ["a"], tmp_path)
    source_defs = root.find(f"{{{SVG_NS}}}defs")
    assert [node.attrib["id"] for node in source_defs] == ["used", "unused"]


def test_two_fragments_each_get_their_own_definitions(tmp_path):
    root = _parse("""
    <svg xmlns="@NS@" viewBox="0 0 100 100">
      <defs><linearGradient id="g1"/><linearGradient id="g2"/></defs>
      <path id="a" fill="url(#g1)"/>
      <path id="b" fill="url(#g2)"/>
    </svg>
    """)
    assert _ids_in_defs(_fragment(root, ["a"], tmp_path / "one")) == {"g1"}
    assert _ids_in_defs(_fragment(root, ["b"], tmp_path / "two")) == {"g2"}


# --- survival through canvas normalization -------------------------------

def test_definitions_survive_normalization(tmp_path):
    source = tmp_path / "frag.svg"
    build_svg_fragment(
        _parse(BASIC),
        [node for node in _parse(BASIC) if node.attrib.get("id") == "a"],
        Box(0, 0, 100, 100),
        0,
        source,
    )
    out = tmp_path / "final.svg"
    normalize_svg_to_canvas(source, out, (512, 512), "individual_fit", uniform_scale=1.0)

    final = ET.parse(out).getroot()
    assert _ids_in_defs(final) == {"used"}
    # Definitions stay outside the scaling group so they are not transformed twice.
    group = next(node for node in final if node.tag.endswith("g"))
    assert group.find(f"{{{SVG_NS}}}defs") is None


def test_stylesheets_survive_normalization_at_the_root(tmp_path):
    source = tmp_path / "frag.svg"
    source.write_text(
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 10 10">'
        f'<style>.ink {{ fill: red; }}</style><path id="a"/></svg>',
        encoding="utf-8",
    )
    out = tmp_path / "final.svg"
    normalize_svg_to_canvas(source, out, (512, 512), "original", uniform_scale=1.0)

    final = ET.parse(out).getroot()
    assert final.find(f"{{{SVG_NS}}}style") is not None
    group = next(node for node in final if node.tag.endswith("g"))
    assert group.find(f"{{{SVG_NS}}}style") is None
