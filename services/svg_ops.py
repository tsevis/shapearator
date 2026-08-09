"""SVG parsing, fragment building, canvas normalization, and tool invocation.

Everything that reads or writes SVG lives here, including the Inkscape and
potrace subprocess calls the vector path depends on.
"""
from __future__ import annotations

import base64
import json
import math
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import Box

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

# fill="url(#grad)", clip-path="url('#clip')", style="filter:url(#f)" -- one
# pattern covers every paint/clip/mask/marker attribute without listing them.
_URL_REFERENCE = re.compile(r"url\(\s*['\"]?#([^)'\"\s]+)")
# A bare "#name" is only a reference in an href, never in a paint attribute --
# otherwise fill="#ff0000" would read as a reference to an element "ff0000".
_HREF_ATTRIBUTES = frozenset({"href", f"{{{XLINK_NS}}}href"})


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise RuntimeError(f"Required binary '{name}' was not found on PATH.")
    return binary


# --- reading source documents --------------------------------------------

def parse_svg_length(value: str) -> float:
    cleaned = value.strip().replace("px", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def get_svg_canvas_size(root: ET.Element) -> tuple[float, float]:
    view_box = root.attrib.get("viewBox")
    if view_box:
        parts = [float(part) for part in view_box.replace(",", " ").split()]
        if len(parts) == 4:
            return parts[2], parts[3]
    width = parse_svg_length(root.attrib.get("width", "0"))
    height = parse_svg_length(root.attrib.get("height", "0"))
    return max(1.0, width), max(1.0, height)


def parse_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    view_box = root.attrib.get("viewBox")
    if not view_box:
        raise RuntimeError("SVG must include a viewBox.")
    parts = [float(part) for part in view_box.replace(",", " ").split()]
    if len(parts) != 4:
        raise RuntimeError(f"Unexpected viewBox: {view_box}")
    return tuple(parts)  # type: ignore[return-value]


def ensure_element_ids(root: ET.Element) -> None:
    """Give every drawable top-level child a stable id for box queries.

    Generated ids are checked against the ids already present anywhere in the
    document: a source file that happens to use the ``shape_NNNN`` pattern
    would otherwise produce duplicates, and the box query returns a dict keyed
    by id, so a collision silently maps an element to the wrong bounds.
    """
    taken = {element.attrib["id"] for element in root.iter() if "id" in element.attrib}
    counter = 1
    for child in root:
        if not isinstance(child.tag, str):
            continue
        if child.tag.endswith("defs"):
            continue
        if "id" in child.attrib:
            continue
        while f"shape_{counter:04d}" in taken:
            counter += 1
        generated = f"shape_{counter:04d}"
        child.set("id", generated)
        taken.add(generated)
        counter += 1


def query_svg_boxes(svg_path: Path) -> dict[str, Box]:
    """Ask Inkscape for the rendered bounds of every identified element."""
    require_binary("inkscape")
    result = subprocess.run(["inkscape", "--query-all", str(svg_path)], check=True, capture_output=True, text=True)
    boxes: dict[str, Box] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        shape_id, x, y, w, h = parts
        if shape_id == "Layer_1":
            continue
        boxes[shape_id] = Box(
            int(math.floor(float(x))),
            int(math.floor(float(y))),
            max(1, int(math.ceil(float(w)))),
            max(1, int(math.ceil(float(h)))),
        )
    return boxes


def svg_box_from_raster_box(
    raster_box: Box, raster_w: int, raster_h: int, view_box: tuple[float, float, float, float]
) -> Box:
    """Map a box measured on the rasterized preview back into viewBox units."""
    vb_x, vb_y, vb_w, vb_h = view_box
    scale_x = vb_w / raster_w
    scale_y = vb_h / raster_h
    x = vb_x + raster_box.x * scale_x
    y = vb_y + raster_box.y * scale_y
    w = raster_box.w * scale_x
    h = raster_box.h * scale_y
    return Box(int(math.floor(x)), int(math.floor(y)), max(1, int(math.ceil(w))), max(1, int(math.ceil(h))))


# --- definitions ----------------------------------------------------------

def collect_referenced_ids(elements: list[ET.Element]) -> set[str]:
    """Return every document id referenced by ``elements`` or their children.

    Scanning attribute *values* rather than a list of known attribute names
    means paint servers, clip paths, masks, filters, markers, and `<use>`
    targets are all caught, including inside a ``style`` attribute.
    """
    found: set[str] = set()
    for element in elements:
        for node in element.iter():
            for name, value in node.attrib.items():
                found.update(_URL_REFERENCE.findall(value))
                if name in _HREF_ATTRIBUTES and value.strip().startswith("#"):
                    found.add(value.strip()[1:])
            if _is_tag(node, "style") and node.text:
                found.update(_URL_REFERENCE.findall(node.text))
    return found


def collect_style_elements(root: ET.Element) -> list[ET.Element]:
    """Return every <style> element in the document, in document order.

    Stylesheets are copied wholesale: resolving which rules apply would mean
    implementing the CSS cascade, and dropping a rule silently restyles the
    icon. A few unused rules cost bytes; a missing one costs correctness.
    """
    return [node for node in root.iter() if _is_tag(node, "style")]


def resolve_definitions(root: ET.Element, selected: list[ET.Element]) -> list[ET.Element]:
    """Return the definition elements ``selected`` needs, transitively.

    Follows chains -- a shape referencing a clip path whose contents reference
    a gradient pulls in both -- and tolerates dangling and circular references.
    Elements already present in ``selected`` are skipped so a `<use>` pointing
    at a sibling shape does not duplicate it.
    """
    index = {node.attrib["id"]: node for node in root.iter() if "id" in node.attrib}
    already_present = {node.attrib["id"] for node in selected if "id" in node.attrib}

    pending = collect_referenced_ids(selected) | collect_referenced_ids(collect_style_elements(root))
    resolved: dict[str, ET.Element] = {}
    while pending:
        target_id = pending.pop()
        if target_id in resolved or target_id in already_present:
            continue
        definition = index.get(target_id)
        if definition is None:  # dangling reference: nothing to copy
            continue
        resolved[target_id] = definition
        pending |= collect_referenced_ids([definition])

    document_order = [node.attrib["id"] for node in root.iter() if node.attrib.get("id") in resolved]
    return [resolved[node_id] for node_id in dict.fromkeys(document_order)]


def _is_tag(element: ET.Element, name: str) -> bool:
    return isinstance(element.tag, str) and element.tag.split("}")[-1] == name


# --- writing fragments ----------------------------------------------------

def build_svg_fragment(
    source_root: ET.Element, children: list[ET.Element], bounds: Box, padding: int, output_path: Path
) -> None:
    """Write the selected children into their own document, re-origined.

    The fragment carries any definitions and stylesheets the children depend
    on; without them an extracted icon loses its gradients, clipping, masks,
    filters, and symbols, and can render as blank or invalid.
    """
    fragment = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "viewBox": f"0 0 {bounds.w + padding * 2} {bounds.h + padding * 2}",
        },
    )
    if "style" in source_root.attrib:
        fragment.set("style", source_root.attrib["style"])

    for style in collect_style_elements(source_root):
        fragment.append(deepcopy(style))

    definitions = resolve_definitions(source_root, children)
    if definitions:
        # Definitions are deliberately not translated: a userSpaceOnUse
        # gradient or clip path is resolved in the referencing shape's space,
        # so it already moves with the transform applied below.
        defs = ET.SubElement(fragment, f"{{{SVG_NS}}}defs")
        for definition in definitions:
            defs.append(deepcopy(definition))

    dx = padding - bounds.x
    dy = padding - bounds.y
    for child in children:
        node = deepcopy(child)
        existing_transform = node.attrib.get("transform", "").strip()
        translate = f"translate({dx} {dy})"
        node.set("transform", f"{translate} {existing_transform}".strip())
        fragment.append(node)
    ET.ElementTree(fragment).write(output_path, encoding="utf-8", xml_declaration=True)


def normalize_svg_to_canvas(
    source_svg_path: Path,
    output_svg_path: Path,
    canvas_size: tuple[int, int],
    mode: str,
    uniform_scale: float,
) -> None:
    """Re-wrap a fragment onto the export canvas, centred and scaled.

    Definitions stay at the root, outside the scaling group, so referenced
    gradients and clip paths are not transformed twice.
    """
    tree = ET.parse(source_svg_path)
    root = tree.getroot()
    source_w, source_h = get_svg_canvas_size(root)
    target_w, target_h = canvas_size

    scale = 1.0
    if mode == "uniform_to_largest":
        scale = uniform_scale
    elif mode == "individual_fit":
        scale = min(target_w / max(1.0, source_w), target_h / max(1.0, source_h))

    # Definitions and stylesheets stay at the root; only drawables are scaled.
    root_level_children = []
    drawable_children = []
    for child in root:
        if _is_tag(child, "defs") or _is_tag(child, "style"):
            root_level_children.append(deepcopy(child))
        else:
            drawable_children.append(deepcopy(child))

    canvas_root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": str(target_w),
            "height": str(target_h),
            "viewBox": f"0 0 {target_w} {target_h}",
        },
    )
    for child in root_level_children:
        canvas_root.append(child)

    scaled_w = source_w * scale
    scaled_h = source_h * scale
    offset_x = (target_w - scaled_w) / 2.0
    offset_y = (target_h - scaled_h) / 2.0
    group = ET.SubElement(
        canvas_root,
        f"{{{SVG_NS}}}g",
        {"transform": f"translate({offset_x:.4f} {offset_y:.4f}) scale({scale:.6f})"},
    )
    for child in drawable_children:
        group.append(child)
    ET.ElementTree(canvas_root).write(output_svg_path, encoding="utf-8", xml_declaration=True)


def wrap_png_in_svg(png_path: Path, svg_path: Path, canvas_size: tuple[int, int]) -> None:
    """Embed a bitmap as a data URI, for icons that tracing would ruin."""
    target_w, target_h = canvas_size
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": str(target_w),
            "height": str(target_h),
            "viewBox": f"0 0 {target_w} {target_h}",
        },
    )
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}image",
        {
            "width": str(target_w),
            "height": str(target_h),
            "href": f"data:image/png;base64,{encoded}",
        },
    )
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)


def inject_svg_metadata(svg_path: Path, payload: dict) -> None:
    """Replace the <metadata> block with ``payload``; a no-op if unparseable."""
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except ET.ParseError:
        return
    for child in list(root):
        if isinstance(child.tag, str) and child.tag.endswith("metadata"):
            root.remove(child)
    metadata = ET.Element(f"{{{SVG_NS}}}metadata")
    metadata.text = json.dumps(payload, ensure_ascii=True)
    root.insert(0, metadata)
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)


# --- external rendering ---------------------------------------------------

def vectorize_png_crop(png_path: Path, svg_path: Path) -> None:
    """Trace the alpha silhouette into real vector paths via potrace."""
    require_binary("potrace")
    with Image.open(png_path) as image:
        alpha = np.array(image.getchannel("A"))
        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        bmp_path = svg_path.with_suffix(".bmp")
        Image.fromarray(mask, mode="L").save(bmp_path)
    try:
        subprocess.run(["potrace", "-s", str(bmp_path), "-o", str(svg_path)], check=True, capture_output=True, text=True)
    finally:
        bmp_path.unlink(missing_ok=True)


def export_svg_to_png(svg_path: Path, output_path: Path) -> None:
    require_binary("inkscape")
    subprocess.run(
        ["inkscape", str(svg_path), "--export-type=png", f"--export-filename={output_path}"],
        check=True,
        capture_output=True,
        text=True,
    )


def render_svg_to_png(svg_path: Path, output_path: Path, width: int = 2000) -> None:
    require_binary("inkscape")
    subprocess.run(
        [
            "inkscape",
            str(svg_path),
            "--export-type=png",
            f"--export-filename={output_path}",
            f"--export-width={width}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
