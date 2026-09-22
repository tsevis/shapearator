"""Vector geometry inside a PSD.

A path lives in a PSD as 26-byte records holding knots, with every coordinate
stored as a fraction of the document in 8.24 fixed point, vertical before
horizontal. Each of those is a chance to write something that parses and
traces the wrong shape, so these read the records back with `psd-tools` and
compare against coordinates worked out by hand.
"""
from __future__ import annotations

import pytest
from psd_tools import PSDImage

from services import psd_vector, svg_paths
from services.psd_writer import PsdLayer, write_psd

import numpy as np


def square_rgba(size=10):
    patch = np.zeros((size, size, 4), dtype=np.uint8)
    patch[:, :, 3] = 255
    return patch


def rectangle(x, y, w, h) -> tuple:
    return svg_paths.parse_points(
        f"{x},{y} {x + w},{y} {x + w},{y + h} {x},{y + h}", closed=True)


# --- the record encoding --------------------------------------------------

def test_every_record_is_twenty_six_bytes():
    """The format has no length field: a record is found by counting."""
    blob = psd_vector.path_records(rectangle(0, 0, 10, 10), (100, 100))
    assert len(blob) % 26 == 0
    assert len(blob) > 0


def test_a_closed_subpath_declares_how_many_knots_follow():
    blob = psd_vector.path_records(rectangle(0, 0, 10, 10), (100, 100))
    selector, count = psd_vector.read_record_header(blob, 0)
    assert selector == psd_vector.CLOSED_LENGTH
    assert count == 4, "a rectangle is four knots"


def test_an_open_subpath_is_marked_open():
    subpaths = svg_paths.parse_path_data("M 0 0 L 10 0 L 10 10")
    blob = psd_vector.path_records(subpaths, (100, 100))
    selector, _count = psd_vector.read_record_header(blob, 0)
    assert selector == psd_vector.OPEN_LENGTH


def test_a_coordinate_is_a_fraction_of_the_document():
    """Half way across a 200pt document is 0.5, not 100."""
    blob = psd_vector.path_records(rectangle(100, 50, 10, 10), (200, 100))
    knot = psd_vector.read_knot(blob, 1)
    assert knot.anchor == pytest.approx((0.5, 0.5), abs=1e-6)


def test_vertical_is_stored_before_horizontal():
    """Swapping them transposes every path, which looks like valid artwork."""
    blob = psd_vector.path_records(rectangle(160, 20, 4, 4), (200, 100))
    knot = psd_vector.read_knot(blob, 1)
    assert knot.vertical == pytest.approx(0.2, abs=1e-6)
    assert knot.horizontal == pytest.approx(0.8, abs=1e-6)


def test_a_knot_carries_both_of_its_controls():
    subpaths = svg_paths.parse_path_data("M 0 0 C 25 0 75 100 100 100 Z")
    blob = psd_vector.path_records(subpaths, (100, 100))
    knot = psd_vector.read_knot(blob, 1)
    assert knot.preceding != knot.anchor or knot.following != knot.anchor


def test_several_subpaths_each_get_their_own_length_record():
    subpaths = rectangle(0, 0, 10, 10) + rectangle(50, 50, 10, 10)
    blob = psd_vector.path_records(subpaths, (100, 100))
    headers = [psd_vector.read_record_header(blob, i)[0] for i in range(len(blob) // 26)]
    assert headers.count(psd_vector.CLOSED_LENGTH) == 2


def test_geometry_that_parsed_to_nothing_writes_nothing():
    assert psd_vector.path_records((), (100, 100)) == b""


# --- the Paths panel ------------------------------------------------------

def test_a_path_resource_reaches_the_paths_panel(tmp_path):
    resources = psd_vector.path_resources(
        [("brick_001", rectangle(10, 10, 30, 30))], (100, 100))
    out = tmp_path / "paths.psd"
    write_psd(out, [PsdLayer("brick_001", square_rgba())], (100, 100),
              extra_resources=resources)

    resources = PSDImage.open(out).image_resources
    named = {resources[key].name for key in resources}
    assert "brick_001" in named, named


def test_every_shape_gets_its_own_entry(tmp_path):
    shapes = [(f"brick_{i:03d}", rectangle(i * 5, 0, 4, 4)) for i in range(1, 6)]
    out = tmp_path / "paths.psd"
    write_psd(out, [PsdLayer("one", square_rgba())], (100, 100),
              extra_resources=psd_vector.path_resources(shapes, (100, 100)))

    resources = PSDImage.open(out).image_resources
    ids = [int(key) for key in resources
           if psd_vector.PATH_RESOURCE_FIRST <= int(key) <= psd_vector.PATH_RESOURCE_LAST]
    assert len(ids) == 5
    assert len(set(ids)) == 5, "two paths sharing an id would overwrite each other"
    assert {resources[key].name for key in resources} == {
        f"brick_{i:03d}" for i in range(1, 6)}


def test_more_paths_than_the_format_holds_are_dropped_not_corrupted(tmp_path):
    """Resource ids run 2000 to 2997; a run can exceed that."""
    shapes = [(f"s{i}", rectangle(0, 0, 2, 2)) for i in range(1100)]
    resources = psd_vector.path_resources(shapes, (100, 100))
    ids = psd_vector.resource_ids(resources)
    assert len(ids) == psd_vector.PATH_RESOURCE_LAST - psd_vector.PATH_RESOURCE_FIRST + 1
    assert max(ids) == psd_vector.PATH_RESOURCE_LAST
