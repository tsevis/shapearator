"""Characterization tests for the geometry/detection primitives."""
from __future__ import annotations

import numpy as np
import pytest

from services.geometry import (
    Box,
    build_binary_mask,
    build_foreground_mask,
    compute_uniform_scale,
    box_contains_point,
    detect_icon_boxes,
    make_odd,
    sample_border_pixels,
    sort_boxes_reading_order,
    tighten_box,
)


# --- Box ------------------------------------------------------------------

def test_box_derived_edges_and_center():
    box = Box(10, 20, 30, 40)
    assert (box.x2, box.y2) == (40, 60)
    assert (box.cx, box.cy) == (25.0, 40.0)


def test_padded_grows_on_every_side_when_unbounded():
    assert Box(10, 20, 30, 40).padded(5) == Box(5, 15, 40, 50)


def test_padded_clamps_to_the_image_bounds():
    assert Box(2, 2, 10, 10).padded(5, limit_w=100, limit_h=100) == Box(0, 0, 17, 17)
    assert Box(90, 90, 10, 10).padded(5, limit_w=100, limit_h=100) == Box(85, 85, 15, 15)


def test_padded_returns_a_new_box():
    original = Box(10, 20, 30, 40)
    original.padded(5)
    assert original == Box(10, 20, 30, 40)


def test_union_covers_both_boxes():
    assert Box(0, 0, 10, 10).union(Box(20, 5, 10, 10)) == Box(0, 0, 30, 15)


def test_union_is_commutative():
    a, b = Box(0, 0, 10, 10), Box(20, 5, 10, 10)
    assert a.union(b) == b.union(a)


def test_box_contains_point_honours_padding():
    box = Box(0, 0, 10, 10)
    assert box_contains_point(box, 5, 5)
    assert not box_contains_point(box, 12, 5)
    assert box_contains_point(box, 12, 5, pad=3)


# --- helpers --------------------------------------------------------------

def test_make_odd_leaves_odd_values_alone():
    assert make_odd(3) == 3
    assert make_odd(4) == 5


@pytest.mark.parametrize(
    "sizes,canvas,expected",
    [
        ([(100, 50), (40, 80)], (512, 512), 5.12),  # bounded by the tallest, 512/100
        ([(1000, 1000)], (500, 500), 0.5),
        ([], (512, 512), 1.0),
    ],
)
def test_compute_uniform_scale_fits_the_largest_source(sizes, canvas, expected):
    assert compute_uniform_scale(sizes, canvas) == pytest.approx(expected)


# --- masks and detection --------------------------------------------------

def test_build_binary_mask_marks_dark_ink_as_foreground():
    gray = np.full((40, 40), 255, dtype=np.uint8)
    gray[10:20, 10:20] = 0
    mask = build_binary_mask(gray)
    assert mask[15, 15] == 255
    assert mask[2, 2] == 0


def test_sample_border_pixels_reads_all_four_edges():
    image = np.zeros((40, 40, 3), dtype=np.float32)
    image[:, :] = (10, 10, 10)
    image[0, 0] = (99, 99, 99)  # a corner pixel must appear in the sample
    sampled = sample_border_pixels(image, band=4)
    assert sampled.shape[1] == 3
    assert (sampled == 99).any()


def test_sample_border_pixels_shrinks_the_band_for_small_images():
    tiny = np.zeros((4, 4, 3), dtype=np.float32)
    assert sample_border_pixels(tiny, band=12).shape[1] == 3


def test_foreground_mask_finds_coloured_ink_on_coloured_paper():
    # Otsu on luminance alone can miss this; the LAB colour distance must catch it.
    bgr = np.zeros((80, 80, 3), dtype=np.uint8)
    bgr[:, :] = (200, 200, 120)  # tinted paper
    bgr[30:50, 30:50] = (60, 60, 200)  # red-ish ink at a similar lightness
    mask = build_foreground_mask(bgr)
    assert mask[40, 40] == 255
    assert mask[5, 5] == 0


def test_foreground_mask_matches_the_image_shape():
    bgr = np.full((30, 45, 3), 255, dtype=np.uint8)
    bgr[10:20, 10:20] = 0
    assert build_foreground_mask(bgr).shape == (30, 45)


def test_detect_icon_boxes_finds_separated_blobs():
    binary = np.zeros((100, 200), dtype=np.uint8)
    binary[20:40, 20:40] = 255
    binary[20:40, 150:170] = 255
    boxes = detect_icon_boxes(binary, min_area=50, merge_gap=5)
    assert len(boxes) == 2
    assert [box.x for box in boxes] == sorted(box.x for box in boxes)


def test_detect_icon_boxes_drops_blobs_under_min_area():
    binary = np.zeros((100, 100), dtype=np.uint8)
    binary[10:12, 10:12] = 255
    assert detect_icon_boxes(binary, min_area=5000, merge_gap=3) == []


def test_detect_icon_boxes_merges_marks_within_the_gap():
    binary = np.zeros((60, 60), dtype=np.uint8)
    binary[20:30, 10:20] = 255
    binary[20:30, 23:33] = 255  # 3px apart, inside a generous merge gap
    assert len(detect_icon_boxes(binary, min_area=10, merge_gap=15)) == 1


def test_tighten_box_shrinks_to_the_ink():
    binary = np.zeros((50, 50), dtype=np.uint8)
    binary[20:30, 15:25] = 255
    assert tighten_box(binary, Box(0, 0, 50, 50)) == Box(15, 20, 10, 10)


def test_tighten_box_returns_the_input_when_the_region_is_empty():
    binary = np.zeros((50, 50), dtype=np.uint8)
    box = Box(0, 0, 10, 10)
    assert tighten_box(binary, box) == box


def test_sort_boxes_reading_order_is_row_major():
    # Two rows of two, deliberately shuffled.
    boxes = [
        Box(200, 200, 40, 40),  # row 2, right
        Box(10, 10, 40, 40),    # row 1, left
        Box(200, 10, 40, 40),   # row 1, right
        Box(10, 200, 40, 40),   # row 2, left
    ]
    ordered = sort_boxes_reading_order(boxes)
    assert [(box.x, box.y) for box in ordered] == [(10, 10), (200, 10), (10, 200), (200, 200)]


def test_sort_boxes_reading_order_handles_empty_input():
    assert sort_boxes_reading_order([]) == []


def test_sort_boxes_reading_order_groups_slightly_misaligned_rows():
    # Hand-drawn sheets never align perfectly; small y jitter must stay one row.
    boxes = [Box(200, 18, 40, 40), Box(10, 10, 40, 40), Box(100, 25, 40, 40)]
    assert [box.x for box in sort_boxes_reading_order(boxes)] == [10, 100, 200]
