"""Turning SVG geometry into cubic Bézier subpaths.

PSD stores a path as knots with two control points each -- cubics and nothing
else. SVG has lines, quadratics, arcs, shorthand forms and relative variants,
so everything has to be converted before it can be written, and a conversion
that is subtly wrong produces a path that looks plausible and traces the wrong
shape.

These check the conversion against geometry whose answer is known by hand.
"""
from __future__ import annotations

import math

import pytest

from services import svg_paths


def ends(subpath) -> tuple[tuple[float, float], tuple[float, float]]:
    return subpath.segments[0].start, subpath.segments[-1].end


def close_to(a, b, tol=1e-6):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


# --- straight lines -------------------------------------------------------

def test_a_line_becomes_one_cubic():
    (sub,) = svg_paths.parse_path_data("M 0 0 L 10 0")
    assert len(sub.segments) == 1
    assert ends(sub) == ((0.0, 0.0), (10.0, 0.0))
    assert not sub.closed


def test_a_line_s_controls_lie_on_the_line():
    """A cubic with controls off the line is not the line it replaced."""
    (sub,) = svg_paths.parse_path_data("M 0 0 L 30 0")
    segment = sub.segments[0]
    assert segment.c1[1] == 0.0 and segment.c2[1] == 0.0
    assert 0.0 < segment.c1[0] < segment.c2[0] < 30.0


def test_relative_commands_accumulate():
    (sub,) = svg_paths.parse_path_data("M 5 5 l 10 0 l 0 10")
    assert ends(sub) == ((5.0, 5.0), (15.0, 15.0))


def test_horizontal_and_vertical_shorthands():
    (sub,) = svg_paths.parse_path_data("M 0 0 H 10 V 20 h -4 v -5")
    assert ends(sub) == ((0.0, 0.0), (6.0, 15.0))


def test_a_closed_path_says_so_and_returns_to_its_start():
    (sub,) = svg_paths.parse_path_data("M 0 0 L 10 0 L 10 10 Z")
    assert sub.closed
    assert close_to(sub.segments[-1].end, (0.0, 0.0))


def test_several_subpaths_come_back_separately():
    subs = svg_paths.parse_path_data("M 0 0 L 1 0 Z M 5 5 L 6 5 Z")
    assert len(subs) == 2
    assert subs[0].segments[0].start == (0.0, 0.0)
    assert subs[1].segments[0].start == (5.0, 5.0)


# --- curves ---------------------------------------------------------------

def test_a_cubic_is_carried_through_unchanged():
    (sub,) = svg_paths.parse_path_data("M 0 0 C 1 2 3 4 5 6")
    segment = sub.segments[0]
    assert (segment.c1, segment.c2, segment.end) == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))


def test_a_smooth_cubic_reflects_the_previous_control():
    """S takes its first control from the mirror of the last one."""
    (sub,) = svg_paths.parse_path_data("M 0 0 C 1 1 2 2 3 3 S 5 5 6 6")
    assert sub.segments[1].c1 == (4.0, 4.0)


def test_a_quadratic_becomes_the_cubic_that_draws_the_same_curve():
    """The standard elevation: controls sit two thirds along each leg."""
    (sub,) = svg_paths.parse_path_data("M 0 0 Q 3 3 6 0")
    segment = sub.segments[0]
    assert close_to(segment.c1, (2.0, 2.0))
    assert close_to(segment.c2, (4.0, 2.0))


def test_a_smooth_quadratic_reflects_the_previous_control():
    (sub,) = svg_paths.parse_path_data("M 0 0 Q 2 2 4 0 T 8 0")
    assert close_to(sub.segments[1].end, (8.0, 0.0))


def test_an_arc_becomes_curves_that_stay_on_the_circle():
    """A quarter circle of radius 10, sampled at the joins between cubics."""
    (sub,) = svg_paths.parse_path_data("M 10 0 A 10 10 0 0 1 0 10")
    assert close_to(sub.segments[0].start, (10.0, 0.0))
    assert close_to(sub.segments[-1].end, (0.0, 10.0))
    for segment in sub.segments:
        for point in (segment.start, segment.end):
            assert abs(math.hypot(*point) - 10.0) < 1e-6, point


def test_an_arc_with_no_distance_to_cover_is_dropped():
    """Per the SVG rules a zero-length arc is not an error, it is nothing."""
    (sub,) = svg_paths.parse_path_data("M 5 5 A 10 10 0 0 1 5 5 L 6 5")
    assert len(sub.segments) == 1


# --- the other shape elements ---------------------------------------------

def test_a_polygon_closes_itself():
    (sub,) = svg_paths.parse_points("0,0 10,0 10,10", closed=True)
    assert sub.closed
    assert len(sub.segments) == 3
    assert close_to(sub.segments[-1].end, (0.0, 0.0))


def test_a_polyline_does_not_close():
    (sub,) = svg_paths.parse_points("0,0 10,0 10,10", closed=False)
    assert not sub.closed
    assert len(sub.segments) == 2


def test_points_may_be_separated_however_the_file_likes():
    (spaced,) = svg_paths.parse_points("0 0 10 0 10 10", closed=True)
    (commas,) = svg_paths.parse_points("0,0 10,0 10,10", closed=True)
    assert [s.end for s in spaced.segments] == [s.end for s in commas.segments]


# --- placing the geometry -------------------------------------------------

def test_a_transform_moves_every_point():
    subs = svg_paths.parse_path_data("M 0 0 L 10 0")
    moved = svg_paths.transform(subs, scale=2.0, translate=(5.0, 7.0))
    assert ends(moved[0]) == ((5.0, 7.0), (25.0, 7.0))


def test_a_transform_moves_the_controls_too():
    """Controls left behind turn every curve inside out."""
    subs = svg_paths.parse_path_data("M 0 0 C 1 1 2 2 3 3")
    moved = svg_paths.transform(subs, scale=10.0, translate=(0.0, 0.0))
    assert moved[0].segments[0].c1 == (10.0, 10.0)


# --- refusals -------------------------------------------------------------

def test_geometry_that_never_starts_is_empty():
    assert svg_paths.parse_path_data("L 10 10") == ()


def test_nonsense_is_empty_rather_than_an_exception():
    """Path data comes from a file; a parser that raises stops a whole run."""
    assert svg_paths.parse_path_data("banana") == ()


def test_a_single_point_draws_nothing():
    assert svg_paths.parse_path_data("M 4 4") == ()


# --- the primitive shape elements -----------------------------------------

def test_a_rect_is_four_corners():
    (sub,) = svg_paths.parse_rect(10, 20, 30, 40)
    assert sub.closed
    corners = [s.start for s in sub.segments]
    assert corners == [(10.0, 20.0), (40.0, 20.0), (40.0, 60.0), (10.0, 60.0)]


def test_a_rect_with_no_area_draws_nothing():
    assert svg_paths.parse_rect(0, 0, 0, 10) == ()


def test_a_circle_stays_on_its_radius():
    (sub,) = svg_paths.parse_ellipse(50, 50, 20, 20)
    assert sub.closed
    for segment in sub.segments:
        assert abs(math.hypot(segment.start[0] - 50, segment.start[1] - 50) - 20) < 1e-6


def test_an_ellipse_uses_both_radii():
    (sub,) = svg_paths.parse_ellipse(0, 0, 30, 10)
    xs = [s.start[0] for s in sub.segments]
    ys = [s.start[1] for s in sub.segments]
    assert max(xs) == pytest.approx(30.0)
    assert max(ys) == pytest.approx(10.0)


def test_a_line_is_one_open_segment():
    (sub,) = svg_paths.parse_line(0, 0, 10, 10)
    assert not sub.closed
    assert len(sub.segments) == 1
