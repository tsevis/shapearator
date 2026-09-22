"""SVG geometry as cubic Bézier subpaths.

PSD stores a path as knots carrying two control points each: cubics, and
nothing else. SVG has lines, quadratics, arcs, shorthand forms and a relative
variant of every command, so all of it has to be converted before any of it
can be written.

Nothing here raises. Path data arrives from a file someone else wrote, and a
parser that throws on one malformed `d` attribute takes a whole sheet with it;
geometry that cannot be read comes back empty and the shape simply has no
vector path, which the caller reports.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

Point = tuple[float, float]

#: A command letter, or a number in any of the spellings SVG allows.
_TOKENS = re.compile(r"[MmZzLlHhVvCcSsQqTtAa]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
#: How many numbers each command consumes per repetition.
_ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}
#: Quarter turn at most per cubic; beyond that the error becomes visible.
_MAX_ARC_SWEEP = math.pi / 2


@dataclass(frozen=True)
class CubicSegment:
    """One cubic, absolute, with both controls."""

    start: Point
    c1: Point
    c2: Point
    end: Point


@dataclass(frozen=True)
class Subpath:
    """A run of connected cubics, open or closed."""

    segments: tuple[CubicSegment, ...]
    closed: bool


def parse_path_data(data: str) -> tuple[Subpath, ...]:
    """Every subpath of a `d` attribute, as cubics in absolute coordinates."""
    try:
        return _parse(data)
    except Exception:
        return ()


def parse_points(points: str, closed: bool) -> tuple[Subpath, ...]:
    """A `<polygon>` or `<polyline>` points list as one subpath."""
    numbers = [float(value) for value in re.findall(r"[-+]?[\d.]+(?:[eE][-+]?\d+)?", points)]
    coordinates = list(zip(numbers[::2], numbers[1::2]))
    if len(coordinates) < 2:
        return ()
    segments = [_line(coordinates[i], coordinates[i + 1]) for i in range(len(coordinates) - 1)]
    if closed and coordinates[-1] != coordinates[0]:
        segments.append(_line(coordinates[-1], coordinates[0]))
    return (Subpath(tuple(segments), closed),)


def parse_rect(x: float, y: float, width: float, height: float) -> tuple[Subpath, ...]:
    """A `<rect>` as its four corners. Rounded corners are not honoured."""
    if width <= 0 or height <= 0:
        return ()
    return parse_points(
        f"{x},{y} {x + width},{y} {x + width},{y + height} {x},{y + height}", closed=True)


def parse_ellipse(cx: float, cy: float, rx: float, ry: float) -> tuple[Subpath, ...]:
    """A `<circle>` or `<ellipse>` as four cubics, the usual approximation."""
    if rx <= 0 or ry <= 0:
        return ()
    k = 0.5522847498307936        # (4/3)(sqrt(2) - 1), the circle constant
    right, top = (cx + rx, cy), (cx, cy - ry)
    left, bottom = (cx - rx, cy), (cx, cy + ry)
    segments = (
        CubicSegment(right, (cx + rx, cy + ry * k), (cx + rx * k, cy + ry), bottom),
        CubicSegment(bottom, (cx - rx * k, cy + ry), (cx - rx, cy + ry * k), left),
        CubicSegment(left, (cx - rx, cy - ry * k), (cx - rx * k, cy - ry), top),
        CubicSegment(top, (cx + rx * k, cy - ry), (cx + rx, cy - ry * k), right),
    )
    return (Subpath(segments, closed=True),)


def parse_line(x1: float, y1: float, x2: float, y2: float) -> tuple[Subpath, ...]:
    """A `<line>` as one open segment."""
    if (x1, y1) == (x2, y2):
        return ()
    return (Subpath((_line((x1, y1), (x2, y2)),), closed=False),)


def transform(subpaths: tuple[Subpath, ...], scale: float, translate: Point) -> tuple[Subpath, ...]:
    """Scale then translate every point, controls included.

    Controls left behind while anchors move turn each curve inside out, which
    is why this touches all four points of every segment.
    """
    def move(point: Point) -> Point:
        return (point[0] * scale + translate[0], point[1] * scale + translate[1])

    return tuple(
        Subpath(
            tuple(CubicSegment(move(s.start), move(s.c1), move(s.c2), move(s.end))
                  for s in sub.segments),
            sub.closed,
        )
        for sub in subpaths
    )


# --- the parser -----------------------------------------------------------

def _parse(data: str) -> tuple[Subpath, ...]:
    tokens = _TOKENS.findall(data or "")
    subpaths: list[Subpath] = []
    segments: list[CubicSegment] = []

    current: Point | None = None
    start: Point | None = None
    last_cubic_control: Point | None = None
    last_quadratic_control: Point | None = None
    command = ""
    index = 0

    def flush(closed: bool) -> None:
        nonlocal segments
        if segments:
            subpaths.append(Subpath(tuple(segments), closed))
        segments = []

    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
            if command in "Zz":
                if current is not None and start is not None and current != start:
                    segments.append(_line(current, start))
                flush(closed=True)
                current = start
                last_cubic_control = last_quadratic_control = None
                continue
        elif not command:
            return ()       # numbers before any command: nothing to draw

        key = command.upper()
        relative = command.islower()
        arity = _ARITY.get(key)
        if arity is None:
            return ()
        if index + arity > len(tokens):
            break
        values = [float(value) for value in tokens[index:index + arity]]
        index += arity

        if key == "M":
            if current is not None and segments:
                flush(closed=False)
            point = _absolute(values, current, relative)
            current = start = point
            last_cubic_control = last_quadratic_control = None
            # A repeated M draws lines, which is what a bare L would.
            command = "l" if relative else "L"
            continue

        if current is None:
            return ()

        if key == "L":
            end = _absolute(values, current, relative)
            segments.append(_line(current, end))
            last_cubic_control = last_quadratic_control = None
        elif key == "H":
            end = (current[0] + values[0] if relative else values[0], current[1])
            segments.append(_line(current, end))
            last_cubic_control = last_quadratic_control = None
        elif key == "V":
            end = (current[0], current[1] + values[0] if relative else values[0])
            segments.append(_line(current, end))
            last_cubic_control = last_quadratic_control = None
        elif key in ("C", "S"):
            if key == "C":
                c1 = _absolute(values[0:2], current, relative)
                c2 = _absolute(values[2:4], current, relative)
                end = _absolute(values[4:6], current, relative)
            else:
                c1 = _reflect(last_cubic_control, current)
                c2 = _absolute(values[0:2], current, relative)
                end = _absolute(values[2:4], current, relative)
            segments.append(CubicSegment(current, c1, c2, end))
            last_cubic_control, last_quadratic_control = c2, None
        elif key in ("Q", "T"):
            if key == "Q":
                control = _absolute(values[0:2], current, relative)
                end = _absolute(values[2:4], current, relative)
            else:
                control = _reflect(last_quadratic_control, current)
                end = _absolute(values[0:2], current, relative)
            segments.append(_from_quadratic(current, control, end))
            last_quadratic_control, last_cubic_control = control, None
        elif key == "A":
            end = _absolute(values[5:7], current, relative)
            segments.extend(_arc_to_cubics(current, values[0], values[1], values[2],
                                           bool(values[3]), bool(values[4]), end))
            last_cubic_control = last_quadratic_control = None
        else:
            return ()
        current = end

    flush(closed=False)
    return tuple(subpaths)


def _absolute(values: list[float], current: Point | None, relative: bool) -> Point:
    x, y = values[0], values[1]
    if relative and current is not None:
        return (current[0] + x, current[1] + y)
    return (x, y)


def _reflect(control: Point | None, current: Point) -> Point:
    """The mirror of the last control, or the point itself when there is none."""
    if control is None:
        return current
    return (2 * current[0] - control[0], 2 * current[1] - control[1])


def _line(start: Point, end: Point) -> CubicSegment:
    """A straight line as a cubic, with both controls on the line.

    Controls placed anywhere else describe a curve, not the line they replaced.
    """
    c1 = (start[0] + (end[0] - start[0]) / 3.0, start[1] + (end[1] - start[1]) / 3.0)
    c2 = (start[0] + 2 * (end[0] - start[0]) / 3.0, start[1] + 2 * (end[1] - start[1]) / 3.0)
    return CubicSegment(start, c1, c2, end)


def _from_quadratic(start: Point, control: Point, end: Point) -> CubicSegment:
    """Degree elevation: the controls sit two thirds along each leg."""
    c1 = (start[0] + 2.0 / 3.0 * (control[0] - start[0]),
          start[1] + 2.0 / 3.0 * (control[1] - start[1]))
    c2 = (end[0] + 2.0 / 3.0 * (control[0] - end[0]),
          end[1] + 2.0 / 3.0 * (control[1] - end[1]))
    return CubicSegment(start, c1, c2, end)


def _arc_to_cubics(
    start: Point, rx: float, ry: float, rotation: float,
    large_arc: bool, sweep: bool, end: Point,
) -> list[CubicSegment]:
    """An elliptical arc as cubics, following the SVG implementation notes.

    Split so no piece turns more than a quarter, which keeps the standard
    approximation's error below a thousandth of the radius.
    """
    if start == end or rx == 0 or ry == 0:
        return []                       # a zero-length arc is nothing, not an error

    rx, ry = abs(rx), abs(ry)
    phi = math.radians(rotation)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    dx2, dy2 = (start[0] - end[0]) / 2.0, (start[1] - end[1]) / 2.0
    x1 = cos_phi * dx2 + sin_phi * dy2
    y1 = -sin_phi * dx2 + cos_phi * dy2

    # Radii too small to reach are scaled up until they exactly can.
    oversize = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry)
    if oversize > 1:
        rx *= math.sqrt(oversize)
        ry *= math.sqrt(oversize)

    denominator = rx * rx * y1 * y1 + ry * ry * x1 * x1
    if denominator == 0:
        return []
    factor = math.sqrt(max(0.0, (rx * rx * ry * ry - denominator) / denominator))
    if large_arc == sweep:
        factor = -factor
    cx1, cy1 = factor * rx * y1 / ry, -factor * ry * x1 / rx
    cx = cos_phi * cx1 - sin_phi * cy1 + (start[0] + end[0]) / 2.0
    cy = sin_phi * cx1 + cos_phi * cy1 + (start[1] + end[1]) / 2.0

    theta = math.atan2((y1 - cy1) / ry, (x1 - cx1) / rx)
    theta_end = math.atan2((-y1 - cy1) / ry, (-x1 - cx1) / rx)
    delta = theta_end - theta
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi

    pieces = max(1, int(math.ceil(abs(delta) / _MAX_ARC_SWEEP)))
    step = delta / pieces
    alpha = 4.0 / 3.0 * math.tan(step / 4.0)

    def on_ellipse(angle: float) -> Point:
        x, y = rx * math.cos(angle), ry * math.sin(angle)
        return (cos_phi * x - sin_phi * y + cx, sin_phi * x + cos_phi * y + cy)

    def derivative(angle: float) -> Point:
        x, y = -rx * math.sin(angle), ry * math.cos(angle)
        return (cos_phi * x - sin_phi * y, sin_phi * x + cos_phi * y)

    segments: list[CubicSegment] = []
    angle = theta
    point = start
    for _ in range(pieces):
        nxt = angle + step
        end_point = on_ellipse(nxt)
        d1, d2 = derivative(angle), derivative(nxt)
        segments.append(CubicSegment(
            point,
            (point[0] + alpha * d1[0], point[1] + alpha * d1[1]),
            (end_point[0] - alpha * d2[0], end_point[1] - alpha * d2[1]),
            end_point,
        ))
        angle, point = nxt, end_point
    if segments:
        last = segments[-1]
        segments[-1] = CubicSegment(last.start, last.c1, last.c2, end)
    return segments
