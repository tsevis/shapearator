"""Vector geometry inside a PSD: path records, and what holds them.

A path is stored as a run of 26-byte records. There is no length field
anywhere -- a reader finds the records by counting -- so a record of the wrong
size does not fail, it shifts everything after it.

Coordinates are fractions of the document, not points: a knot half way across
a 200pt document stores 0.5. They are written in 8.24 fixed point, and
**vertical before horizontal**, which is the opposite of every other
coordinate pair in this codebase. Swapping them transposes the artwork into
something that still opens and still looks like a drawing.

The same records serve both vector modes. ``bitmap_paths`` puts them in image
resources, where they become entries in the Paths panel; ``vector`` puts them
in a layer's own `vmsk` block, where they become its shape.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from services.svg_paths import Subpath

#: Record selectors, from Adobe's path resource table.
CLOSED_LENGTH = 0
CLOSED_KNOT_LINKED = 1
OPEN_LENGTH = 3
OPEN_KNOT_LINKED = 4

#: Paths live in image resources 2000 to 2997, and nowhere else.
PATH_RESOURCE_FIRST = 2000
PATH_RESOURCE_LAST = 2997

_RECORD_SIZE = 26
_FIXED_ONE = 1 << 24
_BLOCK_SIGNATURE = b"8BIM"


@dataclass(frozen=True)
class Knot:
    """A knot as read back: three points, each (vertical, horizontal)."""

    preceding: tuple[float, float]
    anchor: tuple[float, float]
    following: tuple[float, float]

    @property
    def vertical(self) -> float:
        return self.anchor[0]

    @property
    def horizontal(self) -> float:
        return self.anchor[1]


def path_records(subpaths: tuple[Subpath, ...], document: tuple[int, int]) -> bytes:
    """Every subpath as PSD records, ready for a resource or a vector mask."""
    if not subpaths:
        return b""
    width, height = max(1, document[0]), max(1, document[1])
    # No fill-rule record: it is optional, and a reader counts records to
    # find them, so every record that is not needed is one more chance to
    # shift everything after it.
    out: list[bytes] = []
    for subpath in subpaths:
        knots = _knots_of(subpath)
        if not knots:
            continue
        length_selector = CLOSED_LENGTH if subpath.closed else OPEN_LENGTH
        knot_selector = CLOSED_KNOT_LINKED if subpath.closed else OPEN_KNOT_LINKED
        out.append(_record(length_selector, struct.pack(">H", len(knots))))
        for preceding, anchor, following in knots:
            out.append(_record(knot_selector, b"".join(
                _point(point, width, height) for point in (preceding, anchor, following))))
    return b"".join(out)


def path_resources(
    named: list[tuple[str, tuple[Subpath, ...]]], document: tuple[int, int]
) -> bytes:
    """One Paths-panel entry per shape, as image resource blocks.

    The format has room for 998 of them. A sheet with more shapes than that
    keeps the first 998 rather than writing ids outside the range, which a
    reader would either ignore or refuse.
    """
    blocks: list[bytes] = []
    resource_id = PATH_RESOURCE_FIRST
    for name, subpaths in named:
        if resource_id > PATH_RESOURCE_LAST:
            break
        records = path_records(subpaths, document)
        if not records:
            continue
        blocks.append(_resource_block(resource_id, name, records))
        resource_id += 1
    return b"".join(blocks)


def shape_layer_blocks(
    subpaths: tuple[Subpath, ...], document: tuple[int, int], colour: tuple[int, int, int]
) -> bytes:
    """The two blocks that turn a pixel layer into a shape layer.

    A shape layer in Photoshop is a solid-colour fill clipped by a vector
    mask, and both halves are additional layer information: `vmsk` holds the
    outline, `SoCo` the fill. A layer with only the mask reads as a pixel
    layer that happens to have a vector mask -- close, but not the thing.
    """
    records = path_records(subpaths, document)
    if not records:
        return b""
    # Version 3, flags 1: the mask is used, not disabled and not inverted.
    mask = struct.pack(">II", 3, 1) + records
    return _layer_block(b"SoCo", _solid_colour_descriptor(colour)) + _layer_block(b"vmsk", mask)


def read_solid_colour(blocks: bytes) -> tuple[float, float, float] | None:
    """The RGB a `SoCo` block carries, for checking what was written."""
    marker = _BLOCK_SIGNATURE + b"SoCo"
    start = blocks.find(marker)
    if start < 0:
        return None
    channels: list[float] = []
    for key in (b"Rd  ", b"Grn ", b"Bl  "):
        at = blocks.find(key + b"doub", start)
        if at < 0:
            return None
        channels.append(struct.unpack(">d", blocks[at + 8:at + 16])[0])
    return tuple(channels)


# --- Adobe descriptors ----------------------------------------------------
# A descriptor is a format inside the format: a class, then a count, then
# key/type/value triples. Only the shape a solid-colour fill needs is built
# here, which is one nested descriptor holding three doubles.

def _solid_colour_descriptor(colour: tuple[int, int, int]) -> bytes:
    rgb = _descriptor(b"RGBC", [
        (b"Rd  ", _double(colour[0])),
        (b"Grn ", _double(colour[1])),
        (b"Bl  ", _double(colour[2])),
    ])
    payload = _descriptor(b"null", [(b"Clr ", b"Objc" + rgb)])
    return struct.pack(">I", 16) + payload      # descriptor version


def _descriptor(class_id: bytes, entries: list[tuple[bytes, bytes]]) -> bytes:
    out = struct.pack(">I", 0)                  # no unicode class name
    out += struct.pack(">I", len(class_id)) + class_id
    out += struct.pack(">I", len(entries))
    for key, value in entries:
        out += struct.pack(">I", 0) + key       # a four-character key writes its length as 0
        out += value
    return out


def _double(value: float) -> bytes:
    return b"doub" + struct.pack(">d", float(value))


def _layer_block(key: bytes, payload: bytes) -> bytes:
    body = payload + b"\x00" * (-len(payload) % 4)
    return _BLOCK_SIGNATURE + key + struct.pack(">I", len(body)) + body


# --- reading back, for the tests and for anyone checking a file -----------

def read_record_header(blob: bytes, index: int) -> tuple[int, int]:
    """The selector and the 16-bit value of record ``index``."""
    offset = index * _RECORD_SIZE
    selector, value = struct.unpack(">HH", blob[offset:offset + 4])
    return selector, value


def read_knot(blob: bytes, index: int) -> Knot:
    offset = index * _RECORD_SIZE + 2
    numbers = struct.unpack(">6i", blob[offset:offset + 24])
    points = [(numbers[i] / _FIXED_ONE, numbers[i + 1] / _FIXED_ONE) for i in (0, 2, 4)]
    return Knot(*points)


def resource_ids(resources: bytes) -> list[int]:
    """The resource ids in a block of image resources, in order."""
    ids: list[int] = []
    offset = 0
    while offset + 12 <= len(resources):
        if resources[offset:offset + 4] != _BLOCK_SIGNATURE:
            break
        resource_id = struct.unpack(">H", resources[offset + 4:offset + 6])[0]
        name_length = resources[offset + 6]
        name_field = 1 + name_length
        name_field += name_field % 2
        data_offset = offset + 6 + name_field
        size = struct.unpack(">I", resources[data_offset:data_offset + 4])[0]
        ids.append(resource_id)
        offset = data_offset + 4 + size + (size % 2)
    return ids


# --- the bytes ------------------------------------------------------------

def _record(selector: int, payload: bytes) -> bytes:
    body = payload + b"\x00" * (_RECORD_SIZE - 2 - len(payload))
    return struct.pack(">H", selector) + body


def _point(point: tuple[float, float], width: int, height: int) -> bytes:
    """One point as vertical then horizontal, each a fraction in 8.24 fixed."""
    return struct.pack(">ii", _fixed(point[1] / height), _fixed(point[0] / width))


def _fixed(value: float) -> int:
    return int(round(value * _FIXED_ONE))


def _knots_of(subpath: Subpath) -> list[tuple[tuple[float, float], ...]]:
    """Segments become knots: each anchor carries the controls either side.

    SVG describes a curve per segment; PSD describes a point and what leaves
    it, so the two controls of one knot come from two different segments.
    """
    segments = subpath.segments
    if not segments:
        return []
    knots = []
    for index, segment in enumerate(segments):
        previous = segments[index - 1] if index else (segments[-1] if subpath.closed else None)
        preceding = previous.c2 if previous is not None else segment.start
        knots.append((preceding, segment.start, segment.c1))
    if not subpath.closed:
        last = segments[-1]
        knots.append((last.c2, last.end, last.end))
    return knots


def _resource_block(resource_id: int, name: str, data: bytes) -> bytes:
    raw = name.encode("utf-8")[:255]
    name_field = struct.pack(">B", len(raw)) + raw
    name_field += b"\x00" * (len(name_field) % 2)
    return (
        _BLOCK_SIGNATURE
        + struct.pack(">H", resource_id)
        + name_field
        + struct.pack(">I", len(data))
        + data
        + b"\x00" * (len(data) % 2)
    )
