"""Writing a layered Photoshop file, one layer per extracted shape.

Written here rather than taken from a library. The one Python package that
writes PSD at all, `pytoshop`, last shipped in 2019; its Cython extension no
longer builds, which leaves two of its three compression paths raising
`NameError` at write time. Bundling that into an app meant shipping a
dependency whose working surface is one code path wide.

The format is Adobe's own and the part we need is small: a header, an empty
colour-mode block, image resources, the layer section, and a flattened
composite. Everything is written big-endian, which is what `>` means in every
`struct` format below.

What this module does *not* do is vector shape layers. Those live in the
layer's additional information as `vmsk`/`vsms` path records plus a `SoCo`
solid-colour fill, and they are added by :mod:`services.psd_vector`.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

import numpy as np

#: 8-bit RGB, which is the only mode this writer produces.
_DEPTH = 8
_COLOR_MODE_RGB = 3
#: 0 raw, 1 RLE, 2 zlib, 3 zlib with prediction. Plain zlib is 2, and
#: getting that wrong is silent: a reader decodes 3 as deltas and either
#: walks off the end of the row or returns smeared colour.
_COMPRESSION_ZIP = 2
#: The layer section names its channels by id, and Photoshop writes the
#: transparency mask first.
_CHANNEL_IDS = (-1, 0, 1, 2)
#: The composite section names nothing: channels are positional, red first
#: and alpha last. Reusing the layer order there shifts every channel by
#: one, which reads as a plausible image in the wrong colours.
_COMPOSITE_ORDER = (0, 1, 2, 3)
_SIGNATURE = b"8BPS"
_BLOCK_SIGNATURE = b"8BIM"


@dataclass(frozen=True)
class PsdLayer:
    """One layer: its pixels, and where they sit on the canvas.

    ``rgba`` is (height, width, 4) uint8 and is the layer's own bounds, not
    the canvas. A layer covering the whole canvas when it holds a 60pt shape
    would store a megabyte of transparent pixels per shape.
    """

    name: str
    rgba: np.ndarray
    top: int = 0
    left: int = 0

    @property
    def height(self) -> int:
        return int(self.rgba.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgba.shape[1])

    @property
    def is_empty(self) -> bool:
        return self.height == 0 or self.width == 0


def write_psd(
    path: Path,
    layers: Sequence[PsdLayer],
    canvas_size: tuple[int, int],
    extra_resources: bytes = b"",
    layer_extras: Sequence[bytes] | None = None,
) -> None:
    """Write ``layers`` into one PSD at ``path``.

    ``canvas_size`` is stated rather than derived from the layers: a sheet's
    icons are meant to sit on the canvas the user chose, and deriving it from
    the layers' union would silently crop to whatever happened to be drawn.

    ``extra_resources`` is pre-built image-resource blocks (paths, for
    instance) and ``layer_extras`` is one additional-information blob per
    layer, which is where vector shape data goes.
    """
    width, height = canvas_size
    if width <= 0 or height <= 0:
        raise ValueError(f"PSD canvas must be positive, got {canvas_size}")

    drawn = [layer for layer in layers if not layer.is_empty]
    extras = list(layer_extras or [])
    if extras and len(extras) != len(layers):
        raise ValueError("layer_extras must have one entry per layer")
    # Keep the extras aligned with the layers that survived the empty filter.
    paired = [
        (layer, extras[index] if extras else b"")
        for index, layer in enumerate(layers)
        if not layer.is_empty
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        _write_header(fh, width, height)
        _write_length_prefixed(fh, b"")                 # colour mode data
        _write_length_prefixed(fh, extra_resources)     # image resources
        _write_layer_and_mask_info(fh, paired)
        _write_composite(fh, drawn, width, height)


# --- header and simple sections -------------------------------------------

def _write_header(fh: BinaryIO, width: int, height: int) -> None:
    fh.write(_SIGNATURE)
    fh.write(struct.pack(">H", 1))          # version 1 (PSD, not PSB)
    fh.write(b"\x00" * 6)                   # reserved
    fh.write(struct.pack(">H", len(_CHANNEL_IDS)))
    fh.write(struct.pack(">II", height, width))
    fh.write(struct.pack(">HH", _DEPTH, _COLOR_MODE_RGB))


def _write_length_prefixed(fh: BinaryIO, payload: bytes) -> None:
    fh.write(struct.pack(">I", len(payload)))
    fh.write(payload)


# --- the layer section ----------------------------------------------------

def _write_layer_and_mask_info(
    fh: BinaryIO, paired: Sequence[tuple[PsdLayer, bytes]]
) -> None:
    layer_info = _build_layer_info(paired)
    # Global layer mask info: present but empty, which Photoshop expects even
    # when nothing uses it.
    section = layer_info + struct.pack(">I", 0)
    _write_length_prefixed(fh, section)


def _build_layer_info(paired: Sequence[tuple[PsdLayer, bytes]]) -> bytes:
    if not paired:
        # A layer count of zero is legal, and simpler than omitting the
        # section: readers still find a well-formed, empty layer list.
        return struct.pack(">I", _pad2(struct.pack(">h", 0)).__len__()) + _pad2(struct.pack(">h", 0))

    records: list[bytes] = []
    channel_data: list[bytes] = []
    for layer, extra in paired:
        channels = [_compress_channel(_channel_plane(layer, cid)) for cid in _CHANNEL_IDS]
        records.append(_layer_record(layer, [len(blob) for blob in channels], extra))
        channel_data.extend(channels)

    body = struct.pack(">h", len(paired)) + b"".join(records) + b"".join(channel_data)
    body = _pad2(body)
    return struct.pack(">I", len(body)) + body


def _channel_plane(layer: PsdLayer, channel_id: int) -> np.ndarray:
    """One channel of a layer, as a contiguous uint8 plane."""
    index = 3 if channel_id == -1 else channel_id
    return np.ascontiguousarray(layer.rgba[:, :, index], dtype=np.uint8)


def _compress_channel(plane: np.ndarray) -> bytes:
    return struct.pack(">H", _COMPRESSION_ZIP) + zlib.compress(plane.tobytes())


def _layer_record(layer: PsdLayer, channel_lengths: Sequence[int], extra: bytes) -> bytes:
    top, left = layer.top, layer.left
    bottom, right = top + layer.height, left + layer.width

    out = struct.pack(">iiii", top, left, bottom, right)
    out += struct.pack(">H", len(_CHANNEL_IDS))
    for channel_id, length in zip(_CHANNEL_IDS, channel_lengths):
        out += struct.pack(">hI", channel_id, length)

    out += _BLOCK_SIGNATURE
    out += b"norm"                      # blend mode
    out += struct.pack(">BBBB", 255, 0, 0, 0)   # opacity, clipping, flags, filler

    additional = _unicode_name_block(layer.name) + extra
    extra_data = (
        struct.pack(">I", 0)            # layer mask data: none
        + struct.pack(">I", 0)          # blending ranges: none
        + _pascal_string(layer.name)
        + additional
    )
    out += struct.pack(">I", len(extra_data)) + extra_data
    return out


def _unicode_name_block(name: str) -> bytes:
    """A `luni` block, so a name survives non-ASCII and more than 31 bytes.

    The Pascal string in the record is the legacy name and is truncated; every
    reader since Photoshop 5 prefers this one.
    """
    encoded = name.encode("utf-16-be")
    payload = struct.pack(">I", len(name)) + encoded
    return _additional_info_block(b"luni", _pad4(payload))


def _additional_info_block(key: bytes, payload: bytes) -> bytes:
    return _BLOCK_SIGNATURE + key + struct.pack(">I", len(payload)) + payload


def _pascal_string(name: str) -> bytes:
    """A length-prefixed name, padded so the whole field is a multiple of 4."""
    raw = name.encode("utf-8")[:255]
    field = struct.pack(">B", len(raw)) + raw
    return field + b"\x00" * (-len(field) % 4)


def _pad2(payload: bytes) -> bytes:
    return payload + b"\x00" * (len(payload) % 2)


def _pad4(payload: bytes) -> bytes:
    return payload + b"\x00" * (-len(payload) % 4)


# --- the flattened preview ------------------------------------------------

def _write_composite(
    fh: BinaryIO, layers: Sequence[PsdLayer], width: int, height: int
) -> None:
    """The flattened image every reader shows before it parses layers.

    Composited here rather than left blank: a PSD whose preview is empty opens
    as a blank document in anything that does not read the layer section,
    which includes Finder's own thumbnail.
    """
    canvas = np.zeros((height, width, 4), dtype=np.float32)
    for layer in layers:
        top, left = layer.top, layer.left
        bottom, right = min(top + layer.height, height), min(left + layer.width, width)
        if bottom <= top or right <= left or top < 0 or left < 0:
            continue
        patch = layer.rgba[: bottom - top, : right - left].astype(np.float32)
        alpha = patch[:, :, 3:4] / 255.0
        target = canvas[top:bottom, left:right]
        target[:, :, :3] = patch[:, :, :3] * alpha + target[:, :, :3] * (1.0 - alpha)
        target[:, :, 3:4] = np.clip(alpha + target[:, :, 3:4] * (1.0 - alpha), 0.0, 1.0)

    flat = canvas.astype(np.uint8)
    flat[:, :, 3] = (canvas[:, :, 3] * 255.0).astype(np.uint8)

    # One compression marker, then a single stream covering every channel in
    # order. Per-channel streams are what the layer section uses; the image
    # data section does not, and a reader hits the end of the first one while
    # still expecting three more channels' worth of bytes.
    planes = b"".join(
        np.ascontiguousarray(flat[:, :, index]).tobytes() for index in _COMPOSITE_ORDER
    )
    fh.write(struct.pack(">H", _COMPRESSION_ZIP))
    fh.write(zlib.compress(planes))
