"""The PSD writer, checked by a reader that shares no code with it.

Writing a binary format against its own parser proves only that the two agree.
These tests read every file back with `psd-tools`, an independent
implementation, so a field written to the wrong offset or a length that counts
the wrong bytes shows up as a wrong answer rather than a matching mistake.

`psd-tools` is a test dependency only. The app bundles the writer, never a
reader.
"""
from __future__ import annotations

import numpy as np
import pytest
from psd_tools import PSDImage

from services.psd_writer import PsdLayer, write_psd


def square(size: int, rgb: tuple[int, int, int], alpha: int = 255) -> np.ndarray:
    patch = np.zeros((size, size, 4), dtype=np.uint8)
    patch[:, :, 0], patch[:, :, 1], patch[:, :, 2] = rgb
    patch[:, :, 3] = alpha
    return patch


@pytest.fixture
def written(tmp_path):
    def write(layers, canvas=(256, 256), **kwargs):
        path = tmp_path / "out.psd"
        write_psd(path, layers, canvas, **kwargs)
        return path

    return write


# --- the document -------------------------------------------------------

def test_the_canvas_is_the_one_asked_for(written):
    """Derived from the layers it would crop to whatever happened to be drawn."""
    psd = PSDImage.open(written([PsdLayer("one", square(20, (255, 0, 0)), top=5, left=5)],
                                canvas=(400, 300)))
    assert psd.size == (400, 300)


def test_a_document_with_no_layers_is_still_readable(written):
    psd = PSDImage.open(written([], canvas=(64, 48)))
    assert psd.size == (64, 48)
    assert len(list(psd)) == 0


def test_every_layer_arrives(written):
    layers = [PsdLayer(f"shape_{i:03d}", square(30, (i * 20, 0, 0)), top=i * 10, left=i * 10)
              for i in range(1, 6)]
    psd = PSDImage.open(written(layers))
    assert len(list(psd)) == 5


# --- what a layer says about itself ---------------------------------------

def test_a_layer_keeps_its_name(written):
    psd = PSDImage.open(written([PsdLayer("icon_042", square(10, (1, 2, 3)))]))
    assert [layer.name for layer in psd] == ["icon_042"]


def test_a_name_survives_accents_and_length(written):
    """The Pascal string is legacy and truncates; the luni block must carry it."""
    name = "Ψηφιδωτό — icon_001 from a rather long sheet name.svg"
    psd = PSDImage.open(written([PsdLayer(name, square(10, (1, 2, 3)))]))
    assert [layer.name for layer in psd] == [name]


def test_a_layer_sits_where_it_was_put(written):
    psd = PSDImage.open(written([PsdLayer("one", square(40, (0, 255, 0)), top=17, left=23)]))
    layer = list(psd)[0]
    assert layer.offset == (23, 17)
    assert layer.size == (40, 40)


def test_layers_are_bounded_to_their_own_pixels(written):
    """A layer the size of the canvas per shape is how a PSD reaches a gigabyte."""
    psd = PSDImage.open(written([PsdLayer("small", square(12, (9, 9, 9)), top=100, left=100)],
                                canvas=(2000, 2000)))
    assert list(psd)[0].size == (12, 12)


# --- the pixels themselves ------------------------------------------------

def test_a_layer_carries_the_colour_it_was_given(written):
    psd = PSDImage.open(written([PsdLayer("red", square(8, (237, 28, 36)))]))
    pixels = np.array(list(psd)[0].topil())
    assert tuple(pixels[4, 4][:3]) == (237, 28, 36)


def test_transparency_is_preserved_per_layer(written):
    patch = square(8, (0, 0, 0))
    patch[:4, :, 3] = 0          # top half cut away
    psd = PSDImage.open(written([PsdLayer("half", patch)]))
    pixels = np.array(list(psd)[0].topil().convert("RGBA"))
    assert pixels[1, 4][3] == 0
    assert pixels[6, 4][3] == 255


def test_the_flattened_preview_shows_the_layers(written):
    """A blank composite opens as an empty document anywhere layers are not read.

    `topil` returns the preview as stored, which is the thing written here;
    `composite` re-renders from the layers and would pass even if the stored
    preview were empty.
    """
    psd = PSDImage.open(written(
        [PsdLayer("block", square(40, (10, 200, 30)), top=10, left=10)], canvas=(100, 100)))
    preview = psd.topil()
    assert preview.getpixel((30, 30)) == (10, 200, 30)
    assert preview.getpixel((90, 90)) != (10, 200, 30), "the shape must not fill the canvas"


def test_a_later_layer_covers_an_earlier_one_in_the_preview(written):
    psd = PSDImage.open(written([
        PsdLayer("under", square(40, (255, 0, 0)), top=0, left=0),
        PsdLayer("over", square(40, (0, 0, 255)), top=0, left=0),
    ], canvas=(64, 64)))
    assert psd.topil().getpixel((20, 20)) == (0, 0, 255)


def test_the_preview_keeps_its_transparency(written):
    """Otherwise the file opens on a black background instead of nothing.

    `topil` hands back RGB by convention, which hides this; the fourth channel
    is where the answer is, and getting the composite channel order wrong put
    the alpha in the red slot without changing the file's size or structure.
    """
    psd = PSDImage.open(written([PsdLayer("red", square(20, (255, 0, 0)), top=5, left=5)],
                                canvas=(64, 64)))
    channels = psd.numpy()
    assert channels.shape[-1] == 4, "the composite must carry an alpha channel"
    assert channels[10, 10, 3] == 1.0
    assert channels[50, 50, 3] == 0.0


# --- refusals -------------------------------------------------------------

def test_an_impossible_canvas_is_refused(tmp_path):
    with pytest.raises(ValueError):
        write_psd(tmp_path / "bad.psd", [], (0, 100))


def test_a_layer_with_no_pixels_is_dropped_rather_than_written(written):
    psd = PSDImage.open(written([
        PsdLayer("real", square(10, (1, 1, 1))),
        PsdLayer("empty", np.zeros((0, 0, 4), dtype=np.uint8)),
    ]))
    assert [layer.name for layer in psd] == ["real"]
