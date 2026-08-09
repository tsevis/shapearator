"""Characterization tests for the pixel-level export helpers."""
from __future__ import annotations

import numpy as np
from PIL import Image

from services.raster_ops import (
    build_transparent_crop_rgba,
    compose_icon_on_canvas,
    compute_exterior_background_mask,
    estimate_background_rgba,
    extract_palette_from_image,
    is_effectively_monochrome_png,
    rgb_to_hex,
    write_rgba_crop,
)


def _solid_png(path, size=(20, 20), color=(255, 0, 0, 255)):
    Image.new("RGBA", size, color).save(path)
    return path


# --- colour helpers -------------------------------------------------------

def test_rgb_to_hex_zero_pads_each_channel():
    assert rgb_to_hex(np.array([255, 0, 128])) == "#ff0080"
    assert rgb_to_hex(np.array([0, 0, 0])) == "#000000"


def test_estimate_background_rgba_reads_the_border():
    bgr = np.zeros((50, 50, 3), dtype=np.uint8)
    bgr[:, :] = (200, 100, 50)  # BGR
    bgr[20:30, 20:30] = (0, 0, 0)  # centre ink must not sway the estimate
    assert estimate_background_rgba(bgr) == (50, 100, 200, 255)  # returned RGBA


# --- canvas composition ---------------------------------------------------

def test_compose_original_mode_centres_without_scaling():
    image = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
    canvas = compose_icon_on_canvas(image, (200, 200), "original", uniform_scale=1.0)
    assert canvas.size == (200, 200)
    assert canvas.getpixel((100, 100)) == (255, 0, 0, 255)  # centre is ink
    assert canvas.getpixel((0, 0)) == (0, 0, 0, 0)  # margin is transparent


def test_compose_individual_fit_fills_the_limiting_axis():
    image = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
    canvas = compose_icon_on_canvas(image, (200, 200), "individual_fit", uniform_scale=1.0)
    # 200/100 is the binding ratio, so the icon becomes 200x100 and is centred.
    assert canvas.getpixel((0, 100)) == (255, 0, 0, 255)
    assert canvas.getpixel((100, 5)) == (0, 0, 0, 0)


def test_compose_uniform_mode_uses_the_supplied_scale():
    image = Image.new("RGBA", (100, 100), (0, 255, 0, 255))
    canvas = compose_icon_on_canvas(image, (200, 200), "uniform_to_largest", uniform_scale=0.5)
    assert canvas.getpixel((100, 100)) == (0, 255, 0, 255)
    assert canvas.getpixel((60, 60)) == (0, 0, 0, 0)  # 50x50 render centred at 75..125


def test_compose_clips_an_oversized_render_instead_of_failing():
    image = Image.new("RGBA", (400, 400), (0, 0, 255, 255))
    canvas = compose_icon_on_canvas(image, (100, 100), "original", uniform_scale=1.0)
    assert canvas.size == (100, 100)
    assert canvas.getpixel((50, 50)) == (0, 0, 255, 255)


def test_compose_honours_an_opaque_background():
    image = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
    canvas = compose_icon_on_canvas(
        image, (50, 50), "original", uniform_scale=1.0, background_rgba=(255, 255, 255, 255)
    )
    assert canvas.getpixel((0, 0)) == (255, 255, 255, 255)


def test_compose_does_not_mutate_the_source_image():
    image = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
    compose_icon_on_canvas(image, (200, 200), "individual_fit", uniform_scale=1.0)
    assert image.size == (100, 50)


# --- transparency ---------------------------------------------------------

def test_exterior_mask_keeps_enclosed_holes_opaque():
    foreground = np.zeros((10, 10), dtype=bool)
    foreground[2:8, 2:8] = True
    foreground[4:6, 4:6] = False  # a hole fully enclosed by ink
    exterior = compute_exterior_background_mask(foreground)
    assert exterior[0, 0]  # outside the shape
    assert not exterior[5, 5]  # enclosed hole stays part of the icon
    assert not exterior[3, 3]  # ink itself


def test_build_transparent_crop_clears_only_the_outside():
    crop_bgr = np.full((10, 10, 3), 255, dtype=np.uint8)
    crop_bgr[2:8, 2:8] = 0
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    rgba = build_transparent_crop_rgba(crop_bgr, mask)
    assert rgba.shape == (10, 10, 4)
    assert rgba[0, 0, 3] == 0
    assert rgba[5, 5, 3] == 255


# --- analysis -------------------------------------------------------------

def test_monochrome_detection_accepts_greys(tmp_path):
    assert is_effectively_monochrome_png(_solid_png(tmp_path / "grey.png", color=(80, 80, 80, 255)))


def test_monochrome_detection_rejects_saturated_colour(tmp_path):
    assert not is_effectively_monochrome_png(_solid_png(tmp_path / "red.png", color=(255, 0, 0, 255)))


def test_fully_transparent_image_counts_as_monochrome(tmp_path):
    assert is_effectively_monochrome_png(_solid_png(tmp_path / "blank.png", color=(0, 0, 0, 0)))


def test_palette_ranks_by_coverage_and_ignores_transparency(tmp_path):
    image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (0, 0, 10, 7))
    image.paste((0, 0, 255, 255), (0, 7, 10, 10))
    path = tmp_path / "two-tone.png"
    image.save(path)
    dominant, palette = extract_palette_from_image(path)
    # Channels are floored to a multiple of 16 before counting, so pure red is
    # reported as #f00000; the ranking, not the exact hex, is what matters here.
    assert dominant == "#f00000"
    assert "#0000f0" in palette


def test_palette_of_an_empty_image_is_empty(tmp_path):
    dominant, palette = extract_palette_from_image(_solid_png(tmp_path / "blank.png", color=(0, 0, 0, 0)))
    assert dominant is None and palette == []


def test_palette_respects_the_colour_cap(tmp_path):
    pixels = np.zeros((1, 8, 4), dtype=np.uint8)
    for index in range(8):
        pixels[0, index] = (index * 32, 255 - index * 16, index * 16, 255)
    path = tmp_path / "many.png"
    Image.fromarray(pixels, mode="RGBA").save(path)
    _dominant, palette = extract_palette_from_image(path, max_colors=3)
    assert len(palette) == 3


# --- io -------------------------------------------------------------------

def test_write_rgba_crop_round_trips(tmp_path):
    rgba = np.zeros((4, 6, 4), dtype=np.uint8)
    rgba[:, :] = (10, 20, 30, 255)
    path = tmp_path / "crop.png"
    write_rgba_crop(rgba, path)
    with Image.open(path) as written:
        assert written.size == (6, 4)
        assert written.convert("RGBA").getpixel((0, 0)) == (10, 20, 30, 255)
