"""Pixel-level work: canvas composition, transparency, and colour analysis."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .geometry import sample_border_pixels

DEFAULT_BG_RGBA = (255, 255, 255, 255)


def write_rgba_crop(rgba_crop: np.ndarray, output_path: Path) -> None:
    Image.fromarray(rgba_crop, mode="RGBA").save(output_path)


def compose_icon_on_canvas(
    image: Image.Image,
    canvas_size: tuple[int, int],
    mode: str,
    uniform_scale: float,
    background_rgba: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
    """Scale ``image`` per ``mode`` and centre it on a fresh canvas.

    A render larger than the canvas is clipped rather than rejected, so an
    aggressive uniform scale degrades visibly instead of raising.
    """
    target_w, target_h = canvas_size
    scale = 1.0
    if mode == "uniform_to_largest":
        scale = uniform_scale
    elif mode == "individual_fit":
        scale = min(target_w / max(1, image.width), target_h / max(1, image.height))

    render = image
    if abs(scale - 1.0) > 1e-6:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        new_size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        render = image.resize(new_size, resampling)

    canvas = Image.new("RGBA", (target_w, target_h), background_rgba)
    x = (target_w - render.width) // 2
    y = (target_h - render.height) // 2
    if x >= 0 and y >= 0:
        canvas.alpha_composite(render, (x, y))
        return canvas

    src_x = max(0, -x)
    src_y = max(0, -y)
    dst_x = max(0, x)
    dst_y = max(0, y)
    crop_w = min(render.width - src_x, target_w - dst_x)
    crop_h = min(render.height - src_y, target_h - dst_y)
    if crop_w > 0 and crop_h > 0:
        clipped = render.crop((src_x, src_y, src_x + crop_w, src_y + crop_h))
        canvas.alpha_composite(clipped, (dst_x, dst_y))
    return canvas


def estimate_background_rgba(bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Guess the paper colour from the border median, returned as RGBA."""
    border = sample_border_pixels(bgr.astype(np.float32))
    if border.size == 0:
        return DEFAULT_BG_RGBA
    median_bgr = np.median(border, axis=0)
    return int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0]), 255


def compute_exterior_background_mask(foreground_mask: np.ndarray) -> np.ndarray:
    """Return the pixels reachable from outside the shape.

    Flood-filling from a padded corner means holes fully enclosed by ink (the
    inside of an 'O', say) are *not* marked as background, so they survive the
    transparency pass instead of being punched out.
    """
    padded_foreground = np.pad(foreground_mask.astype(bool), 1, constant_values=False)
    flood = np.where(~padded_foreground, 255, 0).astype(np.uint8)
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 128)
    exterior = flood == 128
    return exterior[1:-1, 1:-1]


def build_transparent_crop_rgba(crop_bgr: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGBA)
    exterior_background = compute_exterior_background_mask(crop_mask > 0)
    rgba[:, :, 3] = np.where(exterior_background, 0, 255).astype(np.uint8)
    return rgba


def is_effectively_monochrome_png(png_path: Path) -> bool:
    """True when every visible pixel is near-grey, so tracing beats embedding."""
    with Image.open(png_path).convert("RGBA") as image:
        pixels = np.array(image)
    alpha = pixels[:, :, 3] > 0
    if not np.any(alpha):
        return True
    rgb = pixels[:, :, :3][alpha]
    channel_spread = np.max(np.abs(rgb[:, 0].astype(np.int16) - rgb[:, 1].astype(np.int16)))
    channel_spread = max(channel_spread, np.max(np.abs(rgb[:, 1].astype(np.int16) - rgb[:, 2].astype(np.int16))))
    return channel_spread <= 6


def rgb_to_hex(rgb: np.ndarray) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def extract_palette_from_image(image_path: Path, max_colors: int = 4) -> tuple[str | None, list[str]]:
    """Return the dominant colour and a short palette, ignoring transparency."""
    with Image.open(image_path).convert("RGBA") as image:
        rgba = np.array(image)
    alpha = rgba[:, :, 3] > 0
    if not np.any(alpha):
        return None, []
    rgb = rgba[:, :, :3][alpha]
    if len(rgb) == 0:
        return None, []

    # Quantize lightly so hand-drawn anti-aliasing does not explode the palette.
    quantized = (rgb // 16) * 16
    unique, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    palette = [rgb_to_hex(unique[idx]) for idx in order[:max_colors]]
    dominant = palette[0] if palette else None
    return dominant, palette
