"""Bounding boxes, foreground masks, and icon detection.

Pure geometry and OpenCV work: no file I/O, no external binaries, no settings.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def padded(self, pad: int, limit_w: int | None = None, limit_h: int | None = None) -> "Box":
        x = self.x - pad
        y = self.y - pad
        x2 = self.x2 + pad
        y2 = self.y2 + pad
        if limit_w is not None:
            x = max(0, x)
            x2 = min(limit_w, x2)
        if limit_h is not None:
            y = max(0, y)
            y2 = min(limit_h, y2)
        return Box(int(x), int(y), int(x2 - x), int(y2 - y))

    def union(self, other: "Box") -> "Box":
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return Box(x1, y1, x2 - x1, y2 - y1)


def box_contains_point(box: Box, x: float, y: float, pad: int = 0) -> bool:
    return (box.x - pad) <= x <= (box.x2 + pad) and (box.y - pad) <= y <= (box.y2 + pad)


def make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def sample_border_pixels(image: np.ndarray, band: int = 12) -> np.ndarray:
    """Return the pixels along all four edges, flattened to one row per pixel."""
    h, w = image.shape[:2]
    band = max(1, min(band, h // 4 or 1, w // 4 or 1))
    top = image[:band, :, :].reshape(-1, 3)
    bottom = image[-band:, :, :].reshape(-1, 3)
    left = image[:, :band, :].reshape(-1, 3)
    right = image[:, -band:, :].reshape(-1, 3)
    return np.concatenate([top, bottom, left, right], axis=0)


def build_binary_mask(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)


def build_foreground_mask(bgr: np.ndarray) -> np.ndarray:
    """Separate ink from paper using both luminance and colour distance.

    Otsu alone loses coloured marks on a coloured ground, so the LAB distance
    from the sampled border colour is unioned into the plain grayscale mask.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    base_mask = build_binary_mask(gray)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = sample_border_pixels(lab)
    if border.size == 0:
        return base_mask

    bg_mean = border.mean(axis=0)
    bg_l = bg_mean[0]
    bg_dist = np.linalg.norm(border - bg_mean, axis=1)
    bg_l_delta = np.abs(border[:, 0] - bg_l)

    delta = np.linalg.norm(lab - bg_mean, axis=2)
    l_delta = np.abs(lab[:, :, 0] - bg_l)
    delta_threshold = max(12.0, float(np.percentile(bg_dist, 95)) * 2.5 + 6.0)
    l_threshold = max(10.0, float(np.percentile(bg_l_delta, 95)) * 2.5 + 4.0)

    color_mask = (delta > delta_threshold) | (l_delta > l_threshold)
    combined = np.where(color_mask, 255, 0).astype(np.uint8)
    combined = cv2.bitwise_or(combined, base_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return combined


def detect_icon_boxes(binary: np.ndarray, min_area: int, merge_gap: int) -> list[Box]:
    """Group nearby marks into icon-sized components, in reading order."""
    kernel_size = max(3, make_odd(merge_gap))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    grouped = cv2.dilate(binary, kernel, iterations=1)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    boxes: list[Box] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        boxes.append(Box(int(x), int(y), int(w), int(h)))
    return sort_boxes_reading_order(boxes)


def tighten_box(binary: np.ndarray, box: Box) -> Box:
    """Shrink ``box`` to the ink it actually contains, after dilation grew it."""
    region = binary[box.y:box.y2, box.x:box.x2]
    ys, xs = np.where(region > 0)
    if len(xs) == 0 or len(ys) == 0:
        return box
    x1 = box.x + int(xs.min())
    y1 = box.y + int(ys.min())
    x2 = box.x + int(xs.max()) + 1
    y2 = box.y + int(ys.max()) + 1
    return Box(x1, y1, x2 - x1, y2 - y1)


def sort_boxes_reading_order(boxes: list[Box]) -> list[Box]:
    """Order boxes left-to-right within rows, top-to-bottom across rows.

    Rows are clustered by vertical centre with a tolerance derived from the
    median height, so a hand-drawn sheet that is not perfectly aligned still
    numbers in the order a person would read it.
    """
    if not boxes:
        return []
    median_height = sorted(box.h for box in boxes)[len(boxes) // 2]
    row_threshold = max(20, int(median_height * 0.75))
    rows: list[list[Box]] = []
    for box in sorted(boxes, key=lambda item: item.cy):
        placed = False
        for row in rows:
            row_center = sum(item.cy for item in row) / len(row)
            if abs(box.cy - row_center) <= row_threshold:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    ordered: list[Box] = []
    for row in sorted(rows, key=lambda items: sum(item.cy for item in items) / len(items)):
        ordered.extend(sorted(row, key=lambda item: item.x))
    return ordered


def compute_uniform_scale(source_sizes: list[tuple[int, int]], canvas_size: tuple[int, int]) -> float:
    """Return the one scale that fits the largest source inside the canvas."""
    if not source_sizes:
        return 1.0
    max_w = max(width for width, _height in source_sizes)
    max_h = max(height for _width, height in source_sizes)
    target_w, target_h = canvas_size
    return min(target_w / max(1, max_w), target_h / max(1, max_h))
