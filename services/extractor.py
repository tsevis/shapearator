from __future__ import annotations

from datetime import datetime, timezone
import math
import json
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from .config_store import AppSettings
from .vision import active_vision_model, build_vision_client, semantic_naming_enabled


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
APP_VERSION = "0.3.2"
DEFAULT_BG_RGBA = (255, 255, 255, 255)


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


@dataclass
class ExtractedIcon:
    index: int
    stem: str
    outputs: dict[str, Path]
    preview_path: Path | None
    canvas_size: tuple[int, int]
    source_size: tuple[int, int]
    source_bounds: tuple[int, int, int, int]
    semantic_label: str | None = None
    semantic_tags: list[str] | None = None
    semantic_confidence: float | None = None
    vector_mode: str | None = None
    metadata_path: Path | None = None


@dataclass
class ExtractionResult:
    input_path: Path
    output_dir: Path
    icons: list[ExtractedIcon]
    provider_summary: str


@dataclass
class ExtractionProgress:
    phase: str
    current: int
    total: int
    message: str

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.current / self.total))


class IconExtractor:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def extract(
        self,
        input_path: Path,
        output_dir: Path,
        formats: set[str],
        progress_callback: Callable[[ExtractionProgress], None] | None = None,
    ) -> ExtractionResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._emit_progress(progress_callback, "prepare", 0, 1, "Loading source sheet")
        suffix = input_path.suffix.lower()
        if suffix == ".png":
            icons = self._extract_from_png(input_path, output_dir, formats, progress_callback)
        elif suffix == ".svg":
            icons = self._extract_from_svg(input_path, output_dir, formats, progress_callback)
        else:
            raise RuntimeError("Supported inputs are .png and .svg")

        if semantic_naming_enabled(self.settings) and active_vision_model(self.settings):
            self._emit_progress(progress_callback, "naming", 0, max(1, len(icons)), "Naming icons with local model")
            self._apply_semantic_names(icons, progress_callback)

        self._emit_progress(progress_callback, "metadata", 0, max(1, len(icons)), "Writing metadata")
        self._write_metadata_files(icons, output_dir, input_path, progress_callback)

        provider_summary = {
            "geometry": "Geometry-first local extractor",
            "ollama": f"Ollama local + {self.settings.ollama_model}",
            "llamacpp": f"llama.cpp local + {self.settings.llamacpp_model or 'loaded model'}",
            "directory": f"Directory model catalog + {self.settings.local_model_name or 'no active adapter'}",
        }.get(self.settings.provider, "Geometry-first local extractor")

        return ExtractionResult(
            input_path=input_path,
            output_dir=output_dir,
            icons=icons,
            provider_summary=provider_summary,
        )

    def _emit_progress(
        self,
        callback: Callable[[ExtractionProgress], None] | None,
        phase: str,
        current: int,
        total: int,
        message: str,
    ) -> None:
        if callback is not None:
            callback(ExtractionProgress(phase=phase, current=current, total=total, message=message))

    def _apply_semantic_names(
        self,
        icons: list[ExtractedIcon],
        progress_callback: Callable[[ExtractionProgress], None] | None,
    ) -> None:
        client = build_vision_client(self.settings)
        model = active_vision_model(self.settings)
        used_names: dict[str, int] = {}
        for index, icon in enumerate(icons, start=1):
            if icon.preview_path is None or not icon.preview_path.exists():
                continue
            try:
                semantic = client.identify_icon(model, icon.preview_path)
                raw = semantic.get("label", "")
                stem = slugify(raw) or f"icon-{icon.index:03d}"
                icon.semantic_label = stem
                icon.semantic_tags = semantic.get("tags", [])
                icon.semantic_confidence = semantic.get("confidence")
            except Exception:
                continue
            if stem in used_names:
                used_names[stem] += 1
                stem = f"{stem}-{used_names[stem]:02d}"
            else:
                used_names[stem] = 1
            new_outputs: dict[str, Path] = {}
            for fmt, old_path in icon.outputs.items():
                new_path = old_path.with_name(f"{stem}{old_path.suffix.lower()}")
                old_path.rename(new_path)
                new_outputs[fmt] = new_path
            icon.stem = stem
            icon.outputs = new_outputs
            icon.preview_path = new_outputs.get("png") or new_outputs.get("jpg") or new_outputs.get("tiff") or icon.preview_path
            self._emit_progress(progress_callback, "naming", index, len(icons), f"Naming icon {index} of {len(icons)}")

    def _write_metadata_files(
        self,
        icons: list[ExtractedIcon],
        output_dir: Path,
        input_path: Path,
        progress_callback: Callable[[ExtractionProgress], None] | None,
    ) -> None:
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        exported_at = datetime.now(timezone.utc).isoformat()
        for index, icon in enumerate(icons, start=1):
            dominant_color, palette = extract_icon_palette(icon)
            payload = {
                "stem": icon.stem,
                "label": icon.semantic_label or icon.stem,
                "tags": icon.semantic_tags or [],
                "semantic_confidence": icon.semantic_confidence,
                "group_id": f"group-{icon.index:03d}",
                "sheet_index": icon.index,
                "source_file": str(input_path),
                "source_bounds": list(icon.source_bounds),
                "source_size": list(icon.source_size),
                "canvas_size": list(icon.canvas_size),
                "pipeline": "classical_cv" + (f" + {self.settings.provider}_labeling" if semantic_naming_enabled(self.settings) else ""),
                "provider": self.settings.provider,
                "model_used": active_vision_model(self.settings) if semantic_naming_enabled(self.settings) else None,
                "formats": {fmt: str(path) for fmt, path in sorted(icon.outputs.items())},
                "canvas_mode": self.settings.canvas_mode,
                "dominant_color": dominant_color,
                "palette": palette,
                "vector_mode": icon.vector_mode,
                "exported_at": exported_at,
                "app_version": APP_VERSION,
            }
            metadata_path = metadata_dir / f"{icon.stem}.json"
            metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
            icon.metadata_path = metadata_path
            if "svg" in icon.outputs:
                inject_svg_metadata(icon.outputs["svg"], payload)
            self._emit_progress(progress_callback, "metadata", index, len(icons), f"Writing metadata {index} of {len(icons)}")

    def _extract_from_png(
        self,
        input_path: Path,
        output_dir: Path,
        formats: set[str],
        progress_callback: Callable[[ExtractionProgress], None] | None,
    ) -> list[ExtractedIcon]:
        bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not load raster image: {input_path}")
        binary = build_foreground_mask(bgr)
        background_rgba = estimate_background_rgba(bgr)
        boxes = [tighten_box(binary, box) for box in detect_icon_boxes(binary, self.settings.min_area, self.settings.merge_gap)]
        self._emit_progress(progress_callback, "detect", len(boxes), len(boxes), f"Detected {len(boxes)} icons")
        canvas_size = self._canvas_size()
        source_sizes = [(box.w + self.settings.padding * 2, box.h + self.settings.padding * 2) for box in boxes]
        uniform_scale = compute_uniform_scale(source_sizes, canvas_size)
        icons: list[ExtractedIcon] = []
        for index, box in enumerate(boxes, start=1):
            crop_box = box.padded(self.settings.padding, bgr.shape[1], bgr.shape[0])
            crop_bgr = bgr[crop_box.y:crop_box.y2, crop_box.x:crop_box.x2].copy()
            crop_mask = binary[crop_box.y:crop_box.y2, crop_box.x:crop_box.x2]
            crop_rgba = build_transparent_crop_rgba(crop_bgr, crop_mask)
            opaque_crop_rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGBA)
            stem = f"icon_{index:03d}"
            outputs = self._write_bitmap_outputs(
                crop_rgba,
                stem,
                output_dir,
                formats,
                uniform_scale=uniform_scale,
                background_rgba=background_rgba,
                opaque_rgba=opaque_crop_rgba,
            )
            preview = outputs.get("png") or outputs.get("jpg") or outputs.get("tiff")
            icons.append(
                ExtractedIcon(
                    index=index,
                    stem=stem,
                    outputs=outputs,
                    preview_path=preview,
                    canvas_size=canvas_size,
                    source_size=(crop_rgba.shape[1], crop_rgba.shape[0]),
                    source_bounds=(crop_box.x, crop_box.y, crop_box.w, crop_box.h),
                    vector_mode=self._vector_mode_for_raster_crop(outputs, preview),
                )
            )
            self._emit_progress(progress_callback, "export", index, len(boxes), f"Exporting icon {index} of {len(boxes)}")
        return icons

    def _extract_from_svg(
        self,
        input_path: Path,
        output_dir: Path,
        formats: set[str],
        progress_callback: Callable[[ExtractionProgress], None] | None,
    ) -> list[ExtractedIcon]:
        tree = ET.parse(input_path)
        root = tree.getroot()
        view_box = parse_viewbox(root)
        ensure_element_ids(root)

        with tempfile.TemporaryDirectory(prefix="shapearator_svg_") as temp_dir:
            temp_source = Path(temp_dir) / "source.svg"
            temp_raster = Path(temp_dir) / "source.png"
            tree.write(temp_source, encoding="utf-8", xml_declaration=True)
            element_boxes = query_svg_boxes(temp_source)
            render_svg_to_png(temp_source, temp_raster)
            raster_gray = cv2.imread(str(temp_raster), cv2.IMREAD_GRAYSCALE)
            if raster_gray is None:
                raise RuntimeError(f"Could not rasterize SVG: {input_path}")
            raster_binary = build_binary_mask(raster_gray)
            raster_groups = [tighten_box(raster_binary, box) for box in detect_icon_boxes(raster_binary, self.settings.min_area, max(7, self.settings.merge_gap - 2))]

        drawable_children = [
            child
            for child in root
            if isinstance(child.tag, str) and not child.tag.endswith("defs") and child.attrib.get("id") in element_boxes
        ]
        assigned_ids: set[str] = set()
        grouped_items: list[tuple[Box, list[ET.Element]]] = []

        for raster_box in raster_groups:
            svg_group_box = svg_box_from_raster_box(raster_box, raster_gray.shape[1], raster_gray.shape[0], view_box)
            children: list[ET.Element] = []
            union_box: Box | None = None
            for child in drawable_children:
                child_id = child.attrib["id"]
                if child_id in assigned_ids:
                    continue
                child_box = element_boxes[child_id]
                if box_contains_point(svg_group_box, child_box.cx, child_box.cy, pad=3):
                    children.append(child)
                    assigned_ids.add(child_id)
                    union_box = child_box if union_box is None else union_box.union(child_box)
            if children and union_box is not None:
                grouped_items.append((union_box, children))

        for child in drawable_children:
            child_id = child.attrib["id"]
            if child_id not in assigned_ids:
                grouped_items.append((element_boxes[child_id], [child]))

        grouped_items.sort(key=lambda item: (item[0].cy, item[0].x))
        self._emit_progress(progress_callback, "detect", len(grouped_items), len(grouped_items), f"Detected {len(grouped_items)} icons")
        canvas_size = self._canvas_size()
        source_sizes = [
            (box.w + self.settings.padding * 2, box.h + self.settings.padding * 2)
            for box, _children in grouped_items
        ]
        uniform_scale = compute_uniform_scale(source_sizes, canvas_size)
        icons: list[ExtractedIcon] = []
        for index, (box, children) in enumerate(grouped_items, start=1):
            stem = f"icon_{index:03d}"
            outputs: dict[str, Path] = {}
            raw_svg_path = output_dir / "_work_svg" / f"{stem}.svg"
            wrote_intermediate_svg = False
            if "svg" in formats or any(fmt in formats for fmt in ("png", "jpg", "tiff")):
                raw_svg_path.parent.mkdir(parents=True, exist_ok=True)
                build_svg_fragment(root, children, box, self.settings.padding, raw_svg_path)
                if "svg" in formats:
                    final_svg_path = output_dir / "svg" / f"{stem}.svg"
                    final_svg_path.parent.mkdir(parents=True, exist_ok=True)
                    normalize_svg_to_canvas(
                        raw_svg_path,
                        final_svg_path,
                        canvas_size,
                        self.settings.canvas_mode,
                        uniform_scale=uniform_scale,
                    )
                    outputs["svg"] = final_svg_path
                else:
                    wrote_intermediate_svg = True

            preview: Path | None = None
            if any(fmt in formats for fmt in ("png", "jpg", "tiff")):
                temp_png = output_dir / "_work_png" / f"{stem}.png"
                temp_png.parent.mkdir(parents=True, exist_ok=True)
                export_svg_to_png(raw_svg_path, temp_png)
                bitmap_outputs = self._convert_png_to_selected_formats(
                    temp_png,
                    stem,
                    output_dir,
                    formats,
                    uniform_scale=uniform_scale,
                    background_rgba=DEFAULT_BG_RGBA,
                )
                outputs.update(bitmap_outputs)
                preview = outputs.get("png") or outputs.get("jpg") or outputs.get("tiff")
                if temp_png.exists():
                    temp_png.unlink()
            if wrote_intermediate_svg and raw_svg_path.exists():
                raw_svg_path.unlink()
            icons.append(
                ExtractedIcon(
                    index=index,
                    stem=stem,
                    outputs=outputs,
                    preview_path=preview,
                    canvas_size=canvas_size,
                    source_size=(box.w + self.settings.padding * 2, box.h + self.settings.padding * 2),
                    source_bounds=(box.x - self.settings.padding, box.y - self.settings.padding, box.w + self.settings.padding * 2, box.h + self.settings.padding * 2),
                    vector_mode=self._vector_mode_for_svg_source(outputs),
                )
            )
            self._emit_progress(progress_callback, "export", index, len(grouped_items), f"Exporting icon {index} of {len(grouped_items)}")
        png_dir = output_dir / "png"
        svg_dir = output_dir / "svg"
        work_png_dir = output_dir / "_work_png"
        work_svg_dir = output_dir / "_work_svg"
        if "png" not in formats and png_dir.exists() and not any(png_dir.iterdir()):
            png_dir.rmdir()
        if "svg" not in formats:
            if svg_dir.exists() and not any(svg_dir.iterdir()):
                svg_dir.rmdir()
        if work_png_dir.exists() and not any(work_png_dir.iterdir()):
            work_png_dir.rmdir()
        if work_svg_dir.exists() and not any(work_svg_dir.iterdir()):
            work_svg_dir.rmdir()
        return icons

    def _write_bitmap_outputs(
        self,
        rgba_crop: np.ndarray,
        stem: str,
        output_dir: Path,
        formats: set[str],
        uniform_scale: float,
        background_rgba: tuple[int, int, int, int],
        opaque_rgba: np.ndarray | None = None,
    ) -> dict[str, Path]:
        work_png_dir = output_dir / "_work_png"
        work_png_dir.mkdir(parents=True, exist_ok=True)
        png_path = work_png_dir / f"{stem}.png"
        write_rgba_crop(rgba_crop, png_path)
        opaque_png_path: Path | None = None
        if opaque_rgba is not None:
            opaque_png_path = work_png_dir / f"{stem}_opaque.png"
            write_rgba_crop(opaque_rgba, opaque_png_path)
        outputs = self._convert_png_to_selected_formats(
            png_path,
            stem,
            output_dir,
            formats,
            uniform_scale=uniform_scale,
            background_rgba=background_rgba,
            opaque_png_path=opaque_png_path,
        )
        if "svg" in formats:
            svg_dir = output_dir / "svg"
            svg_dir.mkdir(parents=True, exist_ok=True)
            svg_path = svg_dir / f"{stem}.svg"
            vector_source = outputs.get("png")
            temp_vector_source: Path | None = None
            if vector_source is None:
                source_for_vector = opaque_png_path if self.settings.bitmap_export_mode == "keep_background" and opaque_png_path is not None else png_path
                with Image.open(source_for_vector).convert("RGBA") as image:
                    rendered = compose_icon_on_canvas(
                        image,
                        self._canvas_size(),
                        self.settings.canvas_mode,
                        uniform_scale=uniform_scale,
                        background_rgba=background_rgba if self.settings.bitmap_export_mode == "keep_background" else (0, 0, 0, 0),
                    )
                    temp_vector_source = work_png_dir / f"{stem}_vector.png"
                    rendered.save(temp_vector_source)
                    vector_source = temp_vector_source
            if is_effectively_monochrome_png(vector_source):
                vectorize_png_crop(vector_source, svg_path)
            else:
                wrap_png_in_svg(vector_source, svg_path, self._canvas_size())
            outputs["svg"] = svg_path
            if temp_vector_source is not None and temp_vector_source.exists():
                temp_vector_source.unlink()
        if png_path.exists():
            png_path.unlink()
        if opaque_png_path is not None and opaque_png_path.exists():
            opaque_png_path.unlink()
        if work_png_dir.exists() and not any(work_png_dir.iterdir()):
            work_png_dir.rmdir()
        return outputs

    def _convert_png_to_selected_formats(
        self,
        png_path: Path,
        stem: str,
        output_dir: Path,
        formats: set[str],
        uniform_scale: float,
        background_rgba: tuple[int, int, int, int],
        opaque_png_path: Path | None = None,
    ) -> dict[str, Path]:
        outputs: dict[str, Path] = {}
        source_path = opaque_png_path if self.settings.bitmap_export_mode == "keep_background" and opaque_png_path is not None else png_path
        canvas_background = background_rgba if self.settings.bitmap_export_mode == "keep_background" else (0, 0, 0, 0)
        with Image.open(source_path).convert("RGBA") as image:
            rendered = compose_icon_on_canvas(
                image,
                self._canvas_size(),
                self.settings.canvas_mode,
                uniform_scale=uniform_scale,
                background_rgba=canvas_background,
            )
            if "png" in formats:
                png_dir = output_dir / "png"
                png_dir.mkdir(parents=True, exist_ok=True)
                final_png = png_dir / f"{stem}.png"
                rendered.save(final_png)
                outputs["png"] = final_png
            if "jpg" in formats:
                jpg_dir = output_dir / "jpg"
                jpg_dir.mkdir(parents=True, exist_ok=True)
                jpg_bg = background_rgba[:3] if self.settings.bitmap_export_mode == "keep_background" else (255, 255, 255)
                rgb = Image.new("RGB", rendered.size, jpg_bg)
                rgb.paste(rendered, mask=rendered.getchannel("A"))
                jpg_path = jpg_dir / f"{stem}.jpg"
                rgb.save(jpg_path, quality=95)
                outputs["jpg"] = jpg_path
            if "tiff" in formats:
                tiff_dir = output_dir / "tiff"
                tiff_dir.mkdir(parents=True, exist_ok=True)
                tiff_path = tiff_dir / f"{stem}.tiff"
                rendered.save(tiff_path)
                outputs["tiff"] = tiff_path
        return outputs

    def _canvas_size(self) -> tuple[int, int]:
        return max(1, int(self.settings.output_width)), max(1, int(self.settings.output_height))

    def _vector_mode_for_raster_crop(self, outputs: dict[str, Path], preview: Path | None) -> str | None:
        if "svg" not in outputs:
            return None
        if preview is None:
            return "traced-monochrome"
        return "traced-monochrome" if is_effectively_monochrome_png(preview) else "embedded-raster-color"

    def _vector_mode_for_svg_source(self, outputs: dict[str, Path]) -> str | None:
        return "vector-native-grouped" if "svg" in outputs else None


def slugify(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", normalized)


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise RuntimeError(f"Required binary '{name}' was not found on PATH.")
    return binary


def build_foreground_mask(bgr: np.ndarray) -> np.ndarray:
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


def sample_border_pixels(lab: np.ndarray, band: int = 12) -> np.ndarray:
    h, w = lab.shape[:2]
    band = max(1, min(band, h // 4 or 1, w // 4 or 1))
    top = lab[:band, :, :].reshape(-1, 3)
    bottom = lab[-band:, :, :].reshape(-1, 3)
    left = lab[:, :band, :].reshape(-1, 3)
    right = lab[:, -band:, :].reshape(-1, 3)
    return np.concatenate([top, bottom, left, right], axis=0)


def build_binary_mask(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)


def make_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def detect_icon_boxes(binary: np.ndarray, min_area: int, merge_gap: int) -> list[Box]:
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


def write_rgba_crop(rgba_crop: np.ndarray, output_path: Path) -> None:
    Image.fromarray(rgba_crop, mode="RGBA").save(output_path)


def compose_icon_on_canvas(
    image: Image.Image,
    canvas_size: tuple[int, int],
    mode: str,
    uniform_scale: float,
    background_rgba: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
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


def extract_icon_palette(icon: ExtractedIcon) -> tuple[str | None, list[str]]:
    if icon.preview_path is not None and icon.preview_path.exists():
        return extract_palette_from_image(icon.preview_path)
    if "svg" in icon.outputs and icon.outputs["svg"].exists():
        with tempfile.TemporaryDirectory(prefix="shapearator_palette_") as temp_dir:
            temp_png = Path(temp_dir) / "palette.png"
            try:
                export_svg_to_png(icon.outputs["svg"], temp_png)
                return extract_palette_from_image(temp_png)
            except Exception:
                return None, []
    return None, []


def extract_palette_from_image(image_path: Path, max_colors: int = 4) -> tuple[str | None, list[str]]:
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


def rgb_to_hex(rgb: np.ndarray) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def estimate_background_rgba(bgr: np.ndarray) -> tuple[int, int, int, int]:
    border = sample_border_pixels(bgr.astype(np.float32))
    if border.size == 0:
        return DEFAULT_BG_RGBA
    median_bgr = np.median(border, axis=0)
    return int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0]), 255


def build_transparent_crop_rgba(crop_bgr: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGBA)
    exterior_background = compute_exterior_background_mask(crop_mask > 0)
    rgba[:, :, 3] = np.where(exterior_background, 0, 255).astype(np.uint8)
    return rgba


def compute_exterior_background_mask(foreground_mask: np.ndarray) -> np.ndarray:
    padded_foreground = np.pad(foreground_mask.astype(bool), 1, constant_values=False)
    flood = np.where(~padded_foreground, 255, 0).astype(np.uint8)
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 128)
    exterior = flood == 128
    return exterior[1:-1, 1:-1]


def compute_uniform_scale(source_sizes: list[tuple[int, int]], canvas_size: tuple[int, int]) -> float:
    if not source_sizes:
        return 1.0
    max_w = max(width for width, _height in source_sizes)
    max_h = max(height for _width, height in source_sizes)
    target_w, target_h = canvas_size
    return min(target_w / max(1, max_w), target_h / max(1, max_h))


def vectorize_png_crop(png_path: Path, svg_path: Path) -> None:
    require_binary("potrace")
    with Image.open(png_path) as image:
        alpha = np.array(image.getchannel("A"))
        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        bmp_path = svg_path.with_suffix(".bmp")
        Image.fromarray(mask, mode="L").save(bmp_path)
    try:
        subprocess.run(["potrace", "-s", str(bmp_path), "-o", str(svg_path)], check=True, capture_output=True, text=True)
    finally:
        bmp_path.unlink(missing_ok=True)


def is_effectively_monochrome_png(png_path: Path) -> bool:
    with Image.open(png_path).convert("RGBA") as image:
        pixels = np.array(image)
    alpha = pixels[:, :, 3] > 0
    if not np.any(alpha):
        return True
    rgb = pixels[:, :, :3][alpha]
    channel_spread = np.max(np.abs(rgb[:, 0].astype(np.int16) - rgb[:, 1].astype(np.int16)))
    channel_spread = max(channel_spread, np.max(np.abs(rgb[:, 1].astype(np.int16) - rgb[:, 2].astype(np.int16))))
    return channel_spread <= 6


def wrap_png_in_svg(png_path: Path, svg_path: Path, canvas_size: tuple[int, int]) -> None:
    import base64

    target_w, target_h = canvas_size
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    data_uri = f"data:image/png;base64,{encoded}"
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": str(target_w),
            "height": str(target_h),
            "viewBox": f"0 0 {target_w} {target_h}",
        },
    )
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}image",
        {
            "width": str(target_w),
            "height": str(target_h),
            "href": data_uri,
        },
    )
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)


def get_svg_canvas_size(root: ET.Element) -> tuple[float, float]:
    view_box = root.attrib.get("viewBox")
    if view_box:
        parts = [float(part) for part in view_box.replace(",", " ").split()]
        if len(parts) == 4:
            return parts[2], parts[3]
    width = parse_svg_length(root.attrib.get("width", "0"))
    height = parse_svg_length(root.attrib.get("height", "0"))
    return max(1.0, width), max(1.0, height)


def parse_svg_length(value: str) -> float:
    cleaned = value.strip().replace("px", "")
    try:
        return float(cleaned)
    except Exception:
        return 0.0


def parse_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    view_box = root.attrib.get("viewBox")
    if not view_box:
        raise RuntimeError("SVG must include a viewBox.")
    parts = [float(part) for part in view_box.replace(",", " ").split()]
    if len(parts) != 4:
        raise RuntimeError(f"Unexpected viewBox: {view_box}")
    return tuple(parts)  # type: ignore[return-value]


def ensure_element_ids(root: ET.Element) -> None:
    counter = 1
    for child in root:
        if not isinstance(child.tag, str):
            continue
        if child.tag.endswith("defs"):
            continue
        if "id" not in child.attrib:
            child.set("id", f"shape_{counter:04d}")
            counter += 1


def query_svg_boxes(svg_path: Path) -> dict[str, Box]:
    require_binary("inkscape")
    result = subprocess.run(["inkscape", "--query-all", str(svg_path)], check=True, capture_output=True, text=True)
    boxes: dict[str, Box] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        shape_id, x, y, w, h = parts
        if shape_id == "Layer_1":
            continue
        boxes[shape_id] = Box(
            int(math.floor(float(x))),
            int(math.floor(float(y))),
            max(1, int(math.ceil(float(w)))),
            max(1, int(math.ceil(float(h)))),
        )
    return boxes


def build_svg_fragment(source_root: ET.Element, children: list[ET.Element], bounds: Box, padding: int, output_path: Path) -> None:
    fragment = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "viewBox": f"0 0 {bounds.w + padding * 2} {bounds.h + padding * 2}",
        },
    )
    if "style" in source_root.attrib:
        fragment.set("style", source_root.attrib["style"])
    dx = padding - bounds.x
    dy = padding - bounds.y
    for child in children:
        node = deepcopy(child)
        existing_transform = node.attrib.get("transform", "").strip()
        translate = f"translate({dx} {dy})"
        node.set("transform", f"{translate} {existing_transform}".strip())
        fragment.append(node)
    ET.ElementTree(fragment).write(output_path, encoding="utf-8", xml_declaration=True)


def normalize_svg_to_canvas(
    source_svg_path: Path,
    output_svg_path: Path,
    canvas_size: tuple[int, int],
    mode: str,
    uniform_scale: float,
) -> None:
    tree = ET.parse(source_svg_path)
    root = tree.getroot()
    source_w, source_h = get_svg_canvas_size(root)
    target_w, target_h = canvas_size

    scale = 1.0
    if mode == "uniform_to_largest":
        scale = uniform_scale
    elif mode == "individual_fit":
        scale = min(target_w / max(1.0, source_w), target_h / max(1.0, source_h))

    defs_children = []
    drawable_children = []
    for child in root:
        if isinstance(child.tag, str) and child.tag.endswith("defs"):
            defs_children.append(deepcopy(child))
        else:
            drawable_children.append(deepcopy(child))

    canvas_root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": str(target_w),
            "height": str(target_h),
            "viewBox": f"0 0 {target_w} {target_h}",
        },
    )
    for defs in defs_children:
        canvas_root.append(defs)

    scaled_w = source_w * scale
    scaled_h = source_h * scale
    offset_x = (target_w - scaled_w) / 2.0
    offset_y = (target_h - scaled_h) / 2.0
    group = ET.SubElement(
        canvas_root,
        f"{{{SVG_NS}}}g",
        {"transform": f"translate({offset_x:.4f} {offset_y:.4f}) scale({scale:.6f})"},
    )
    for child in drawable_children:
        group.append(child)
    ET.ElementTree(canvas_root).write(output_svg_path, encoding="utf-8", xml_declaration=True)


def inject_svg_metadata(svg_path: Path, payload: dict) -> None:
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception:
        return
    for child in list(root):
        if isinstance(child.tag, str) and child.tag.endswith("metadata"):
            root.remove(child)
    metadata = ET.Element(f"{{{SVG_NS}}}metadata")
    metadata.text = json.dumps(payload, ensure_ascii=True)
    root.insert(0, metadata)
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)


def export_svg_to_png(svg_path: Path, output_path: Path) -> None:
    require_binary("inkscape")
    subprocess.run(
        ["inkscape", str(svg_path), "--export-type=png", f"--export-filename={output_path}"],
        check=True,
        capture_output=True,
        text=True,
    )


def render_svg_to_png(svg_path: Path, output_path: Path, width: int = 2000) -> None:
    require_binary("inkscape")
    subprocess.run(
        [
            "inkscape",
            str(svg_path),
            "--export-type=png",
            f"--export-filename={output_path}",
            f"--export-width={width}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def svg_box_from_raster_box(raster_box: Box, raster_w: int, raster_h: int, view_box: tuple[float, float, float, float]) -> Box:
    vb_x, vb_y, vb_w, vb_h = view_box
    scale_x = vb_w / raster_w
    scale_y = vb_h / raster_h
    x = vb_x + raster_box.x * scale_x
    y = vb_y + raster_box.y * scale_y
    w = raster_box.w * scale_x
    h = raster_box.h * scale_y
    return Box(int(math.floor(x)), int(math.floor(y)), max(1, int(math.ceil(w))), max(1, int(math.ceil(h))))


def box_contains_point(box: Box, x: float, y: float, pad: int = 0) -> bool:
    return (box.x - pad) <= x <= (box.x2 + pad) and (box.y - pad) <= y <= (box.y2 + pad)
