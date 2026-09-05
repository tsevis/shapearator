"""Icon extraction orchestration.

Detection lives in :mod:`services.geometry`, pixel work in
:mod:`services.raster_ops`, and everything SVG in :mod:`services.svg_ops`.
This module wires them together against the user's settings.
"""
from __future__ import annotations

import json
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2

import numpy as np
from PIL import Image

from .config_store import AppSettings
from .export_commit import ExportStaging
from .extraction_types import (
    NAMING_NAMED,
    ExtractedIcon,
    ExtractionProgress,
    ExtractionResult,
    NamingSummary,
)
from .metadata_paths import portable_output_path, scrub_local_paths
from .geometry import (
    Box,
    build_binary_mask,
    build_foreground_mask,
    box_contains_point,
    compute_uniform_scale,
    detect_icon_boxes,
    tighten_box,
)
from .raster_ops import (
    DEFAULT_BG_RGBA,
    build_transparent_crop_rgba,
    compose_icon_on_canvas,
    estimate_background_rgba,
    extract_palette_from_image,
    is_effectively_monochrome_png,
    write_rgba_crop,
)
from .semantic_naming import (
    SemanticPreflightError,
    apply_semantic_names,
    check_backend_ready,
    mark_all_unnamed,
    naming_requested,
    slugify,
)
from .svg_ops import (
    build_parent_map,
    build_svg_fragment,
    ensure_element_ids,
    export_svg_to_png,
    find_icon_elements,
    inject_svg_metadata,
    is_grouped_artwork,
    normalize_svg_to_canvas,
    parse_viewbox,
    query_svg_boxes,
    render_svg_to_png,
    svg_box_from_raster_box,
    vectorize_png_crop,
    wrap_png_in_svg,
)
from .vision import active_vision_model

__all__ = [
    "APP_VERSION",
    "ExtractedIcon",
    "ExtractionProgress",
    "ExtractionResult",
    "IconExtractor",
    "SemanticPreflightError",
    "extract_icon_palette",
    "slugify",
]

APP_VERSION = "0.4.3"


def _relocate_icon(icon: ExtractedIcon, staging: ExportStaging) -> ExtractedIcon:
    """Point an icon's paths at their committed location instead of staging."""
    return replace(
        icon,
        outputs={fmt: staging.final_path(path) for fmt, path in icon.outputs.items()},
        preview_path=staging.final_path(icon.preview_path) if icon.preview_path else None,
        metadata_path=staging.final_path(icon.metadata_path) if icon.metadata_path else None,
    )


def extract_icon_palette(preview_path: Path | None, svg_path: Path | None) -> tuple[str | None, list[str]]:
    """Sample an icon's colours, rendering the SVG only if there is no bitmap."""
    if preview_path is not None and preview_path.exists():
        return extract_palette_from_image(preview_path)
    if svg_path is not None and svg_path.exists():
        with tempfile.TemporaryDirectory(prefix="shapearator_palette_") as temp_dir:
            temp_png = Path(temp_dir) / "palette.png"
            try:
                export_svg_to_png(svg_path, temp_png)
                return extract_palette_from_image(temp_png)
            except Exception:
                return None, []
    return None, []


class IconExtractor:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def extract(
        self,
        input_path: Path,
        output_dir: Path,
        formats: set[str],
        progress_callback: Callable[[ExtractionProgress], None] | None = None,
        allow_unnamed: bool = False,
    ) -> ExtractionResult:
        """Extract every icon in ``input_path`` into ``output_dir``.

        When semantic naming is on, the vision backend is checked *before* any
        pixel is written: an unreachable model raises ``SemanticPreflightError``
        in seconds rather than after a full export that silently produced
        generic names. Pass ``allow_unnamed`` to downgrade to a geometry-only
        run instead of aborting.
        """
        suffix = input_path.suffix.lower()
        if suffix not in {".png", ".svg"}:
            raise RuntimeError("Supported inputs are .png and .svg")

        self._emit_progress(progress_callback, "preflight", 0, 1, "Checking local model backend")
        readiness = check_backend_ready(self.settings, allow_unnamed)
        warnings = readiness.warnings
        if readiness.result is not None:
            self._emit_progress(progress_callback, "preflight", 1, 1, readiness.result.message)

        with ExportStaging(output_dir) as staging:
            staged = staging.staging_dir
            self._emit_progress(progress_callback, "prepare", 0, 1, "Loading source sheet")
            if suffix == ".png":
                icons = self._extract_from_png(input_path, staged, formats, progress_callback)
            else:
                icons = self._extract_from_svg(input_path, staged, formats, progress_callback)

            if readiness.proceed:
                self._emit_progress(progress_callback, "naming", 0, max(1, len(icons)), "Naming icons with local model")
                icons, naming, naming_warnings = apply_semantic_names(self.settings, icons, progress_callback)
                warnings += naming_warnings
            else:
                icons = mark_all_unnamed(icons)
                naming = NamingSummary()

            self._emit_progress(progress_callback, "metadata", 0, max(1, len(icons)), "Writing metadata")
            icons = self._write_metadata_files(icons, staged, input_path, staging, progress_callback)

            self._emit_progress(progress_callback, "commit", 0, 1, "Publishing export")
            report = staging.commit(input_path, formats, len(icons), APP_VERSION)
            icons = [_relocate_icon(icon, staging) for icon in icons]
            warnings += report.warnings
            self._emit_progress(progress_callback, "commit", 1, 1, f"Published {report.written} files")

        return ExtractionResult(
            input_path=input_path,
            output_dir=output_dir,
            icons=icons,
            provider_summary=self._provider_summary(),
            naming=naming,
            warnings=warnings,
            commit=report,
        )

    def _provider_summary(self) -> str:
        return {
            "geometry": "Geometry-first local extractor",
            "ollama": f"Ollama local + {self.settings.ollama_model}",
            "llamacpp": f"llama.cpp local + {self.settings.llamacpp_model or 'loaded model'}",
            "directory": f"Directory model catalog + {self.settings.local_model_name or 'no active adapter'}",
        }.get(self.settings.provider, "Geometry-first local extractor")

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

    def _write_metadata_files(
        self,
        icons: list[ExtractedIcon],
        output_dir: Path,
        input_path: Path,
        staging: ExportStaging,
        progress_callback: Callable[[ExtractionProgress], None] | None,
    ) -> list[ExtractedIcon]:
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        exported_at = datetime.now(timezone.utc).isoformat()
        written: list[ExtractedIcon] = []
        for index, icon in enumerate(icons, start=1):
            payload = self._build_metadata_payload(icon, input_path, exported_at, staging)
            metadata_path = metadata_dir / f"{icon.stem}.json"
            metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
            if "svg" in icon.outputs:
                inject_svg_metadata(icon.outputs["svg"], payload)
            written.append(replace(icon, metadata_path=metadata_path))
            self._emit_progress(progress_callback, "metadata", index, len(icons), f"Writing metadata {index} of {len(icons)}")
        return written

    def _build_metadata_payload(
        self, icon: ExtractedIcon, input_path: Path, exported_at: str, staging: ExportStaging
    ) -> dict:
        """Describe one icon, distinguishing what was asked for from what happened.

        ``requested_provider``/``requested_model`` record the configuration;
        ``model_used`` is populated only for an icon a model actually named, so
        a failed or skipped labeling pass can never read as a successful one.

        Exported paths are recorded where they will live after the commit, not
        where they are staged.
        """
        dominant_color, palette = extract_icon_palette(icon.preview_path, icon.outputs.get("svg"))
        was_named = icon.naming_status == NAMING_NAMED
        requested = naming_requested(self.settings)
        return {
            "stem": icon.stem,
            "label": icon.semantic_label or icon.stem,
            "tags": icon.semantic_tags or [],
            "semantic_confidence": icon.semantic_confidence,
            "group_id": f"group-{icon.index:03d}",
            "sheet_index": icon.index,
            # Name only: this file is a deliverable, and the full path would
            # disclose the account name and layout of the exporting machine.
            "source_file": input_path.name,
            "source_bounds": list(icon.source_bounds),
            "source_size": list(icon.source_size),
            "canvas_size": list(icon.canvas_size),
            "pipeline": "classical_cv" + (f" + {self.settings.provider}_labeling" if was_named else ""),
            "provider": self.settings.provider,
            "requested_provider": self.settings.provider if requested else None,
            "requested_model": active_vision_model(self.settings) if requested else None,
            "model_used": active_vision_model(self.settings) if was_named else None,
            "naming_status": icon.naming_status,
            "naming_error": scrub_local_paths(icon.naming_error),
            "formats": {
                fmt: portable_output_path(staging.final_path(path), staging.output_dir)
                for fmt, path in sorted(icon.outputs.items())
            },
            "canvas_mode": self.settings.canvas_mode,
            "dominant_color": dominant_color,
            "palette": palette,
            "vector_mode": icon.vector_mode,
            "exported_at": exported_at,
            "app_version": APP_VERSION,
        }

    # --- raster input -----------------------------------------------------

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

    # --- vector input -----------------------------------------------------

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

        parents = build_parent_map(root)
        element_boxes = self._query_element_boxes(tree)

        candidates = find_icon_elements(root, element_boxes)
        if is_grouped_artwork(candidates):
            # The artwork says which pieces belong together, so take it at its
            # word: one group per icon, no detection tuning involved.
            grouped_items = [
                (element_boxes[element.attrib["id"]], [element]) for element in candidates
            ]
        else:
            # Loose shapes, not groups -- a hand-drawn sheet where one icon is
            # several separate strokes. Those have to be clustered visually.
            raster_groups, raster_shape = self._detect_raster_groups(tree)
            grouped_items = self._group_svg_children(
                candidates, element_boxes, raster_groups, raster_shape, view_box
            )

        self._emit_progress(progress_callback, "detect", len(grouped_items), len(grouped_items), f"Detected {len(grouped_items)} icons")
        canvas_size = self._canvas_size()
        source_sizes = [
            (box.w + self.settings.padding * 2, box.h + self.settings.padding * 2)
            for box, _children in grouped_items
        ]
        uniform_scale = compute_uniform_scale(source_sizes, canvas_size)

        icons: list[ExtractedIcon] = []
        for index, (box, children) in enumerate(grouped_items, start=1):
            icons.append(
                self._export_svg_icon(
                    index=index,
                    box=box,
                    children=children,
                    root=root,
                    output_dir=output_dir,
                    formats=formats,
                    canvas_size=canvas_size,
                    uniform_scale=uniform_scale,
                    parents=parents,
                )
            )
            self._emit_progress(progress_callback, "export", index, len(grouped_items), f"Exporting icon {index} of {len(grouped_items)}")

        self._prune_empty_dirs(output_dir, formats)
        return icons

    def _query_element_boxes(self, tree: ET.ElementTree) -> dict[str, Box]:
        """Ask Inkscape for the rendered bounds of every identified element."""
        with tempfile.TemporaryDirectory(prefix="shapearator_svg_") as temp_dir:
            temp_source = Path(temp_dir) / "source.svg"
            tree.write(temp_source, encoding="utf-8", xml_declaration=True)
            return query_svg_boxes(temp_source)

    def _detect_raster_groups(self, tree: ET.ElementTree) -> tuple[list[Box], tuple[int, int]]:
        """Detect visual groupings on a rasterized proof of the whole sheet."""
        with tempfile.TemporaryDirectory(prefix="shapearator_svg_") as temp_dir:
            temp_source = Path(temp_dir) / "source.svg"
            temp_raster = Path(temp_dir) / "source.png"
            tree.write(temp_source, encoding="utf-8", xml_declaration=True)
            render_svg_to_png(temp_source, temp_raster)
            raster_gray = cv2.imread(str(temp_raster), cv2.IMREAD_GRAYSCALE)
            if raster_gray is None:
                raise RuntimeError("Could not rasterize SVG for grouping.")
            raster_binary = build_binary_mask(raster_gray)
            raster_groups = [
                tighten_box(raster_binary, box)
                for box in detect_icon_boxes(raster_binary, self.settings.min_area, max(7, self.settings.merge_gap - 2))
            ]
            return raster_groups, (raster_gray.shape[1], raster_gray.shape[0])

    def _group_svg_children(
        self,
        drawable_children: list[ET.Element],
        element_boxes: dict[str, Box],
        raster_groups: list[Box],
        raster_shape: tuple[int, int],
        view_box: tuple[float, float, float, float],
    ) -> list[tuple[Box, list[ET.Element]]]:
        """Assign each drawable element to the raster group that contains it.

        ``drawable_children`` is the level `find_icon_elements` settled on, so
        this works the same whether the shapes sit at the root or inside a
        layer group.
        """
        raster_w, raster_h = raster_shape
        assigned_ids: set[str] = set()
        grouped_items: list[tuple[Box, list[ET.Element]]] = []

        for raster_box in raster_groups:
            svg_group_box = svg_box_from_raster_box(raster_box, raster_w, raster_h, view_box)
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

        # Anything the raster pass missed still deserves its own icon.
        for child in drawable_children:
            child_id = child.attrib["id"]
            if child_id not in assigned_ids:
                grouped_items.append((element_boxes[child_id], [child]))

        grouped_items.sort(key=lambda item: (item[0].cy, item[0].x))
        return grouped_items

    def _export_svg_icon(
        self,
        index: int,
        box: Box,
        children: list[ET.Element],
        root: ET.Element,
        output_dir: Path,
        formats: set[str],
        canvas_size: tuple[int, int],
        uniform_scale: float,
        parents: dict[ET.Element, ET.Element] | None = None,
    ) -> ExtractedIcon:
        stem = f"icon_{index:03d}"
        outputs: dict[str, Path] = {}
        raw_svg_path = output_dir / "_work_svg" / f"{stem}.svg"
        wants_bitmap = any(fmt in formats for fmt in ("png", "jpg", "tiff"))
        wrote_intermediate_svg = False

        if "svg" in formats or wants_bitmap:
            raw_svg_path.parent.mkdir(parents=True, exist_ok=True)
            build_svg_fragment(root, children, box, self.settings.padding, raw_svg_path, parents)
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
        if wants_bitmap:
            temp_png = output_dir / "_work_png" / f"{stem}.png"
            temp_png.parent.mkdir(parents=True, exist_ok=True)
            export_svg_to_png(raw_svg_path, temp_png)
            outputs.update(
                self._convert_png_to_selected_formats(
                    temp_png,
                    stem,
                    output_dir,
                    formats,
                    uniform_scale=uniform_scale,
                    background_rgba=DEFAULT_BG_RGBA,
                )
            )
            preview = outputs.get("png") or outputs.get("jpg") or outputs.get("tiff")
            temp_png.unlink(missing_ok=True)
        if wrote_intermediate_svg:
            raw_svg_path.unlink(missing_ok=True)

        padded_w = box.w + self.settings.padding * 2
        padded_h = box.h + self.settings.padding * 2
        return ExtractedIcon(
            index=index,
            stem=stem,
            outputs=outputs,
            preview_path=preview,
            canvas_size=canvas_size,
            source_size=(padded_w, padded_h),
            source_bounds=(box.x - self.settings.padding, box.y - self.settings.padding, padded_w, padded_h),
            vector_mode=self._vector_mode_for_svg_source(outputs),
        )

    def _prune_empty_dirs(self, output_dir: Path, formats: set[str]) -> None:
        candidates = ["_work_png", "_work_svg"]
        if "png" not in formats:
            candidates.append("png")
        if "svg" not in formats:
            candidates.append("svg")
        for name in candidates:
            directory = output_dir / name
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()

    # --- shared bitmap writing -------------------------------------------

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
            outputs["svg"] = self._vectorize_bitmap_icon(
                stem,
                output_dir,
                outputs.get("png"),
                png_path,
                opaque_png_path,
                work_png_dir,
                uniform_scale=uniform_scale,
                background_rgba=background_rgba,
            )

        png_path.unlink(missing_ok=True)
        if opaque_png_path is not None:
            opaque_png_path.unlink(missing_ok=True)
        if work_png_dir.exists() and not any(work_png_dir.iterdir()):
            work_png_dir.rmdir()
        return outputs

    def _vectorize_bitmap_icon(
        self,
        stem: str,
        output_dir: Path,
        exported_png: Path | None,
        png_path: Path,
        opaque_png_path: Path | None,
        work_png_dir: Path,
        uniform_scale: float,
        background_rgba: tuple[int, int, int, int],
    ) -> Path:
        """Trace or embed a raster crop as SVG, rendering a source if needed."""
        svg_dir = output_dir / "svg"
        svg_dir.mkdir(parents=True, exist_ok=True)
        svg_path = svg_dir / f"{stem}.svg"

        vector_source = exported_png
        temp_vector_source: Path | None = None
        if vector_source is None:
            # PNG was not among the chosen formats, so compose one just to trace.
            keep_background = self.settings.bitmap_export_mode == "keep_background"
            source_for_vector = opaque_png_path if keep_background and opaque_png_path is not None else png_path
            with Image.open(source_for_vector).convert("RGBA") as image:
                rendered = compose_icon_on_canvas(
                    image,
                    self._canvas_size(),
                    self.settings.canvas_mode,
                    uniform_scale=uniform_scale,
                    background_rgba=background_rgba if keep_background else (0, 0, 0, 0),
                )
                temp_vector_source = work_png_dir / f"{stem}_vector.png"
                rendered.save(temp_vector_source)
                vector_source = temp_vector_source

        if is_effectively_monochrome_png(vector_source):
            vectorize_png_crop(vector_source, svg_path)
        else:
            wrap_png_in_svg(vector_source, svg_path, self._canvas_size())

        if temp_vector_source is not None:
            temp_vector_source.unlink(missing_ok=True)
        return svg_path

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
        keep_background = self.settings.bitmap_export_mode == "keep_background"
        source_path = opaque_png_path if keep_background and opaque_png_path is not None else png_path
        canvas_background = background_rgba if keep_background else (0, 0, 0, 0)
        with Image.open(source_path).convert("RGBA") as image:
            rendered = compose_icon_on_canvas(
                image,
                self._canvas_size(),
                self.settings.canvas_mode,
                uniform_scale=uniform_scale,
                background_rgba=canvas_background,
            )
            if "png" in formats:
                outputs["png"] = self._save_render(rendered, output_dir / "png" / f"{stem}.png")
            if "jpg" in formats:
                jpg_bg = background_rgba[:3] if keep_background else (255, 255, 255)
                rgb = Image.new("RGB", rendered.size, jpg_bg)
                rgb.paste(rendered, mask=rendered.getchannel("A"))
                jpg_path = output_dir / "jpg" / f"{stem}.jpg"
                jpg_path.parent.mkdir(parents=True, exist_ok=True)
                rgb.save(jpg_path, quality=95)
                outputs["jpg"] = jpg_path
            if "tiff" in formats:
                outputs["tiff"] = self._save_render(rendered, output_dir / "tiff" / f"{stem}.tiff")
        return outputs

    @staticmethod
    def _save_render(rendered: Image.Image, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(path)
        return path

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
