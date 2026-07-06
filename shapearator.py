#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from services.config_store import AppSettings, ConfigStore
from services.extractor import ExtractionProgress, ExtractionResult, IconExtractor
from services.vision import is_local_url, preflight


DETECTION_PRESETS = {
    "Balanced": {"padding": 12, "min_area": 200, "merge_gap": 13},
    "Tiny Details": {"padding": 8, "min_area": 70, "merge_gap": 9},
    "Loose Sketches": {"padding": 16, "min_area": 140, "merge_gap": 19},
    "Bold Shapes": {"padding": 14, "min_area": 320, "merge_gap": 15},
}

CANVAS_MODE_CHOICES = ["original", "uniform_to_largest", "individual_fit"]
BITMAP_EXPORT_MODE_CHOICES = ["keep_background", "transparent_preserve_interior"]
FORMAT_CHOICES = ["png", "jpg", "tiff", "svg"]
PROVIDER_CHOICES = ["geometry", "ollama", "llamacpp", "directory"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shapearator CLI: extract hand-drawn icons from PNG or SVG sheets.",
    )
    parser.add_argument("input", type=Path, nargs="?", default=None, help="Path to the source PNG or SVG sheet.")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Download the default local vision model(s) for the available backend(s), then exit.",
    )
    parser.add_argument(
        "--setup-all",
        action="store_true",
        help="Download every recommended vision model for the available backend(s), then exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out"),
        help="Directory where extracted assets will be written. Default: ./out",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=FORMAT_CHOICES,
        default=None,
        help="Export formats to write. Default comes from config or falls back to png svg.",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=None,
        help="Final canvas width in pixels for every export.",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=None,
        help="Final canvas height in pixels for every export.",
    )
    parser.add_argument(
        "--canvas-mode",
        choices=CANVAS_MODE_CHOICES,
        default=None,
        help="Canvas scaling behavior for exported icons.",
    )
    parser.add_argument(
        "--bitmap-export-mode",
        choices=BITMAP_EXPORT_MODE_CHOICES,
        default=None,
        help="Bitmap export behavior: preserve source background or export transparency.",
    )
    parser.add_argument(
        "--detection-preset",
        choices=list(DETECTION_PRESETS.keys()),
        default=None,
        help="Detection preset to seed padding, min-area, and merge-gap.",
    )
    parser.add_argument("--padding", type=int, default=None, help="Extra padding around detected icon crops.")
    parser.add_argument("--min-area", type=int, default=None, help="Minimum connected-component area to keep.")
    parser.add_argument("--merge-gap", type=int, default=None, help="Morphological merge distance for reconnecting marks.")
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default=None,
        help="Active provider mode: geometry, local Ollama, local llama.cpp, or local model directory catalog.",
    )
    parser.add_argument("--ollama-url", default=None, help="Local Ollama endpoint URL.")
    parser.add_argument("--ollama-model", default=None, help="Ollama model name used for semantic naming.")
    parser.add_argument("--llamacpp-url", default=None, help="Local llama.cpp server endpoint URL.")
    parser.add_argument("--llamacpp-model", default=None, help="llama.cpp model name used for semantic naming.")
    parser.add_argument("--local-model-root", default=None, help="Directory used to discover local models.")
    parser.add_argument("--local-model-name", default=None, help="Selected local directory model name.")
    parser.add_argument(
        "--semantic-naming",
        dest="semantic_naming",
        action="store_true",
        help="Enable local-model-based semantic filenames and metadata when supported.",
    )
    parser.add_argument(
        "--no-semantic-naming",
        dest="semantic_naming",
        action="store_false",
        help="Disable semantic filenames and metadata enrichment.",
    )
    parser.set_defaults(semantic_naming=None)
    parser.add_argument(
        "--use-config",
        action="store_true",
        help="Load defaults from config/settings.json before applying CLI overrides.",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Persist the resolved settings back to config/settings.json after validation.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def load_base_settings(use_config: bool) -> AppSettings:
    if use_config:
        return ConfigStore(Path("config") / "settings.json").load()
    return AppSettings()


def apply_detection_preset(settings: AppSettings, preset_name: str | None) -> None:
    if not preset_name:
        return
    preset = DETECTION_PRESETS[preset_name]
    settings.padding = preset["padding"]
    settings.min_area = preset["min_area"]
    settings.merge_gap = preset["merge_gap"]


def apply_cli_overrides(settings: AppSettings, args: argparse.Namespace, input_path: Path, output_dir: Path) -> AppSettings:
    settings.last_input_path = str(input_path)
    settings.last_output_dir = str(output_dir)
    apply_detection_preset(settings, args.detection_preset)

    override_map = {
        "provider": args.provider,
        "ollama_url": args.ollama_url,
        "ollama_model": args.ollama_model,
        "llamacpp_url": args.llamacpp_url,
        "llamacpp_model": args.llamacpp_model,
        "local_model_root": args.local_model_root,
        "local_model_name": args.local_model_name,
        "output_width": args.output_width,
        "output_height": args.output_height,
        "canvas_mode": args.canvas_mode,
        "bitmap_export_mode": args.bitmap_export_mode,
        "padding": args.padding,
        "min_area": args.min_area,
        "merge_gap": args.merge_gap,
    }
    for field_name, value in override_map.items():
        if value is not None:
            setattr(settings, field_name, value)

    if args.formats is not None:
        settings.default_formats = list(args.formats)
    if args.semantic_naming is not None:
        settings.semantic_naming = args.semantic_naming
    return settings


def validate_settings(settings: AppSettings, input_path: Path, formats: set[str]) -> None:
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if input_path.suffix.lower() not in {".png", ".svg"}:
        raise SystemExit("Supported inputs are .png and .svg")
    if not formats:
        raise SystemExit("At least one export format must be selected.")
    if settings.output_width <= 0 or settings.output_height <= 0:
        raise SystemExit("Canvas width and height must be positive integers.")
    if settings.provider == "ollama" and not is_local_url(settings.ollama_url):
        raise SystemExit("Ollama provider requires a local endpoint such as http://127.0.0.1:11434.")
    if settings.provider == "llamacpp" and not is_local_url(settings.llamacpp_url):
        raise SystemExit("llama.cpp provider requires a local endpoint such as http://127.0.0.1:8080.")


def describe_detection_origin(args: argparse.Namespace, settings: AppSettings) -> str:
    if args.detection_preset:
        return (
            f"preset={args.detection_preset} "
            f"(padding={settings.padding}, min_area={settings.min_area}, merge_gap={settings.merge_gap})"
        )
    return f"manual (padding={settings.padding}, min_area={settings.min_area}, merge_gap={settings.merge_gap})"


def provider_notes(settings: AppSettings) -> list[str]:
    notes: list[str] = []
    if settings.provider == "ollama":
        notes.append(f"Ollama endpoint: {settings.ollama_url}")
        notes.append(f"Ollama model: {settings.ollama_model or 'not selected'}")
        notes.append(
            "Semantic naming: enabled" if settings.semantic_naming else "Semantic naming: disabled"
        )
    elif settings.provider == "llamacpp":
        notes.append(f"llama.cpp endpoint: {settings.llamacpp_url}")
        notes.append(f"llama.cpp model: {settings.llamacpp_model or 'loaded server model'}")
        notes.append(
            "Semantic naming: enabled" if settings.semantic_naming else "Semantic naming: disabled"
        )
    elif settings.provider == "directory":
        notes.append(f"Model directory: {settings.local_model_root or 'not set'}")
        notes.append(f"Selected directory model: {settings.local_model_name or 'not selected'}")
        notes.append("Directory provider currently acts as a local catalog/configuration mode, not a direct inference adapter.")
        notes.append(
            "Semantic naming remains unavailable unless provider=ollama or provider=llamacpp with a local model selected."
        )
    else:
        notes.append("Semantic naming: unavailable in geometry-only mode.")
    return notes


def print_run_header(settings: AppSettings, input_path: Path, output_dir: Path, formats: set[str], args: argparse.Namespace) -> None:
    print("Shapearator CLI")
    print(f"Input: {input_path.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Formats: {', '.join(sorted(formats))}")
    print(f"Canvas: {settings.output_width}x{settings.output_height} px")
    print(f"Canvas mode: {settings.canvas_mode}")
    print(f"Bitmap export mode: {settings.bitmap_export_mode}")
    print(f"Provider: {settings.provider}")
    print(f"Detection: {describe_detection_origin(args, settings)}")
    for note in provider_notes(settings):
        print(note)
    if settings.semantic_naming and settings.provider in {"ollama", "llamacpp"}:
        result = preflight(settings)
        marker = "ready" if result.ok else "NOT READY"
        print(f"Preflight [{marker}]: {result.message}")
    print("")


def progress_printer(progress: ExtractionProgress) -> None:
    total = max(progress.total, 1)
    print(f"[{progress.phase}] {progress.current}/{total} - {progress.message}")


_last_setup_print = {"t": 0.0}


def setup_progress_printer(progress) -> None:
    now = time.time()
    if progress.phase in {"resolve", "done", "error"} or now - _last_setup_print["t"] > 1.0:
        _last_setup_print["t"] = now
        if progress.total:
            mb_done = progress.completed / 1e6
            mb_total = progress.total / 1e6
            print(f"  [{progress.phase}] {progress.message} {mb_done:.0f}/{mb_total:.0f} MB ({progress.fraction * 100:.0f}%)")
        else:
            print(f"  [{progress.phase}] {progress.message}")


def run_headless_setup(args: argparse.Namespace) -> int:
    from services import first_run as fr

    settings = load_base_settings(args.use_config)
    if args.ollama_url:
        settings.ollama_url = args.ollama_url
    if args.llamacpp_url:
        settings.llamacpp_url = args.llamacpp_url
    if args.local_model_root:
        settings.models_root = args.local_model_root

    status = fr.detect_backends(settings)
    candidates = fr.build_candidates(settings, status)
    print("Shapearator model setup")
    print(f"Ollama reachable: {status.ollama_reachable} | llama.cpp available: {status.llamacpp_binary}")

    if args.setup_all:
        chosen = [c for c in candidates if not c.installed]
    else:
        chosen = [c for c in candidates if c.default_selected and not c.installed]

    already = [c for c in candidates if c.installed]
    for candidate in already:
        print(f"Already installed: {candidate.spec.display_name} ({candidate.backend})")

    if not chosen:
        print("Nothing to download.")
        fr.mark_setup_complete({"skipped": True})
        return 0

    for candidate in chosen:
        size = f"~{candidate.approx_gb:.1f} GB" if candidate.approx_gb else "small"
        print(f"Installing {candidate.spec.display_name} via {candidate.backend} ({size})…")
        fr.install_candidate(settings, candidate, setup_progress_printer)

    first = chosen[0]
    fr.apply_active_model(settings, first)
    fr.mark_setup_complete({"installed": [c.spec.key for c in chosen]})
    ConfigStore(Path("config") / "settings.json").save(settings)
    print(f"Setup complete. Active provider: {settings.provider}.")
    return 0


def maybe_save_config(settings: AppSettings, args: argparse.Namespace) -> None:
    if not args.save_config:
        return
    ConfigStore(Path("config") / "settings.json").save(settings)
    print(f"Saved resolved settings to {(Path('config') / 'settings.json').resolve()}")


def print_completion(result: ExtractionResult) -> None:
    print("")
    print(f"Provider summary: {result.provider_summary}")
    print(f"Completed: extracted {len(result.icons)} icons")
    print(f"Output written to: {result.output_dir.resolve()}")
    metadata_dir = result.output_dir / "metadata"
    if metadata_dir.exists():
        print(f"Metadata directory: {metadata_dir.resolve()}")


def main() -> int:
    args = parse_args()

    if args.setup or args.setup_all:
        return run_headless_setup(args)

    if args.input is None:
        raise SystemExit("An input .png or .svg is required (or run with --setup to download models).")

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    settings = load_base_settings(args.use_config)
    settings = apply_cli_overrides(settings, args, input_path, output_dir)
    formats = set(settings.default_formats)
    validate_settings(settings, input_path, formats)
    maybe_save_config(settings, args)
    print_run_header(settings, input_path, output_dir, formats, args)

    result = IconExtractor(settings).extract(
        input_path=input_path,
        output_dir=output_dir,
        formats=formats,
        progress_callback=progress_printer,
    )
    print_completion(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
