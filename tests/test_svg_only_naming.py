"""Semantic naming for SVG-only exports, via a throwaway label preview."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from services.config_store import AppSettings
from services.extraction_types import NAMING_FAILED, NAMING_NAMED, ExtractedIcon
from services.semantic_naming import apply_semantic_names


def _settings() -> AppSettings:
    return AppSettings(provider="ollama", semantic_naming=True, ollama_model="qwen2.5vl:3b")


def _svg_icon(tmp_path: Path, index: int = 1) -> ExtractedIcon:
    """An icon exported as SVG only -- no bitmap, so no preview."""
    svg_dir = tmp_path / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    svg_path = svg_dir / f"icon_{index:03d}.svg"
    svg_path.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"/>', encoding="utf-8")
    return ExtractedIcon(
        index=index,
        stem=f"icon_{index:03d}",
        outputs={"svg": svg_path},
        preview_path=None,
        canvas_size=(512, 512),
        source_size=(10, 10),
        source_bounds=(0, 0, 10, 10),
    )


def _client(*responses):
    client = MagicMock()
    client.identify_icon.side_effect = list(responses)
    return client


def _render_stub(calls: list[Path]):
    """Stand in for Inkscape: record the request and write a token PNG."""

    def render(svg_path: Path, output_path: Path) -> None:
        calls.append(output_path)
        output_path.write_bytes(b"\x89PNG\r\n\x1a\nrendered")

    return render


def _apply(icons, client, render=None, calls=None):
    calls = [] if calls is None else calls
    with (
        patch("services.semantic_naming.build_vision_client", return_value=client),
        patch("services.semantic_naming.export_svg_to_png", side_effect=render or _render_stub(calls)),
    ):
        return apply_semantic_names(_settings(), icons)


# --- the gap the audit found ----------------------------------------------

def test_an_svg_only_icon_can_now_be_named(tmp_path):
    icons, summary, warnings = _apply([_svg_icon(tmp_path)], _client({"label": "lightbulb"}))

    assert icons[0].naming_status == NAMING_NAMED
    assert icons[0].stem == "lightbulb"
    assert summary.named == 1 and summary.failed == 0
    assert warnings == ()


def test_the_svg_file_itself_is_renamed(tmp_path):
    icons, _summary, _warnings = _apply([_svg_icon(tmp_path)], _client({"label": "folder"}))

    assert icons[0].outputs["svg"].name == "folder.svg"
    assert icons[0].outputs["svg"].exists()
    assert not (tmp_path / "svg" / "icon_001.svg").exists()


def test_a_preview_is_rendered_from_the_exported_svg(tmp_path):
    icon = _svg_icon(tmp_path)
    calls: list[Path] = []
    _apply([icon], _client({"label": "heart"}), calls=calls)

    assert len(calls) == 1
    assert calls[0].suffix == ".png"


def test_the_model_is_shown_the_rendered_preview(tmp_path):
    seen: dict[str, object] = {}

    def capture(_model, image_path: Path):
        # Inspect during the call: the preview is gone by the time we return.
        seen["suffix"] = image_path.suffix
        seen["bytes"] = image_path.read_bytes()
        return {"label": "heart"}

    client = MagicMock()
    client.identify_icon.side_effect = capture
    _apply([_svg_icon(tmp_path)], client)

    assert seen["suffix"] == ".png"
    assert seen["bytes"].startswith(b"\x89PNG")


# --- the preview must not survive the run ---------------------------------

def test_no_preview_is_left_behind(tmp_path):
    _apply([_svg_icon(tmp_path)], _client({"label": "heart"}))

    leftovers = [p for p in tmp_path.rglob("*.png")]
    assert leftovers == []


def test_the_temporary_preview_does_not_become_the_icon_preview(tmp_path):
    """A dangling preview_path would break the GUI's preview pane."""
    icons, _summary, _warnings = _apply([_svg_icon(tmp_path)], _client({"label": "heart"}))
    assert icons[0].preview_path is None


def test_outputs_gain_no_bitmap_entry(tmp_path):
    icons, _summary, _warnings = _apply([_svg_icon(tmp_path)], _client({"label": "heart"}))
    assert set(icons[0].outputs) == {"svg"}


def test_previews_are_cleaned_up_even_when_naming_fails(tmp_path):
    _apply([_svg_icon(tmp_path)], _client(RuntimeError("model exploded")))
    assert [p for p in tmp_path.rglob("*.png")] == []


# --- failure handling -----------------------------------------------------

def test_a_render_failure_is_reported_per_icon(tmp_path):
    def boom(_svg_path, _output_path):
        raise RuntimeError("inkscape not found")

    icons, summary, _warnings = _apply([_svg_icon(tmp_path)], _client({"label": "heart"}), render=boom)

    assert icons[0].naming_status == NAMING_FAILED
    assert "inkscape not found" in icons[0].naming_error
    assert icons[0].stem == "icon_001"
    assert summary.failed == 1


def test_an_icon_with_neither_preview_nor_svg_says_so(tmp_path):
    icon = ExtractedIcon(
        index=1,
        stem="icon_001",
        outputs={},
        preview_path=None,
        canvas_size=(512, 512),
        source_size=(10, 10),
        source_bounds=(0, 0, 10, 10),
    )
    icons, summary, _warnings = _apply([icon], _client({"label": "heart"}))

    assert icons[0].naming_status == NAMING_FAILED
    assert "no image" in icons[0].naming_error.lower()
    assert summary.failed == 1


def test_a_missing_svg_file_is_reported(tmp_path):
    icon = _svg_icon(tmp_path)
    icon.outputs["svg"].unlink()
    icons, _summary, _warnings = _apply([icon], _client({"label": "heart"}))
    assert icons[0].naming_status == NAMING_FAILED


# --- an existing bitmap is still preferred --------------------------------

def test_an_existing_bitmap_preview_is_used_without_rendering(tmp_path):
    preview = tmp_path / "png" / "icon_001.png"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"\x89PNG\r\n\x1a\nexisting")
    icon = ExtractedIcon(
        index=1,
        stem="icon_001",
        outputs={"png": preview},
        preview_path=preview,
        canvas_size=(512, 512),
        source_size=(10, 10),
        source_bounds=(0, 0, 10, 10),
    )
    calls: list[Path] = []
    icons, _summary, _warnings = _apply([icon], _client({"label": "star"}), calls=calls)

    assert calls == []  # no render was needed
    assert icons[0].naming_status == NAMING_NAMED
    assert icons[0].preview_path.name == "star.png"


def test_mixed_icons_each_take_the_right_route(tmp_path):
    bitmap_preview = tmp_path / "png" / "icon_002.png"
    bitmap_preview.parent.mkdir(parents=True)
    bitmap_preview.write_bytes(b"\x89PNG\r\n\x1a\nexisting")
    bitmap_icon = ExtractedIcon(
        index=2,
        stem="icon_002",
        outputs={"png": bitmap_preview},
        preview_path=bitmap_preview,
        canvas_size=(512, 512),
        source_size=(10, 10),
        source_bounds=(0, 0, 10, 10),
    )
    calls: list[Path] = []
    icons, summary, _warnings = _apply(
        [_svg_icon(tmp_path), bitmap_icon], _client({"label": "a"}, {"label": "b"}), calls=calls
    )

    assert len(calls) == 1  # only the SVG-only icon needed a render
    assert summary.named == 2
    assert [icon.stem for icon in icons] == ["a", "b"]


# --- the extractor no longer demands a bitmap format ----------------------

def test_svg_only_formats_are_accepted_for_a_semantic_run(tmp_path):
    """Nothing in validation should require a bitmap when naming is on."""
    import shapearator as cli

    sheet = tmp_path / "sheet.svg"
    sheet.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"/>', encoding="utf-8")
    settings = _settings()
    cli.validate_settings(settings, sheet, {"svg"})  # must not raise
