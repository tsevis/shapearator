"""Choosing a bitmap to preview for an extracted icon.

The renderer is injected, so none of this needs Inkscape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from services.preview_cache import preview_cache_name, resolve_preview_path


@dataclass
class FakeIcon:
    stem: str = "icon_001"
    preview_path: Path | None = None
    outputs: dict = field(default_factory=dict)


def written(calls):
    """A renderer that records its calls and writes a plausible file."""

    def render(svg_path: Path, out_path: Path) -> None:
        calls.append((svg_path, out_path))
        out_path.write_bytes(b"rendered")

    return render


def exploding(_svg_path, _out_path):
    raise RuntimeError("inkscape is not on PATH")


def never_called(_svg_path, _out_path):
    raise AssertionError("the renderer should not have run")


# --- an existing bitmap wins ---------------------------------------------

def test_an_exported_bitmap_is_used_as_is(tmp_path):
    bitmap = tmp_path / "icon_001.png"
    bitmap.write_bytes(b"png")
    icon = FakeIcon(preview_path=bitmap)

    assert resolve_preview_path(icon, tmp_path, render=never_called) == bitmap


def test_a_preview_path_pointing_at_nothing_is_not_trusted(tmp_path):
    svg = tmp_path / "icon_001.svg"
    svg.write_text("<svg/>")
    icon = FakeIcon(preview_path=tmp_path / "deleted.png", outputs={"svg": svg})

    calls = []
    result = resolve_preview_path(icon, tmp_path, render=written(calls))

    assert result == tmp_path / preview_cache_name("icon_001")
    assert len(calls) == 1


# --- rendering from SVG ---------------------------------------------------

def test_an_svg_only_icon_gets_a_preview_rendered(tmp_path):
    svg = tmp_path / "icon_001.svg"
    svg.write_text("<svg/>")
    cache = tmp_path / "cache"
    cache.mkdir()
    icon = FakeIcon(outputs={"svg": svg})

    calls = []
    result = resolve_preview_path(icon, cache, render=written(calls))

    assert result == cache / "icon_001_preview.png"
    assert result.exists()
    assert calls == [(svg, result)]


def test_a_cached_preview_is_not_rendered_twice(tmp_path):
    svg = tmp_path / "icon_001.svg"
    svg.write_text("<svg/>")
    cached = tmp_path / preview_cache_name("icon_001")
    cached.write_bytes(b"already rendered")
    icon = FakeIcon(outputs={"svg": svg})

    assert resolve_preview_path(icon, tmp_path, render=never_called) == cached


def test_each_icon_gets_its_own_cache_entry(tmp_path):
    assert preview_cache_name("heart") != preview_cache_name("lightbulb")


# --- nothing to show ------------------------------------------------------

def test_an_icon_with_no_bitmap_and_no_svg_has_no_preview(tmp_path):
    assert resolve_preview_path(FakeIcon(), tmp_path, render=never_called) is None


def test_an_svg_output_that_is_missing_on_disk_has_no_preview(tmp_path):
    icon = FakeIcon(outputs={"svg": tmp_path / "absent.svg"})
    assert resolve_preview_path(icon, tmp_path, render=never_called) is None


def test_a_failing_renderer_is_not_fatal(tmp_path):
    """The export already succeeded; the preview pane is a convenience."""
    svg = tmp_path / "icon_001.svg"
    svg.write_text("<svg/>")
    icon = FakeIcon(outputs={"svg": svg})

    assert resolve_preview_path(icon, tmp_path, render=exploding) is None


def test_a_renderer_that_writes_nothing_reports_no_preview(tmp_path):
    svg = tmp_path / "icon_001.svg"
    svg.write_text("<svg/>")
    icon = FakeIcon(outputs={"svg": svg})

    assert resolve_preview_path(icon, tmp_path, render=lambda *_: None) is None
