"""Exported artifacts must not disclose the local filesystem.

Icons and their metadata get shipped; an absolute path embedded in one leaks
the user's account name and directory layout to whoever receives the file.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from services.config_store import AppSettings
from services.extractor import IconExtractor
from services.metadata_paths import portable_output_path, scrub_local_paths
from services.vision import PreflightResult


def _sheet(path: Path, blobs: int = 2) -> Path:
    canvas = np.full((120, 80 + 140 * blobs, 3), 255, dtype=np.uint8)
    for index in range(blobs):
        left = 40 + index * 140
        canvas[30:90, left:left + 60] = 0
    Image.fromarray(canvas).save(path)
    return path


def _run(sheet: Path, out: Path, formats=("png",), **kwargs):
    return IconExtractor(AppSettings()).extract(sheet, out, set(formats), **kwargs)


def _payloads(out: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "metadata").glob("*.json"))]


# --- the source path ------------------------------------------------------

def test_source_file_is_recorded_by_name_only(tmp_path):
    out = tmp_path / "exports"
    sheet = _sheet(tmp_path / "my_sheet.png")
    _run(sheet, out)

    for payload in _payloads(out):
        assert payload["source_file"] == "my_sheet.png"
        assert "/" not in payload["source_file"]


def test_exported_paths_are_relative_to_the_output_folder(tmp_path):
    out = tmp_path / "exports"
    _run(_sheet(tmp_path / "s.png"), out, formats=("png", "jpg"))

    for payload in _payloads(out):
        assert payload["formats"]["png"].startswith("png/")
        assert payload["formats"]["jpg"].startswith("jpg/")
        for recorded in payload["formats"].values():
            assert not Path(recorded).is_absolute()
            assert (out / recorded).exists()  # still resolvable from the output folder


# --- nothing anywhere in the exported tree leaks --------------------------

def test_no_exported_file_contains_an_absolute_path(tmp_path):
    out = tmp_path / "exports"
    _run(_sheet(tmp_path / "s.png"), out, formats=("png",))

    leaked = []
    for path in (out / "metadata").rglob("*"):
        if path.is_file() and str(tmp_path) in path.read_text(encoding="utf-8"):
            leaked.append(path.name)
    assert leaked == []


def test_the_manifest_does_not_carry_the_account_name(tmp_path):
    """The manifest is local bookkeeping, but it ships if the folder is zipped."""
    out = tmp_path / "exports"
    _run(_sheet(tmp_path / "s.png"), out)

    manifest = (out / ".shapearator-manifest.json").read_text(encoding="utf-8")
    assert str(Path.home()) not in manifest


def test_embedded_svg_metadata_contains_no_absolute_path(tmp_path):
    out = tmp_path / "exports"
    icon_svg = out / "svg" / "icon_001.svg"
    with patch("services.extractor.vectorize_png_crop", side_effect=_fake_trace):
        _run(_sheet(tmp_path / "s.png", blobs=1), out, formats=("svg",))

    text = icon_svg.read_text(encoding="utf-8")
    assert "<metadata" in text  # the block is present...
    assert str(tmp_path) not in text  # ...but says nothing about this machine
    assert str(Path.home()) not in text


def _fake_trace(png_path: Path, svg_path: Path) -> None:
    """Stand in for potrace so the test needs no external binary."""
    svg_path.write_text(
        '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 10 10"><path d="M0,0 L10,10"/></svg>',
        encoding="utf-8",
    )


# --- error text is scrubbed too -------------------------------------------

def test_a_naming_error_does_not_carry_a_home_path(tmp_path):
    out = tmp_path / "exports"
    settings = AppSettings(provider="ollama", semantic_naming=True, ollama_model="qwen2.5vl:3b")
    client = MagicMock()
    client.identify_icon.side_effect = OSError(
        f"cannot open {Path.home()}/secret/project/icon_001.png"
    )

    with (
        patch("services.semantic_naming.preflight", return_value=PreflightResult(True, "ollama", "ok", model="m")),
        patch("services.semantic_naming.build_vision_client", return_value=client),
    ):
        IconExtractor(settings).extract(_sheet(tmp_path / "s.png", 1), out, {"png"})

    error = _payloads(out)[0]["naming_error"]
    assert error is not None
    assert str(Path.home()) not in error
    assert "~/" in error  # the shape of the message survives


# --- the helpers ----------------------------------------------------------

def test_scrub_replaces_the_home_prefix():
    text = f"cannot open {Path.home()}/work/sheet.png"
    assert scrub_local_paths(text) == "cannot open ~/work/sheet.png"


def test_scrub_leaves_unrelated_text_alone():
    assert scrub_local_paths("read timed out") == "read timed out"
    assert scrub_local_paths("connect to http://127.0.0.1:11434 failed") == (
        "connect to http://127.0.0.1:11434 failed"
    )


def test_scrub_passes_through_none():
    assert scrub_local_paths(None) is None


def test_portable_path_is_relative_when_inside_the_output_dir(tmp_path):
    assert portable_output_path(tmp_path / "png" / "a.png", tmp_path) == "png/a.png"


def test_portable_path_falls_back_to_the_file_name_when_outside(tmp_path):
    # Should never happen, but a name is still safe to publish; a path is not.
    outside = tmp_path.parent / "elsewhere" / "a.png"
    assert portable_output_path(outside, tmp_path) == "a.png"


def test_portable_path_uses_forward_slashes(tmp_path):
    assert "\\" not in portable_output_path(tmp_path / "svg" / "deep" / "a.svg", tmp_path)
