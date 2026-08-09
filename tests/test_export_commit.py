"""Staged, manifest-tracked exports: no stale assets, no collateral deletion."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from services.config_store import AppSettings
from services.export_commit import (
    MANIFEST_NAME,
    STAGING_NAME,
    read_manifest,
)
from services.extractor import IconExtractor


def _sheet(path: Path, blobs: int) -> Path:
    """A sheet with ``blobs`` well-separated dark squares on white."""
    canvas = np.full((120, 80 + 140 * blobs, 3), 255, dtype=np.uint8)
    for index in range(blobs):
        left = 40 + index * 140
        canvas[30:90, left:left + 60] = 0
    Image.fromarray(canvas).save(path)
    return path


def _run(sheet: Path, out: Path, formats=("png",), **kwargs):
    return IconExtractor(AppSettings()).extract(sheet, out, set(formats), **kwargs)


def _names(directory: Path) -> set[str]:
    return {p.name for p in directory.iterdir()} if directory.exists() else set()


# --- the audit's reproduction --------------------------------------------

def test_a_smaller_second_run_leaves_no_stale_assets(tmp_path):
    out = tmp_path / "exports"
    big = _run(_sheet(tmp_path / "big.png", 3), out)
    assert len(big.icons) == 3

    small = _run(_sheet(tmp_path / "small.png", 1), out)
    assert len(small.icons) == 1
    assert _names(out / "png") == {"icon_001.png"}
    assert _names(out / "metadata") == {"icon_001.json"}


def test_a_format_dropped_from_the_second_run_is_cleaned_up(tmp_path):
    out = tmp_path / "exports"
    _run(_sheet(tmp_path / "a.png", 2), out, formats=("png", "jpg"))
    assert (out / "jpg").exists()

    _run(_sheet(tmp_path / "b.png", 2), out, formats=("png",))
    assert not (out / "jpg").exists()
    assert _names(out / "png") == {"icon_001.png", "icon_002.png"}


def test_repeated_runs_do_not_accumulate(tmp_path):
    out = tmp_path / "exports"
    for _ in range(3):
        _run(_sheet(tmp_path / "s.png", 2), out)
    assert len(_names(out / "png")) == 2


# --- nothing outside the manifest is ever deleted ------------------------

def test_a_first_run_deletes_nothing(tmp_path):
    out = tmp_path / "exports"
    out.mkdir()
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    (out / "png").mkdir()
    (out / "png" / "hand_made.png").write_bytes(b"mine too")

    _run(_sheet(tmp_path / "s.png", 1), out)

    assert (out / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert (out / "png" / "hand_made.png").read_bytes() == b"mine too"
    assert (out / "png" / "icon_001.png").exists()


def test_a_first_run_reports_what_it_left_alone(tmp_path):
    out = tmp_path / "exports"
    (out / "png").mkdir(parents=True)
    (out / "png" / "hand_made.png").write_bytes(b"mine")

    result = _run(_sheet(tmp_path / "s.png", 1), out)
    assert any("hand_made.png" in warning or "1 pre-existing" in warning for warning in result.warnings)


def test_the_commit_report_counts_what_it_touched(tmp_path):
    out = tmp_path / "exports"
    first = _run(_sheet(tmp_path / "a.png", 3), out)
    assert first.commit.written == 6  # 3 png + 3 metadata
    assert first.commit.replaced == 0

    (out / "png" / "mine.png").write_bytes(b"mine")
    second = _run(_sheet(tmp_path / "b.png", 2), out)
    assert second.commit.written == 4
    assert second.commit.replaced == 6
    assert second.commit.preserved == 1  # mine.png, never tracked


def test_unrelated_files_survive_a_replacing_run(tmp_path):
    out = tmp_path / "exports"
    _run(_sheet(tmp_path / "a.png", 2), out)
    (out / "png" / "keep_me.png").write_bytes(b"mine")
    (out / "README.md").write_text("mine", encoding="utf-8")

    _run(_sheet(tmp_path / "b.png", 1), out)

    assert (out / "png" / "keep_me.png").exists()
    assert (out / "README.md").exists()
    assert (out / "png" / "icon_002.png").exists() is False  # stale managed file gone


def test_a_manifest_path_escaping_the_output_dir_is_ignored(tmp_path):
    out = tmp_path / "exports"
    _run(_sheet(tmp_path / "a.png", 1), out)
    victim = tmp_path / "outside.txt"
    victim.write_text("do not delete", encoding="utf-8")

    manifest_path = out / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].append("../outside.txt")
    manifest["files"].append("/etc/passwd")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _run(_sheet(tmp_path / "b.png", 1), out)
    assert victim.read_text(encoding="utf-8") == "do not delete"


# --- the manifest itself --------------------------------------------------

def test_a_manifest_records_the_run(tmp_path):
    out = tmp_path / "exports"
    sheet = _sheet(tmp_path / "s.png", 2)
    _run(sheet, out, formats=("png", "jpg"))

    manifest = read_manifest(out)
    assert manifest is not None
    assert manifest["input_name"] == sheet.name
    assert str(Path.home()) not in manifest["input"]
    assert sorted(manifest["formats"]) == ["jpg", "png"]
    assert manifest["icon_count"] == 2
    assert manifest["manifest_version"] == 1
    assert manifest["created_at"]


def test_a_manifest_lists_every_file_it_wrote(tmp_path):
    out = tmp_path / "exports"
    _run(_sheet(tmp_path / "s.png", 2), out, formats=("png",))

    listed = set(read_manifest(out)["files"])
    assert listed == {
        "png/icon_001.png",
        "png/icon_002.png",
        "metadata/icon_001.json",
        "metadata/icon_002.json",
    }
    for relative in listed:
        assert (out / relative).exists()


def test_an_unreadable_manifest_is_treated_as_absent(tmp_path):
    out = tmp_path / "exports"
    _run(_sheet(tmp_path / "a.png", 2), out)
    (out / MANIFEST_NAME).write_text("{not json", encoding="utf-8")

    result = _run(_sheet(tmp_path / "b.png", 1), out)  # must not raise
    assert (out / "png" / "icon_001.png").exists()
    assert result.icons


# --- failure leaves the previous export intact ---------------------------

def test_a_failed_run_preserves_the_previous_export(tmp_path):
    out = tmp_path / "exports"
    _run(_sheet(tmp_path / "a.png", 3), out)
    before = {p.name: p.read_bytes() for p in (out / "png").iterdir()}

    with patch.object(IconExtractor, "_write_metadata_files", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            _run(_sheet(tmp_path / "b.png", 1), out)

    assert {p.name: p.read_bytes() for p in (out / "png").iterdir()} == before
    assert read_manifest(out)["icon_count"] == 3


def test_a_failed_run_removes_its_staging_directory(tmp_path):
    out = tmp_path / "exports"
    with patch.object(IconExtractor, "_write_metadata_files", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            _run(_sheet(tmp_path / "a.png", 1), out)
    assert not (out / STAGING_NAME).exists()


def test_a_failed_first_run_writes_no_partial_output(tmp_path):
    out = tmp_path / "exports"
    with patch.object(IconExtractor, "_write_metadata_files", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            _run(_sheet(tmp_path / "a.png", 2), out)
    assert not (out / "png").exists()
    assert not (out / MANIFEST_NAME).exists()


def test_a_commit_failure_restores_the_previous_export(tmp_path):
    out = tmp_path / "exports"
    _run(_sheet(tmp_path / "a.png", 3), out)
    before = {p.name: p.read_bytes() for p in (out / "png").iterdir()}
    before_manifest = read_manifest(out)

    real_replace = Path.replace
    calls = {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] == 3:  # fail midway through moving new files into place
            raise OSError("disk full")
        return real_replace(self, target)

    with patch.object(Path, "replace", flaky):
        with pytest.raises(OSError, match="disk full"):
            _run(_sheet(tmp_path / "b.png", 2), out)

    assert {p.name: p.read_bytes() for p in (out / "png").iterdir()} == before
    assert read_manifest(out) == before_manifest


# --- housekeeping ---------------------------------------------------------

def test_a_successful_run_removes_its_staging_directory(tmp_path):
    out = tmp_path / "exports"
    _run(_sheet(tmp_path / "s.png", 1), out)
    assert not (out / STAGING_NAME).exists()


def test_intermediate_work_directories_are_never_committed(tmp_path):
    out = tmp_path / "exports"
    _run(_sheet(tmp_path / "s.png", 1), out, formats=("png",))
    assert not (out / "_work_png").exists()
    assert not (out / "_work_svg").exists()


def test_returned_paths_point_at_the_committed_location(tmp_path):
    out = tmp_path / "exports"
    result = _run(_sheet(tmp_path / "s.png", 2), out)
    for icon in result.icons:
        assert STAGING_NAME not in str(icon.outputs["png"])
        assert icon.outputs["png"].exists()
        assert icon.metadata_path is not None and icon.metadata_path.exists()
        assert icon.preview_path is not None and icon.preview_path.exists()


def test_metadata_records_committed_paths_not_staged_ones(tmp_path):
    out = tmp_path / "exports"
    _run(_sheet(tmp_path / "s.png", 1), out)
    payload = json.loads((out / "metadata" / "icon_001.json").read_text(encoding="utf-8"))
    recorded = payload["formats"]["png"]
    # Recorded relative to the output folder, and pointing at the committed file.
    assert STAGING_NAME not in recorded
    assert (out / recorded).exists()
