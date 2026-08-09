"""Semantic naming: preflight enforcement, per-icon status, truthful metadata."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from services.config_store import AppSettings
from services.extraction_types import NAMING_FAILED, NAMING_NAMED, NAMING_NOT_REQUESTED
from services.extractor import IconExtractor
from services.semantic_naming import SemanticPreflightError
from services.vision import PreflightResult


@pytest.fixture
def sheet(tmp_path) -> Path:
    """A two-icon sheet: dark squares on white, readable without any binaries."""
    canvas = np.full((120, 260, 3), 255, dtype=np.uint8)
    canvas[30:90, 30:90] = 0
    canvas[30:90, 170:230] = 0
    path = tmp_path / "sheet.png"
    Image.fromarray(canvas).save(path)
    return path


def _settings(**overrides) -> AppSettings:
    base = AppSettings(provider="ollama", semantic_naming=True, ollama_model="qwen2.5vl:3b")
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _ready(**kwargs) -> PreflightResult:
    return PreflightResult(True, "ollama", "ready", model="qwen2.5vl:3b", **kwargs)


def _not_ready(message: str = "Ollama is not reachable at http://127.0.0.1:9.") -> PreflightResult:
    return PreflightResult(False, "ollama", message)


def _client(*responses):
    """A vision client whose calls return, or raise, the given items in order."""
    client = MagicMock()
    client.identify_icon.side_effect = list(responses)
    return client


def _run(sheet: Path, out: Path, settings: AppSettings, preflight, client=None, **kwargs):
    patches = [patch("services.semantic_naming.preflight", return_value=preflight)]
    if client is not None:
        patches.append(patch("services.semantic_naming.build_vision_client", return_value=client))
    with patches[0]:
        if client is None:
            return IconExtractor(settings).extract(sheet, out, {"png"}, **kwargs)
        with patches[1]:
            return IconExtractor(settings).extract(sheet, out, {"png"}, **kwargs)


def _metadata(out: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((out / "metadata").glob("*.json"))]


# --- preflight is enforced before any work ------------------------------

def test_unreachable_backend_aborts_the_run(sheet, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(SemanticPreflightError, match="not reachable"):
        _run(sheet, out, _settings(), _not_ready())


def test_abort_happens_before_any_icon_is_written(sheet, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(SemanticPreflightError):
        _run(sheet, out, _settings(), _not_ready())
    assert not (out / "png").exists()
    assert not (out / "metadata").exists()


def test_abort_carries_the_actionable_provider_message(sheet, tmp_path):
    message = "Model 'qwen2.5vl:3b' is not pulled. Run the first-run setup."
    with pytest.raises(SemanticPreflightError) as excinfo:
        _run(sheet, tmp_path / "out", _settings(), _not_ready(message))
    assert excinfo.value.result.message == message
    assert message in str(excinfo.value)


def test_allow_unnamed_downgrades_instead_of_aborting(sheet, tmp_path):
    out = tmp_path / "out"
    result = _run(sheet, out, _settings(), _not_ready(), allow_unnamed=True)

    assert len(result.icons) == 2
    assert [icon.stem for icon in result.icons] == ["icon_001", "icon_002"]
    assert all(icon.naming_status == NAMING_NOT_REQUESTED for icon in result.icons)
    assert result.naming.named == 0
    assert any("not reachable" in warning for warning in result.warnings)


def test_downgraded_run_does_not_claim_a_model_was_used(sheet, tmp_path):
    out = tmp_path / "out"
    _run(sheet, out, _settings(), _not_ready(), allow_unnamed=True)
    for payload in _metadata(out):
        assert payload["model_used"] is None
        assert payload["pipeline"] == "classical_cv"


def test_geometry_provider_never_runs_preflight(sheet, tmp_path):
    settings = AppSettings(provider="geometry", semantic_naming=True)
    with patch("services.semantic_naming.preflight") as check:
        result = IconExtractor(settings).extract(sheet, tmp_path / "out", {"png"})
    check.assert_not_called()
    assert result.naming.requested is False
    assert all(icon.naming_status == NAMING_NOT_REQUESTED for icon in result.icons)


def test_semantic_naming_disabled_skips_preflight(sheet, tmp_path):
    with patch("services.semantic_naming.preflight") as check:
        IconExtractor(_settings(semantic_naming=False)).extract(sheet, tmp_path / "out", {"png"})
    check.assert_not_called()


# --- successful naming ----------------------------------------------------

def test_successful_naming_renames_every_output(sheet, tmp_path):
    out = tmp_path / "out"
    client = _client(
        {"label": "lightbulb", "tags": ["idea"], "confidence": 0.9},
        {"label": "folder", "tags": ["files"], "confidence": 0.8},
    )
    result = _run(sheet, out, _settings(), _ready(), client)

    assert [icon.stem for icon in result.icons] == ["lightbulb", "folder"]
    assert all(icon.naming_status == NAMING_NAMED for icon in result.icons)
    assert result.naming.named == 2 and result.naming.failed == 0
    assert {p.name for p in (out / "png").iterdir()} == {"lightbulb.png", "folder.png"}


def test_named_icons_record_the_model_that_named_them(sheet, tmp_path):
    out = tmp_path / "out"
    client = _client({"label": "heart"}, {"label": "star"})
    _run(sheet, out, _settings(), _ready(), client)
    for payload in _metadata(out):
        assert payload["model_used"] == "qwen2.5vl:3b"
        assert payload["pipeline"] == "classical_cv + ollama_labeling"
        assert payload["naming_status"] == NAMING_NAMED


def test_duplicate_labels_are_disambiguated(sheet, tmp_path):
    out = tmp_path / "out"
    client = _client({"label": "star"}, {"label": "star"})
    result = _run(sheet, out, _settings(), _ready(), client)
    assert [icon.stem for icon in result.icons] == ["star", "star-02"]
    assert len({p.name for p in (out / "png").iterdir()}) == 2


# --- partial failure ------------------------------------------------------

def test_a_failed_icon_keeps_its_generic_name_and_is_counted(sheet, tmp_path):
    out = tmp_path / "out"
    client = _client({"label": "lightbulb"}, RuntimeError("read timed out"))
    result = _run(sheet, out, _settings(), _ready(), client)

    first, second = result.icons
    assert first.stem == "lightbulb" and first.naming_status == NAMING_NAMED
    assert second.stem == "icon_002" and second.naming_status == NAMING_FAILED
    assert "read timed out" in second.naming_error
    assert result.naming.named == 1 and result.naming.failed == 1


def test_partial_failure_completes_with_a_warning(sheet, tmp_path):
    out = tmp_path / "out"
    client = _client({"label": "lightbulb"}, RuntimeError("boom"))
    result = _run(sheet, out, _settings(), _ready(), client)
    assert result.warnings
    warning = result.warnings[0]
    assert "1 of 2" in warning and "boom" in warning


def test_a_failed_icon_does_not_claim_a_model_was_used(sheet, tmp_path):
    out = tmp_path / "out"
    client = _client({"label": "lightbulb"}, RuntimeError("boom"))
    _run(sheet, out, _settings(), _ready(), client)

    by_stem = {payload["stem"]: payload for payload in _metadata(out)}
    assert by_stem["lightbulb"]["model_used"] == "qwen2.5vl:3b"
    assert by_stem["lightbulb"]["pipeline"] == "classical_cv + ollama_labeling"
    assert by_stem["icon_002"]["model_used"] is None
    assert by_stem["icon_002"]["pipeline"] == "classical_cv"
    assert by_stem["icon_002"]["naming_error"] == "boom"


def test_every_icon_records_what_was_requested(sheet, tmp_path):
    """Requested provider/model is always visible, even when naming failed."""
    out = tmp_path / "out"
    client = _client(RuntimeError("boom"), RuntimeError("boom"))
    _run(sheet, out, _settings(), _ready(), client)
    for payload in _metadata(out):
        assert payload["requested_provider"] == "ollama"
        assert payload["requested_model"] == "qwen2.5vl:3b"
        assert payload["model_used"] is None


def test_all_icons_failing_still_completes(sheet, tmp_path):
    out = tmp_path / "out"
    client = _client(RuntimeError("a"), RuntimeError("b"))
    result = _run(sheet, out, _settings(), _ready(), client)
    assert result.naming.named == 0 and result.naming.failed == 2
    assert len(list((out / "png").iterdir())) == 2


def test_an_empty_label_falls_back_to_an_indexed_name(sheet, tmp_path):
    out = tmp_path / "out"
    client = _client({"label": ""}, {"label": "   "})
    result = _run(sheet, out, _settings(), _ready(), client)
    assert [icon.stem for icon in result.icons] == ["icon-001", "icon-002"]
    assert all(icon.naming_status == NAMING_NAMED for icon in result.icons)


# --- summary --------------------------------------------------------------

def test_summary_reports_the_requested_backend(sheet, tmp_path):
    client = _client({"label": "a"}, {"label": "b"})
    result = _run(sheet, tmp_path / "out", _settings(), _ready(), client)
    assert result.naming.requested is True
    assert result.naming.provider == "ollama"
    assert result.naming.model == "qwen2.5vl:3b"


def test_summary_collects_distinct_error_reasons(sheet, tmp_path):
    client = _client(RuntimeError("read timed out"), RuntimeError("read timed out"))
    result = _run(sheet, tmp_path / "out", _settings(), _ready(), client)
    assert result.naming.errors == ("read timed out",)


def test_geometry_run_has_no_warnings(sheet, tmp_path):
    settings = AppSettings(provider="geometry")
    result = IconExtractor(settings).extract(sheet, tmp_path / "out", {"png"})
    assert result.warnings == ()
    assert result.naming.named == 0


# --- rename safety --------------------------------------------------------

def test_a_rename_failure_leaves_the_icon_whole(sheet, tmp_path):
    """A half-renamed icon would split its formats across two names."""
    out = tmp_path / "out"
    client = _client({"label": "heart"}, {"label": "star"})
    real_rename = Path.rename
    calls = {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk went away")
        return real_rename(self, target)

    with patch.object(Path, "rename", flaky):
        result = _run(sheet, out, _settings(), _ready(), client)

    first, second = result.icons
    assert first.naming_status == NAMING_FAILED
    assert "could not rename" in first.naming_error
    assert first.stem == "icon_001"
    assert first.outputs["png"].exists()  # the original file is still there
    assert second.stem == "star"  # a later icon still names normally


def test_rollback_restores_files_renamed_before_the_failure(tmp_path):
    from services.semantic_naming import _rename_outputs

    paths = {}
    for fmt, suffix in (("png", ".png"), ("jpg", ".jpg")):
        path = tmp_path / f"icon_001{suffix}"
        path.write_bytes(b"data")
        paths[fmt] = path

    real_rename = Path.rename
    calls = {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] == 2:  # fail on the second format, after the first moved
            raise OSError("nope")
        return real_rename(self, target)

    with patch.object(Path, "rename", flaky), pytest.raises(OSError):
        _rename_outputs(paths, "heart")

    assert all(path.exists() for path in paths.values())
    assert not (tmp_path / "heart.png").exists()


# --- summary formatting ---------------------------------------------------

def test_summary_describes_a_clean_run():
    from services.extraction_types import NamingSummary

    summary = NamingSummary(requested=True, provider="ollama", model="qwen", named=3)
    assert summary.describe() == "Semantic naming: 3 named via ollama/qwen."
    assert summary.attempted == 3


def test_summary_describes_a_partial_run():
    from services.extraction_types import NamingSummary

    summary = NamingSummary(requested=True, provider="ollama", model="qwen", named=2, failed=1)
    assert "2 named, 1 failed" in summary.describe()


def test_summary_describes_a_downgraded_run():
    from services.extraction_types import NamingSummary

    assert "not requested" in NamingSummary().describe()
    requested_but_idle = NamingSummary(requested=True, provider="ollama", model="qwen")
    assert "no icon was named" in requested_but_idle.describe()


# --- CLI wiring -----------------------------------------------------------

def test_cli_exposes_allow_unnamed():
    import shapearator as cli

    assert cli.build_parser().parse_args(["in.png"]).allow_unnamed is False
    assert cli.build_parser().parse_args(["in.png", "--allow-unnamed"]).allow_unnamed is True
