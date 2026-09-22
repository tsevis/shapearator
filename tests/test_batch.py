"""Extracting a folder of sheets in one run.

`IconExtractor` handles one sheet. A folder of them adds three problems it
does not have: which files count as sheets, where each one's icons go without
overwriting a neighbour's, and what happens when sheet four of nine is
corrupt. These cover those three and nothing else -- the extraction itself is
already covered by `test_extractor_svg_pipeline.py`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import services.batch as batch
from services.extraction_types import ExtractedIcon, ExtractionResult
from services.semantic_naming import SemanticPreflightError
from services.settings_schema import AppSettings
from services.vision import PreflightResult


def write_sheets(folder: Path, names: list[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    return folder


def fake_result(input_path: Path, output_dir: Path, icons: int = 1) -> ExtractionResult:
    return ExtractionResult(
        input_path=input_path,
        output_dir=output_dir,
        icons=[
            ExtractedIcon(index=i, stem=f"icon_{i:03d}", outputs={}, preview_path=None,
                          canvas_size=(512, 512), source_size=(10, 10),
                          source_bounds=(0, 0, 10, 10))
            for i in range(1, icons + 1)
        ],
        provider_summary="test",
    )


# --- which files count as sheets ------------------------------------------

def test_only_supported_sheets_are_picked_up(tmp_path):
    folder = write_sheets(tmp_path / "in", ["a.svg", "b.png", "notes.txt", "art.ai"])
    (folder / ".DS_Store").write_text("junk")
    assert [p.name for p in batch.find_sheets(folder)] == ["a.svg", "b.png"]


def test_sheets_are_returned_in_name_order(tmp_path):
    folder = write_sheets(tmp_path / "in", ["Many3.svg", "Many1.svg", "Many2.svg"])
    assert [p.name for p in batch.find_sheets(folder)] == ["Many1.svg", "Many2.svg", "Many3.svg"]


def test_subfolders_are_not_descended_into(tmp_path):
    """An earlier run's output folder must not become the next run's input."""
    folder = write_sheets(tmp_path / "in", ["a.svg"])
    write_sheets(folder / "previous_export" / "svg", ["icon_001.svg"])
    assert [p.name for p in batch.find_sheets(folder)] == ["a.svg"]


def test_a_suffix_in_capitals_still_counts(tmp_path):
    folder = write_sheets(tmp_path / "in", ["A.SVG", "B.Png"])
    assert [p.name for p in batch.find_sheets(folder)] == ["A.SVG", "B.Png"]


# --- where each sheet's icons go ------------------------------------------

def test_each_sheet_gets_its_own_folder_named_after_it(tmp_path):
    sheets = [tmp_path / "Many1.svg", tmp_path / "Many2.svg"]
    plan = batch.plan_output_dirs(sheets, tmp_path / "out")
    assert [d.name for d in plan.values()] == ["Many1", "Many2"]


def test_two_sheets_sharing_a_stem_do_not_share_a_folder(tmp_path):
    """logo.png and logo.svg are different artwork; icon_001 would collide."""
    sheets = [tmp_path / "logo.png", tmp_path / "logo.svg"]
    plan = batch.plan_output_dirs(sheets, tmp_path / "out")
    names = [d.name for d in plan.values()]
    assert len(set(names)) == 2, names
    assert names[0] == "logo"


# --- one bad sheet must not end the run -----------------------------------

def test_a_sheet_that_fails_does_not_stop_the_others(tmp_path, monkeypatch):
    folder = write_sheets(tmp_path / "in", ["a.svg", "b.svg", "c.svg"])

    class Extractor:
        def __init__(self, settings):
            pass

        def extract(self, input_path, output_dir, formats, progress_callback=None,
                    allow_unnamed=False, uniform_scale=None):
            if input_path.name == "b.svg":
                raise RuntimeError("could not rasterize")
            return fake_result(input_path, output_dir)

    monkeypatch.setattr(batch, "IconExtractor", Extractor)
    outcome = batch.extract_folder(AppSettings(), folder, tmp_path / "out", {"svg"})

    assert [s.input_path.name for s in outcome.sheets] == ["a.svg", "b.svg", "c.svg"]
    assert [s.failed for s in outcome.sheets] == [False, True, False]
    assert outcome.icons and len(outcome.icons) == 2


def test_the_failure_is_reported_with_the_sheet_that_caused_it(tmp_path, monkeypatch):
    folder = write_sheets(tmp_path / "in", ["good.svg", "broken.svg"])

    class Extractor:
        def __init__(self, settings):
            pass

        def extract(self, input_path, output_dir, formats, progress_callback=None,
                    allow_unnamed=False, uniform_scale=None):
            if input_path.name == "broken.svg":
                raise RuntimeError("could not rasterize")
            return fake_result(input_path, output_dir)

    monkeypatch.setattr(batch, "IconExtractor", Extractor)
    outcome = batch.extract_folder(AppSettings(), folder, tmp_path / "out", {"svg"})

    assert len(outcome.warnings) == 1
    assert "broken.svg" in outcome.warnings[0]
    assert "could not rasterize" in outcome.warnings[0]


def test_a_run_where_every_sheet_fails_says_so(tmp_path, monkeypatch):
    """Silence here would read as "nothing to extract" rather than "all broken"."""
    folder = write_sheets(tmp_path / "in", ["a.svg", "b.svg"])

    class Extractor:
        def __init__(self, settings):
            pass

        def extract(self, *args, **kwargs):
            raise RuntimeError("nope")

    monkeypatch.setattr(batch, "IconExtractor", Extractor)
    outcome = batch.extract_folder(AppSettings(), folder, tmp_path / "out", {"svg"})

    assert outcome.icons == []
    assert outcome.failed_count == 2


def test_an_empty_folder_is_refused_before_anything_is_written(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    with pytest.raises(batch.NoSheetsFound):
        batch.extract_folder(AppSettings(), folder, tmp_path / "out", {"svg"})
    assert not (tmp_path / "out").exists()


# --- progress covers the whole folder -------------------------------------

def test_progress_counts_sheets_not_icons(tmp_path, monkeypatch):
    folder = write_sheets(tmp_path / "in", ["a.svg", "b.svg"])

    class Extractor:
        def __init__(self, settings):
            pass

        def extract(self, input_path, output_dir, formats, progress_callback=None,
                    allow_unnamed=False, uniform_scale=None):
            return fake_result(input_path, output_dir)

    monkeypatch.setattr(batch, "IconExtractor", Extractor)
    seen = []
    batch.extract_folder(AppSettings(), folder, tmp_path / "out", {"svg"},
                         progress_callback=seen.append)

    sheet_events = [p for p in seen if p.phase == "sheet"]
    assert [(p.current, p.total) for p in sheet_events] == [(1, 2), (2, 2)]
    assert "a.svg" in sheet_events[0].message


def test_a_sheet_that_failed_leaves_no_empty_folder_behind(tmp_path, monkeypatch):
    """An empty folder named after a sheet reads as "extracted, found nothing"."""
    folder = write_sheets(tmp_path / "in", ["good.svg", "broken.svg"])

    class Extractor:
        def __init__(self, settings):
            pass

        def extract(self, input_path, output_dir, formats, progress_callback=None,
                    allow_unnamed=False, uniform_scale=None):
            output_dir.mkdir(parents=True, exist_ok=True)   # staging does this first
            if input_path.name == "broken.svg":
                raise RuntimeError("could not rasterize")
            (output_dir / "svg").mkdir()
            return fake_result(input_path, output_dir)

    monkeypatch.setattr(batch, "IconExtractor", Extractor)
    out = tmp_path / "out"
    batch.extract_folder(AppSettings(), folder, out, {"svg"})

    assert (out / "good").exists()
    assert not (out / "broken").exists()


def test_a_folder_the_user_already_had_there_is_not_removed(tmp_path, monkeypatch):
    """Only litter this run created goes; anything with content stays."""
    folder = write_sheets(tmp_path / "in", ["broken.svg"])
    out = tmp_path / "out"
    (out / "broken").mkdir(parents=True)
    (out / "broken" / "notes.txt").write_text("mine")

    class Extractor:
        def __init__(self, settings):
            pass

        def extract(self, *args, **kwargs):
            raise RuntimeError("could not rasterize")

    monkeypatch.setattr(batch, "IconExtractor", Extractor)
    batch.extract_folder(AppSettings(), folder, out, {"svg"})

    assert (out / "broken" / "notes.txt").read_text() == "mine"


def test_an_unready_backend_stops_the_run_instead_of_failing_every_sheet(tmp_path, monkeypatch):
    """The backend is down for the whole folder, not for one sheet.

    Recording it per sheet turns one recoverable, actionable failure into nine
    identical unactionable ones, and silences the "export with generic names?"
    offer that both the CLI and the app build on this exception.
    """
    folder = write_sheets(tmp_path / "in", ["a.svg", "b.svg", "c.svg"])
    attempts = []

    class Extractor:
        def __init__(self, settings):
            pass

        def extract(self, input_path, output_dir, formats, progress_callback=None,
                    allow_unnamed=False, uniform_scale=None):
            attempts.append(input_path.name)
            raise SemanticPreflightError(PreflightResult(False, "llamacpp", "not reachable"))

    monkeypatch.setattr(batch, "IconExtractor", Extractor)
    with pytest.raises(SemanticPreflightError):
        batch.extract_folder(AppSettings(), folder, tmp_path / "out", {"svg"})

    assert attempts == ["a.svg"], "it should have given up after the first sheet"


def test_giving_up_on_the_backend_leaves_no_folder_behind(tmp_path, monkeypatch):
    folder = write_sheets(tmp_path / "in", ["a.svg"])

    class Extractor:
        def __init__(self, settings):
            pass

        def extract(self, input_path, output_dir, formats, progress_callback=None,
                    allow_unnamed=False, uniform_scale=None):
            output_dir.mkdir(parents=True, exist_ok=True)
            raise SemanticPreflightError(PreflightResult(False, "llamacpp", "not reachable"))

    monkeypatch.setattr(batch, "IconExtractor", Extractor)
    out = tmp_path / "out"
    with pytest.raises(SemanticPreflightError):
        batch.extract_folder(AppSettings(), folder, out, {"svg"})

    assert not (out / "a").exists()


# --- one scale for the whole folder ---------------------------------------

class MeasuringExtractor:
    """Records the scale it was handed, and what it was asked to measure."""

    calls: dict = {}

    def __init__(self, settings):
        self.settings = settings

    def measure(self, input_path):
        # Padded sizes, which is what the real measure() returns.
        MeasuringExtractor.calls.setdefault("measured", []).append(input_path.name)
        return {"big.svg": [(300, 160)], "small.svg": [(40, 40)]}[input_path.name]

    def extract(self, input_path, output_dir, formats, progress_callback=None,
                allow_unnamed=False, uniform_scale=None):
        MeasuringExtractor.calls.setdefault("scales", []).append((input_path.name, uniform_scale))
        return fake_result(input_path, output_dir)


@pytest.fixture
def measuring(monkeypatch):
    MeasuringExtractor.calls = {}
    monkeypatch.setattr(batch, "IconExtractor", MeasuringExtractor)
    return MeasuringExtractor.calls


def test_uniform_scale_is_computed_across_every_sheet(tmp_path, measuring):
    """The point of the mode is that a set lines up. Per sheet, it does not.

    Two sheets, one holding a 300x160 shape and one a 40x40: scaled per sheet
    the small one comes out seven times larger than it should relative to its
    neighbour, which is the opposite of what "uniform" promises.
    """
    folder = write_sheets(tmp_path / "in", ["big.svg", "small.svg"])
    settings = AppSettings(canvas_mode="uniform_to_largest", output_width=512,
                           output_height=512, padding=8)

    batch.extract_folder(settings, folder, tmp_path / "out", {"svg"})

    scales = {scale for _name, scale in measuring["scales"]}
    assert len(scales) == 1, measuring["scales"]
    # 300 wide is the largest padded source anywhere in the folder, and width
    # is the tighter fit of the two; the small sheet does not get its own.
    assert scales.pop() == pytest.approx(512 / 300)


def test_every_sheet_is_measured_before_any_is_exported(tmp_path, measuring):
    """A scale computed from sheets one to three is not a folder-wide scale."""
    folder = write_sheets(tmp_path / "in", ["big.svg", "small.svg"])
    settings = AppSettings(canvas_mode="uniform_to_largest")

    batch.extract_folder(settings, folder, tmp_path / "out", {"svg"})

    assert measuring["measured"] == ["big.svg", "small.svg"]


def test_the_other_canvas_modes_do_not_pay_for_a_measuring_pass(tmp_path, measuring):
    """Measuring costs an Inkscape query per sheet; only one mode needs it."""
    folder = write_sheets(tmp_path / "in", ["big.svg", "small.svg"])
    settings = AppSettings(canvas_mode="individual_fit")

    batch.extract_folder(settings, folder, tmp_path / "out", {"svg"})

    assert "measured" not in measuring
    assert [scale for _name, scale in measuring["scales"]] == [None, None]


def test_a_sheet_that_cannot_be_measured_does_not_stop_the_run(tmp_path, monkeypatch):
    """A corrupt sheet fails at extraction, where it is reported by name."""
    folder = write_sheets(tmp_path / "in", ["big.svg", "broken.svg"])
    seen = []

    class Extractor:
        def __init__(self, settings):
            pass

        def measure(self, input_path):
            if input_path.name == "broken.svg":
                raise RuntimeError("could not parse")
            return [(300, 160)]

        def extract(self, input_path, output_dir, formats, progress_callback=None,
                    allow_unnamed=False, uniform_scale=None):
            seen.append((input_path.name, uniform_scale))
            if input_path.name == "broken.svg":
                raise RuntimeError("could not parse")
            return fake_result(input_path, output_dir)

    monkeypatch.setattr(batch, "IconExtractor", Extractor)
    settings = AppSettings(canvas_mode="uniform_to_largest", output_width=512,
                           output_height=512, padding=8)
    outcome = batch.extract_folder(settings, folder, tmp_path / "out", {"svg"})

    assert [name for name, _s in seen] == ["big.svg", "broken.svg"]
    assert seen[0][1] == pytest.approx(512 / 300), "the readable sheet still sets the scale"
    assert outcome.failed_count == 1
