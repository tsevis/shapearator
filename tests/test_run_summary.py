"""Display text for settings, finished runs and individual icons."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from services.run_summary import (
    describe_icon,
    describe_progress_completion,
    describe_result,
    format_size,
    provider_summary,
)
from services import run_summary
from services.settings_schema import AppSettings


def settings(**overrides) -> AppSettings:
    base = AppSettings()
    for field, value in overrides.items():
        setattr(base, field, value)
    return base


@dataclass
class FakeIcon:
    stem: str = "icon_001"
    source_size: tuple[int, int] = (44, 42)
    canvas_size: tuple[int, int] = (512, 512)


@dataclass
class FakeNaming:
    requested: bool = False
    summary: str = "named 3, failed 1"

    def describe(self) -> str:
        return self.summary


@dataclass
class FakeCommit:
    replaced: int = 0


@dataclass
class FakeResult:
    icons: list
    output_dir: Path = Path("/exports")
    naming: FakeNaming = None
    commit: FakeCommit = None

    def __post_init__(self):
        if self.naming is None:
            self.naming = FakeNaming()


# --- sizes ----------------------------------------------------------------

def test_a_size_pair_reads_as_width_by_height():
    assert format_size((512, 384)) == "512 x 384"


# --- provider summary -----------------------------------------------------

def test_geometry_is_described_without_a_model():
    assert provider_summary(settings()) == (
        "Active provider: Geometry-only local extraction"
    )


def test_an_unknown_provider_falls_back_to_the_geometry_wording():
    assert provider_summary(settings(provider="something-new")) == (
        "Active provider: Geometry-only local extraction"
    )


def test_ollama_names_its_model():
    text = provider_summary(settings(provider="ollama", ollama_model="qwen2.5vl:3b"))
    assert text == "Active provider: Ollama local (qwen2.5vl:3b)"


def test_llamacpp_without_a_named_model_says_what_is_loaded():
    text = provider_summary(settings(provider="llamacpp", llamacpp_model=""))
    assert text == "Active provider: llama.cpp local (loaded model)"


def test_a_directory_provider_without_a_model_names_the_catalog():
    text = provider_summary(settings(provider="directory", local_model_name=""))
    assert text == "Active provider: Local directory (directory catalog)"


# --- icons ----------------------------------------------------------------

def test_an_icon_caption_carries_both_sizes():
    assert describe_icon(FakeIcon()) == (
        "icon_001  |  source 44 x 42  |  canvas 512 x 512"
    )


# --- finished runs --------------------------------------------------------

def test_a_plain_run_into_a_fresh_folder_is_one_sentence():
    result = FakeResult(icons=[FakeIcon(), FakeIcon()])
    assert describe_result(result) == "Extracted 2 icons to /exports"


def test_replaced_files_are_mentioned_only_when_some_were_replaced():
    replaced = FakeResult(icons=[FakeIcon()], commit=FakeCommit(replaced=141))
    untouched = FakeResult(icons=[FakeIcon()], commit=FakeCommit(replaced=0))

    assert "replaced 141 files from the previous run" in describe_result(replaced)
    assert "replaced" not in describe_result(untouched)


def test_naming_is_mentioned_only_when_naming_was_requested():
    asked = FakeResult(icons=[FakeIcon()], naming=FakeNaming(requested=True))
    skipped = FakeResult(icons=[FakeIcon()], naming=FakeNaming(requested=False))

    assert "named 3, failed 1" in describe_result(asked)
    assert "named 3, failed 1" not in describe_result(skipped)


def test_a_run_with_everything_to_report_keeps_the_parts_separated():
    result = FakeResult(
        icons=[FakeIcon()],
        naming=FakeNaming(requested=True),
        commit=FakeCommit(replaced=7),
    )
    assert describe_result(result) == (
        "Extracted 1 icons to /exports"
        "  |  replaced 7 files from the previous run"
        "  |  named 3, failed 1"
    )


def test_a_result_without_a_commit_is_described_anyway():
    """A run that never reached the commit stage still has a status line."""
    assert describe_result(FakeResult(icons=[])) == "Extracted 0 icons to /exports"


# --- progress -------------------------------------------------------------

def test_the_completion_label_counts_the_icons():
    assert describe_progress_completion(47) == "Done. Exported 47 icons."


# --- what the app sees at the chosen input --------------------------------

def test_a_folder_reports_how_many_sheets_it_will_extract(tmp_path):
    folder = tmp_path / "MANY"
    folder.mkdir()
    for name in ("Many1.svg", "Many2.svg", "Many3.png", "notes.txt"):
        (folder / name).write_text("x")
    assert "3 sheets" in run_summary.describe_input(folder)


def test_one_sheet_is_not_described_as_one_sheets(tmp_path):
    folder = tmp_path / "MANY"
    folder.mkdir()
    (folder / "only.svg").write_text("x")
    text = run_summary.describe_input(folder)
    assert "1 sheet" in text and "1 sheets" not in text


def test_a_folder_with_nothing_in_it_says_so_rather_than_zero(tmp_path):
    """"0 sheets" reads like a count; this has to read like a mistake."""
    folder = tmp_path / "Wrong"
    folder.mkdir()
    text = run_summary.describe_input(folder)
    assert "0" not in text
    assert "no" in text.lower()


def test_a_single_sheet_needs_no_explanation(tmp_path):
    sheet = tmp_path / "one.svg"
    sheet.write_text("x")
    assert run_summary.describe_input(sheet) == ""
