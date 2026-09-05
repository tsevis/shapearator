"""Widget tests that build real windows.

Every test here asks for `gui_root`, which is what marks it `gui` and keeps it
out of a plain `pytest` run — see tests/conftest.py. Run them deliberately with
`pytest -m gui`, on a machine where windows appearing is acceptable.

What earns a window: the wiring that only exists once widgets are real — Tk
variables initialised from settings, a preset write reaching the spinboxes, a
result populating the tree. Logic that can be tested without one already is,
in tests/test_gui_labels.py and the services suites.

Nothing here reaches the network or a subprocess: the settings tab's model
discovery is stubbed, as is the first-run check that would otherwise pop a
download dialog.
"""
from __future__ import annotations

from pathlib import Path
from tkinter import ttk

import pytest

from gui.settings_tab import SettingsTab
from gui.theme_utils import create_themed_toplevel
from gui.workspace_tab import CANVAS_MODE_LABELS, WorkspaceTab
from services.detection_presets import DETECTION_PRESETS
from services.extraction_types import (
    ExtractedIcon,
    ExtractionProgress,
    ExtractionResult,
    NamingSummary,
)
from services.settings_schema import AppSettings


def settings(**overrides) -> AppSettings:
    base = AppSettings()
    for field, value in overrides.items():
        setattr(base, field, value)
    return base


@pytest.fixture
def notebook(gui_root):
    book = ttk.Notebook(gui_root)
    book.pack()
    return book


@pytest.fixture
def workspace(notebook):
    committed = []
    tab = WorkspaceTab(notebook, settings(), committed.append)
    tab.committed = committed
    return tab


@pytest.fixture
def offline_registry(monkeypatch):
    """Keep the settings tab from shelling out to ollama or hitting the network."""
    import gui.settings_tab as st

    monkeypatch.setattr(st.ModelRegistry, "list_ollama_models", lambda self: [])
    monkeypatch.setattr(st.ModelRegistry, "list_llamacpp_models", lambda self, url: [])
    monkeypatch.setattr(st.ModelRegistry, "list_directory_models", lambda self, root: [])
    monkeypatch.setattr(st, "available_startable_models", lambda _root: [])


def icon(index: int = 1, stem: str = "icon_001") -> ExtractedIcon:
    return ExtractedIcon(
        index=index,
        stem=stem,
        outputs={"png": Path("exports/png") / f"{stem}.png"},
        preview_path=None,
        canvas_size=(512, 512),
        source_size=(44, 42),
        source_bounds=(0, 0, 44, 42),
    )


# --- the workspace tab builds and reflects its settings -------------------

def test_the_workspace_builds_from_settings(notebook):
    tab = WorkspaceTab(notebook, settings(padding=7, output_width=640), lambda _s: None)

    assert tab.padding_var.get() == 7
    assert tab.output_width_var.get() == 640


def test_the_format_checkboxes_start_from_the_saved_selection(notebook):
    tab = WorkspaceTab(notebook, settings(default_formats=["png", "svg"]), lambda _s: None)

    assert tab.export_png_var.get() is True
    assert tab.export_svg_var.get() is True
    assert tab.export_jpg_var.get() is False


def test_the_selected_formats_follow_the_checkboxes(workspace):
    workspace.export_png_var.set(True)
    workspace.export_jpg_var.set(False)
    workspace.export_tiff_var.set(True)
    workspace.export_svg_var.set(False)

    assert workspace._selected_formats() == {"png", "tiff"}


def test_clearing_every_checkbox_selects_no_format(workspace):
    for var in (
        workspace.export_png_var,
        workspace.export_jpg_var,
        workspace.export_tiff_var,
        workspace.export_svg_var,
    ):
        var.set(False)

    assert workspace._selected_formats() == set()


# --- detection presets, through real spinbox variables --------------------

def test_choosing_a_preset_writes_its_values_into_the_spinboxes(workspace):
    workspace.detection_preset_var.set("Bold Shapes")

    workspace._apply_detection_preset()

    expected = DETECTION_PRESETS["Bold Shapes"]
    assert workspace.padding_var.get() == expected.padding
    assert workspace.min_area_var.get() == expected.min_area
    assert workspace.merge_gap_var.get() == expected.merge_gap


def test_an_unknown_preset_leaves_the_spinboxes_alone(workspace):
    before = workspace.min_area_var.get()

    workspace.detection_preset_var.set("Not A Preset")
    workspace._apply_detection_preset()

    assert workspace.min_area_var.get() == before


def test_the_preset_name_is_read_back_from_the_live_values(workspace):
    tiny = DETECTION_PRESETS["Tiny Details"]
    workspace.padding_var.set(tiny.padding)
    workspace.min_area_var.set(tiny.min_area)
    workspace.merge_gap_var.set(tiny.merge_gap)

    assert workspace._preset_name_for_values() == "Tiny Details"


# --- labels that update as the user clicks --------------------------------

def test_the_canvas_hint_follows_the_selected_mode(workspace):
    workspace.canvas_mode_var.set("individual_fit")

    workspace._update_canvas_hint()

    assert workspace.canvas_hint_var.get() == CANVAS_MODE_LABELS["individual_fit"]


def test_new_settings_refresh_the_provider_label(workspace):
    workspace.update_settings(settings(provider="ollama", ollama_model="qwen2.5vl:3b"))

    assert workspace.provider_summary_var.get() == (
        "Active provider: Ollama local (qwen2.5vl:3b)"
    )


def test_progress_moves_the_bar_and_the_label(workspace):
    workspace._handle_progress(
        ExtractionProgress(phase="export", current=1, total=4, message="Exporting")
    )

    assert workspace.progress_value_var.get() == pytest.approx(25.0)
    assert workspace.progress_label_var.get() == "Exporting"


# --- results reaching the tree --------------------------------------------

def result(icons, **overrides) -> ExtractionResult:
    fields = dict(
        input_path=Path("sheet.svg"),
        output_dir=Path("exports"),
        icons=icons,
        provider_summary="Geometry-first local extractor",
        naming=NamingSummary(),
    )
    fields.update(overrides)
    return ExtractionResult(**fields)


def test_a_finished_run_fills_the_results_tree(workspace):
    workspace._handle_result(result([icon(1, "icon_001"), icon(2, "icon_002")]))

    rows = workspace.results_tree.get_children()
    assert len(rows) == 2
    assert workspace.results_tree.item(rows[0], "text") == "icon_001"


def test_the_tree_shows_the_sizes_for_each_icon(workspace):
    workspace._handle_result(result([icon()]))

    row = workspace.results_tree.get_children()[0]
    assert workspace.results_tree.item(row, "values")[1] == "44 x 42"


def test_a_second_run_replaces_the_previous_rows(workspace):
    workspace._handle_result(result([icon(1), icon(2)]))
    workspace._handle_result(result([icon(1)]))

    assert len(workspace.results_tree.get_children()) == 1


def test_a_finished_run_reports_its_count_in_the_status(workspace):
    workspace._handle_result(result([icon()]))

    assert "Extracted 1 icons" in workspace.status_var.get()
    assert workspace.progress_value_var.get() == pytest.approx(100.0)


def test_an_icon_with_no_bitmap_says_so_rather_than_failing(workspace):
    """Selecting a result must never raise just because there is no preview."""
    workspace._handle_result(result([icon()]))

    assert "icon_001" in workspace.preview_meta_var.get()


def test_a_run_that_produced_nothing_leaves_an_empty_tree(workspace):
    workspace._handle_result(result([]))

    assert workspace.results_tree.get_children() == ()


# --- theming --------------------------------------------------------------

@pytest.mark.parametrize("mode", ["light", "dark"])
def test_the_workspace_themes_without_error(workspace, mode):
    workspace.apply_theme(mode)


def test_a_themed_toplevel_is_created_and_synced(gui_root):
    applied = []
    gui_root._apply_appearance_to_window = applied.append

    window = create_themed_toplevel(gui_root)

    try:
        assert window.winfo_exists()
        assert applied == [window]
    finally:
        window.destroy()


def test_a_toplevel_is_still_created_when_no_host_can_theme_it(gui_root):
    window = create_themed_toplevel(gui_root)
    try:
        assert window.winfo_exists()
    finally:
        window.destroy()


# --- the settings tab -----------------------------------------------------

def test_the_settings_tab_builds_from_settings(notebook, offline_registry):
    tab = SettingsTab(
        notebook,
        settings(provider="llamacpp", llamacpp_url="http://127.0.0.1:8080"),
        lambda _s: None,
    )

    assert tab.provider_var.get() == "llamacpp"
    assert tab.llamacpp_url_var.get() == "http://127.0.0.1:8080"


def test_the_settings_tab_reports_no_models_when_none_are_installed(
    notebook, offline_registry
):
    tab = SettingsTab(notebook, settings(), lambda _s: None)

    assert tab.ollama_models == []
    assert tab.llamacpp_models == []
