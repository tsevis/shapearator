"""The GUI's choice tables must stay in step with the schema and the CLI.

`gui/workspace_tab.py` keeps its own label maps and its own copy of the
detection presets. Nothing at runtime forces those to agree with
`services/settings_schema.py` or with the CLI's table in `shapearator.py`, so
a value renamed on one side would otherwise drift silently: the GUI would
offer a mode the schema rejects, or the same preset name would mean different
numbers depending on which interface you used. These tests are that check.

No Tk widget is constructed — only module-level constants are read.
"""
from __future__ import annotations

from shapearator import DETECTION_PRESETS as CLI_PRESETS
from gui.workspace_tab import (
    BITMAP_EXPORT_MODE_LABELS,
    CANVAS_MODE_LABELS,
    DETECTION_PRESETS as GUI_PRESETS,
)
from services.settings_schema import BITMAP_EXPORT_MODES, CANVAS_MODES

PRESET_FIELDS = {"padding", "min_area", "merge_gap"}


# --- schema parity --------------------------------------------------------

def test_every_canvas_mode_is_offered_exactly_once():
    assert set(CANVAS_MODE_LABELS) == set(CANVAS_MODES)


def test_every_bitmap_export_mode_is_offered_exactly_once():
    assert set(BITMAP_EXPORT_MODE_LABELS) == set(BITMAP_EXPORT_MODES)


def test_no_label_is_blank():
    for labels in (CANVAS_MODE_LABELS, BITMAP_EXPORT_MODE_LABELS):
        for key, label in labels.items():
            assert label.strip(), f"{key} has no label text"


def test_labels_are_distinguishable_in_a_dropdown():
    for labels in (CANVAS_MODE_LABELS, BITMAP_EXPORT_MODE_LABELS):
        assert len(set(labels.values())) == len(labels)


# --- CLI parity -----------------------------------------------------------

def test_the_gui_and_cli_presets_are_the_same_table():
    assert GUI_PRESETS == CLI_PRESETS


def test_the_documented_four_presets_are_all_present():
    assert set(GUI_PRESETS) == {
        "Balanced",
        "Tiny Details",
        "Loose Sketches",
        "Bold Shapes",
    }


def test_each_preset_sets_every_detection_field_and_nothing_else():
    for name, preset in GUI_PRESETS.items():
        assert set(preset) == PRESET_FIELDS, f"{name} has the wrong fields"


def test_detection_values_are_positive_whole_numbers():
    for name, preset in GUI_PRESETS.items():
        for field, value in preset.items():
            assert isinstance(value, int) and not isinstance(value, bool), (
                f"{name}.{field} is not an int"
            )
            assert value > 0, f"{name}.{field} is not positive"


def test_presets_are_ordered_as_the_detection_settings_describe_them():
    """Tiny Details keeps the smallest marks; Bold Shapes ignores the most."""
    areas = {name: preset["min_area"] for name, preset in GUI_PRESETS.items()}
    assert areas["Tiny Details"] < areas["Balanced"] < areas["Bold Shapes"]
