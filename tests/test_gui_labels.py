"""The GUI's label maps must stay in step with the schema.

`gui/workspace_tab.py` still owns the wording shown against each canvas and
bitmap mode. Nothing at runtime forces those keys to agree with
`services/settings_schema.py`, so a mode renamed on one side would otherwise
drift silently and the GUI would offer a value the schema rejects.

The detection presets are no longer checked for GUI/CLI agreement: both
interfaces now read `services.detection_presets`, so there is no second copy
to disagree with. `tests/test_detection_presets.py` covers that module.

No Tk widget is constructed — only module-level constants are read.
"""
from __future__ import annotations

import shapearator
from gui import workspace_tab
from gui.workspace_tab import BITMAP_EXPORT_MODE_LABELS, CANVAS_MODE_LABELS
from services import detection_presets
from services.settings_schema import BITMAP_EXPORT_MODES, CANVAS_MODES


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


# --- one shared preset table ------------------------------------------

def test_neither_interface_declares_its_own_preset_table():
    """The duplication these tests used to police is gone; keep it gone."""
    assert not hasattr(shapearator, "DETECTION_PRESETS")
    assert not hasattr(workspace_tab, "DETECTION_PRESETS")


def test_both_interfaces_read_the_same_preset_names():
    assert list(detection_presets.preset_names()) == list(
        detection_presets.DETECTION_PRESETS
    )
