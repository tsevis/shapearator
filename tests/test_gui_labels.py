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
from gui.workspace_tab import (
    BITMAP_EXPORT_MODE_LABELS,
    CANVAS_MODE_LABELS,
    SVG_SPLIT_LABELS,
    PSD_LAYER_LABELS,
    PSD_LAYOUT_LABELS,
    psd_layers_key,
    psd_layout_key,
    svg_split_key,
)
from services import detection_presets
from services.settings_schema import (
    BITMAP_EXPORT_MODES,
    CANVAS_MODES,
    PSD_LAYER_MODES,
    PSD_LAYOUTS,
    SVG_SPLIT_MODES,
)


# --- schema parity --------------------------------------------------------

def test_every_canvas_mode_is_offered_exactly_once():
    assert set(CANVAS_MODE_LABELS) == set(CANVAS_MODES)


def test_every_bitmap_export_mode_is_offered_exactly_once():
    assert set(BITMAP_EXPORT_MODE_LABELS) == set(BITMAP_EXPORT_MODES)


def test_every_svg_split_mode_is_offered_exactly_once():
    assert set(SVG_SPLIT_LABELS) == set(SVG_SPLIT_MODES)


def test_a_split_label_maps_back_to_its_schema_value():
    """The dropdown shows prose; the settings file must get the enum."""
    for key, label in SVG_SPLIT_LABELS.items():
        assert svg_split_key(label) == key


def test_an_unknown_split_label_falls_back_to_auto():
    assert svg_split_key("something a future version offered") == "auto"


def test_every_psd_layer_mode_is_offered_exactly_once():
    assert set(PSD_LAYER_LABELS) == set(PSD_LAYER_MODES)


def test_every_psd_layout_is_offered_exactly_once():
    assert set(PSD_LAYOUT_LABELS) == set(PSD_LAYOUTS)


def test_a_psd_label_maps_back_to_its_schema_value():
    for key, label in PSD_LAYER_LABELS.items():
        assert psd_layers_key(label) == key
    for key, label in PSD_LAYOUT_LABELS.items():
        assert psd_layout_key(label) == key


def test_an_unknown_psd_label_falls_back_to_the_default():
    assert psd_layers_key("something a future version offered") == "bitmap"
    assert psd_layout_key("something a future version offered") == "sheet"


def test_no_label_is_blank():
    for labels in (CANVAS_MODE_LABELS, BITMAP_EXPORT_MODE_LABELS, SVG_SPLIT_LABELS,
                   PSD_LAYER_LABELS, PSD_LAYOUT_LABELS):
        for key, label in labels.items():
            assert label.strip(), f"{key} has no label text"


def test_labels_are_distinguishable_in_a_dropdown():
    for labels in (CANVAS_MODE_LABELS, BITMAP_EXPORT_MODE_LABELS, SVG_SPLIT_LABELS,
                   PSD_LAYER_LABELS, PSD_LAYOUT_LABELS):
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
