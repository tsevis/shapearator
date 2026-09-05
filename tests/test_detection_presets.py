"""The detection presets, now owned by one module instead of two.

These values are the documented four presets, and the numbers are part of the
public behaviour: a run reproduced from a preset name has to keep meaning the
same thing across versions.
"""
from __future__ import annotations

import dataclasses

import pytest

from services.detection_presets import (
    DEFAULT_PRESET_NAME,
    DETECTION_PRESETS,
    DetectionValues,
    preset_name_for_values,
    preset_names,
    values_for_preset,
)


# --- the table ------------------------------------------------------------

def test_the_documented_four_presets_are_offered_in_order():
    assert preset_names() == (
        "Balanced",
        "Tiny Details",
        "Loose Sketches",
        "Bold Shapes",
    )


def test_the_default_preset_is_one_of_them():
    assert DEFAULT_PRESET_NAME in DETECTION_PRESETS


def test_detection_values_are_positive_whole_numbers():
    for name, preset in DETECTION_PRESETS.items():
        for field in dataclasses.fields(preset):
            value = getattr(preset, field.name)
            assert isinstance(value, int) and not isinstance(value, bool), (
                f"{name}.{field.name} is not an int"
            )
            assert value > 0, f"{name}.{field.name} is not positive"


def test_presets_are_ordered_as_the_documentation_describes_them():
    """Tiny Details keeps the smallest marks; Bold Shapes ignores the most."""
    areas = {name: preset.min_area for name, preset in DETECTION_PRESETS.items()}
    assert areas["Tiny Details"] < areas["Balanced"] < areas["Bold Shapes"]


def test_the_table_cannot_be_edited_by_a_caller():
    with pytest.raises(TypeError):
        DETECTION_PRESETS["Balanced"] = DetectionValues(1, 1, 1)


def test_a_preset_cannot_be_edited_by_a_caller():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DETECTION_PRESETS["Balanced"].padding = 99


# --- lookup ---------------------------------------------------------------

def test_a_known_name_returns_its_values():
    assert values_for_preset("Bold Shapes") == DetectionValues(
        padding=14, min_area=320, merge_gap=15
    )


@pytest.mark.parametrize("name", [None, "", "Not A Preset"])
def test_an_unusable_name_returns_nothing_rather_than_raising(name):
    """The CLI passes None when no preset was given; neither caller wants a KeyError."""
    assert values_for_preset(name) is None


# --- naming the current values --------------------------------------------

def test_values_taken_from_a_preset_are_named_as_that_preset():
    for name, preset in DETECTION_PRESETS.items():
        assert (
            preset_name_for_values(preset.padding, preset.min_area, preset.merge_gap)
            == name
        )


def test_hand_tuned_values_fall_back_to_the_default_name():
    assert preset_name_for_values(1, 2, 3) == DEFAULT_PRESET_NAME


def test_values_matching_a_preset_on_only_some_fields_are_not_that_preset():
    balanced = DETECTION_PRESETS["Balanced"]
    assert (
        preset_name_for_values(balanced.padding, balanced.min_area, 999)
        == DEFAULT_PRESET_NAME
    )
