"""The detection presets, owned in one place.

The GUI offers these in a dropdown and the CLI as `--detection-preset`. They
used to be declared separately in each, which meant the same preset name could
quietly come to mean different numbers depending on which interface you ran.
Both now read this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

DEFAULT_PRESET_NAME = "Balanced"


@dataclass(frozen=True)
class DetectionValues:
    """The three detection settings a preset fixes."""

    padding: int
    min_area: int
    merge_gap: int


DETECTION_PRESETS: Mapping[str, DetectionValues] = MappingProxyType(
    {
        "Balanced": DetectionValues(padding=12, min_area=200, merge_gap=13),
        "Tiny Details": DetectionValues(padding=8, min_area=70, merge_gap=9),
        "Loose Sketches": DetectionValues(padding=16, min_area=140, merge_gap=19),
        "Bold Shapes": DetectionValues(padding=14, min_area=320, merge_gap=15),
    }
)


def preset_names() -> tuple[str, ...]:
    """Preset names in the order both interfaces should present them."""
    return tuple(DETECTION_PRESETS)


def values_for_preset(name: str | None) -> DetectionValues | None:
    """The values a preset stands for, or None if the name is not one."""
    if not name:
        return None
    return DETECTION_PRESETS.get(name)


def preset_name_for_values(padding: int, min_area: int, merge_gap: int) -> str:
    """Name the preset these values came from.

    Hand-tuned values match no preset; the GUI still has to show something in
    its dropdown, so the default name stands in.
    """
    probe = DetectionValues(padding=padding, min_area=min_area, merge_gap=merge_gap)
    for name, preset in DETECTION_PRESETS.items():
        if preset == probe:
            return name
    return DEFAULT_PRESET_NAME
