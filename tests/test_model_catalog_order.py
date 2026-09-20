"""The order the catalogue offers vision models in.

The app shows these as a download list, best first, and a user picks from the
top. That ranking is a recommendation, so it has to reflect how the models
actually behave on icon sheets rather than how promising they look.
"""
from __future__ import annotations

from services.model_catalog import CATALOG, spec_by_key


def test_priority_matches_list_order():
    """The list is documented as ordered best-first with priority mirroring
    it. Nothing enforced that, so the two could drift apart silently and the
    UI would sort by one while the file reads as the other."""
    priorities = [spec.priority for spec in CATALOG]
    assert priorities == sorted(priorities), [
        (spec.key, spec.priority) for spec in CATALOG
    ]


def test_minicpm_is_not_offered_ahead_of_models_that_behave():
    """Measured on one 47-icon sheet, via Ollama, with the app's own prompt:
    MiniCPM-V returned the wording of the instruction instead of a label for
    30 of 47 icons -- filenames like `short-lowercase-filename-label-17`.
    Qwen2.5-VL 3B did it once and Qwen3-VL not at all, on the same sheet.

    One sheet is not a benchmark, and MiniCPM-V reads some marks well. But it
    was sitting third, above every model with no known failure of this kind,
    which is a recommendation the evidence does not support.
    """
    minicpm = spec_by_key("minicpm-v")
    assert minicpm is not None
    for key in ("qwen3-vl", "qwen2.5-vl", "moondream", "llava"):
        other = spec_by_key(key)
        assert other is not None, key
        assert minicpm.priority > other.priority, (
            f"MiniCPM-V is offered ahead of {key}"
        )
