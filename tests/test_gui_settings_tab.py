"""The settings tab's model-recommendation text.

`_recommendation_text` is a static method, so it can be exercised straight off
the class: no `SettingsTab` is instantiated and no ttk.Frame is built. It
decides what the user is told about the model they picked versus the one the
registry suggests, which is the tab's one piece of real branching logic.
"""
from __future__ import annotations

from gui.settings_tab import SettingsTab
from services.model_registry import ModelDescriptor

recommendation_text = SettingsTab._recommendation_text


def model(name: str, recommendation: str = "") -> ModelDescriptor:
    return ModelDescriptor(
        name=name,
        source="ollama",
        location="local",
        recommendation=recommendation or f"{name} notes",
        supports_vision=True,
    )


# --- nothing to say -------------------------------------------------------

def test_no_models_and_no_recommendation_produces_no_text():
    assert recommendation_text([], "", None) == ""


def test_a_recommendation_alone_is_still_reported():
    text = recommendation_text([], "", model("qwen3-vl", "best overall"))
    assert text == "Recommended: qwen3-vl. best overall"


# --- selected vs recommended ----------------------------------------------

def test_a_selection_differing_from_the_recommendation_reports_both():
    models = [model("llava", "older but fine"), model("qwen3-vl", "best overall")]

    text = recommendation_text(models, "llava", models[1])

    assert text == (
        "Recommended: qwen3-vl. best overall Selected: llava. older but fine"
    )


def test_selecting_the_recommended_model_annotates_rather_than_repeats():
    recommended = model("qwen3-vl", "best overall")

    text = recommendation_text([recommended], "qwen3-vl", recommended)

    assert text == (
        "Recommended: qwen3-vl. best overall Selected model note: best overall"
    )
    assert "Selected: qwen3-vl" not in text


def test_the_recommended_model_is_matched_by_name_not_identity():
    """The registry and the model list need not hand back the same object."""
    listed = model("qwen3-vl", "best overall")
    recommended = model("qwen3-vl", "best overall")

    text = recommendation_text([listed], "qwen3-vl", recommended)

    assert "Selected model note:" in text


# --- edges ----------------------------------------------------------------

def test_a_selection_that_is_no_longer_installed_is_not_described():
    recommended = model("qwen3-vl", "best overall")

    text = recommendation_text([recommended], "deleted-model", recommended)

    assert text == "Recommended: qwen3-vl. best overall"
    assert "deleted-model" not in text


def test_a_selection_with_no_recommendation_available_stands_alone():
    models = [model("llava", "older but fine")]

    text = recommendation_text(models, "llava", None)

    assert text == "Selected: llava. older but fine"


def test_an_empty_selection_reports_only_the_recommendation():
    models = [model("llava")]
    recommended = model("qwen3-vl", "best overall")

    assert recommendation_text(models, "", recommended) == (
        "Recommended: qwen3-vl. best overall"
    )
