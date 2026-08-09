"""Tests for schema-aware settings coercion and resilient config loading."""
from __future__ import annotations

import json

import pytest

from services.config_store import AppSettings, ConfigStore
from services import settings_schema as schema


# --- unknown / malformed input -------------------------------------------

def test_unknown_keys_are_ignored_with_a_warning():
    result = schema.coerce_settings({"unknown_future_setting": True, "padding": 5})
    assert result.settings.padding == 5
    assert any("unknown_future_setting" in warning for warning in result.warnings)


def test_non_object_payload_falls_back_to_defaults():
    for payload in ([1, 2, 3], "text", 42, None):
        result = schema.coerce_settings(payload)
        assert result.settings == AppSettings()
        assert result.warnings


def test_empty_object_yields_defaults_without_warnings():
    result = schema.coerce_settings({})
    assert result.settings == AppSettings()
    assert result.warnings == []


# --- per-field validation -------------------------------------------------

def test_invalid_value_falls_back_per_field_and_keeps_valid_siblings():
    result = schema.coerce_settings({"output_width": "not-an-int", "output_height": 256})
    assert result.settings.output_width == AppSettings().output_width
    assert result.settings.output_height == 256
    assert any("output_width" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("provider", "openai"),
        ("canvas_mode", "stretch"),
        ("bitmap_export_mode", "magic"),
        ("appearance", "neon"),
    ],
)
def test_enum_fields_reject_values_outside_the_catalog(field, bad_value):
    result = schema.coerce_settings({field: bad_value})
    assert getattr(result.settings, field) == getattr(AppSettings(), field)
    assert any(field in warning for warning in result.warnings)


@pytest.mark.parametrize("field", ["provider", "canvas_mode", "bitmap_export_mode", "appearance"])
def test_enum_fields_accept_every_documented_choice(field):
    for choice in schema.CHOICES[field]:
        result = schema.coerce_settings({field: choice})
        assert getattr(result.settings, field) == choice
        assert result.warnings == []


def test_numeric_strings_are_coerced_to_int():
    result = schema.coerce_settings({"output_width": "1024", "padding": "0"})
    assert result.settings.output_width == 1024
    assert result.settings.padding == 0
    assert result.warnings == []


def test_bools_are_not_accepted_as_numbers():
    # True is an int in Python; a JSON `true` for a pixel count is a mistake.
    result = schema.coerce_settings({"output_width": True})
    assert result.settings.output_width == AppSettings().output_width
    assert result.warnings


@pytest.mark.parametrize(
    "field,bad_value",
    [("output_width", 0), ("output_height", -1), ("padding", -3), ("min_area", 0), ("merge_gap", 0)],
)
def test_numeric_bounds_are_enforced(field, bad_value):
    result = schema.coerce_settings({field: bad_value})
    assert getattr(result.settings, field) == getattr(AppSettings(), field)
    assert any(field in warning for warning in result.warnings)


def test_semantic_naming_requires_a_real_bool():
    assert schema.coerce_settings({"semantic_naming": True}).settings.semantic_naming is True
    fallback = schema.coerce_settings({"semantic_naming": "yes"})
    assert fallback.settings.semantic_naming is False
    assert fallback.warnings


# --- format list ----------------------------------------------------------

def test_formats_drop_unknown_entries_and_deduplicate():
    result = schema.coerce_settings({"default_formats": ["png", "gif", "png", "svg"]})
    assert result.settings.default_formats == ["png", "svg"]
    assert any("gif" in warning for warning in result.warnings)


def test_formats_fall_back_when_nothing_valid_remains():
    result = schema.coerce_settings({"default_formats": ["gif", "bmp"]})
    assert result.settings.default_formats == AppSettings().default_formats
    assert result.warnings


def test_formats_reject_non_list_values():
    result = schema.coerce_settings({"default_formats": "png"})
    assert result.settings.default_formats == AppSettings().default_formats
    assert result.warnings


def test_formats_are_a_fresh_list_per_call():
    first = schema.coerce_settings({}).settings
    second = schema.coerce_settings({}).settings
    first.default_formats.append("tiff")
    assert second.default_formats == AppSettings().default_formats


# --- string fields --------------------------------------------------------

def test_string_fields_reject_non_strings():
    result = schema.coerce_settings({"ollama_url": 8080, "ollama_model": "moondream"})
    assert result.settings.ollama_url == AppSettings().ollama_url
    assert result.settings.ollama_model == "moondream"
    assert any("ollama_url" in warning for warning in result.warnings)


def test_url_fields_reject_blank_values():
    result = schema.coerce_settings({"llamacpp_url": "   "})
    assert result.settings.llamacpp_url == AppSettings().llamacpp_url
    assert result.warnings


def test_path_fields_accept_empty_strings():
    result = schema.coerce_settings({"models_root": "", "last_output_dir": "/tmp/out"})
    assert result.settings.models_root == ""
    assert result.settings.last_output_dir == "/tmp/out"
    assert result.warnings == []


# --- coercion is non-mutating --------------------------------------------

def test_coercion_does_not_mutate_the_input_mapping():
    payload = {"padding": "8", "unknown": 1}
    schema.coerce_settings(payload)
    assert payload == {"padding": "8", "unknown": 1}


# --- ConfigStore integration ---------------------------------------------

def test_load_missing_file_returns_defaults(tmp_path):
    assert ConfigStore(tmp_path / "absent.json").load() == AppSettings()


def test_load_survives_unknown_keys(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"unknown_future_setting": True, "padding": 7}), encoding="utf-8")
    assert ConfigStore(path).load().padding == 7


def test_load_survives_malformed_json(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert ConfigStore(path).load() == AppSettings()


def test_load_survives_non_object_json(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert ConfigStore(path).load() == AppSettings()


def test_load_reports_warnings_when_requested(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"provider": "openai", "stray": 1}), encoding="utf-8")
    settings, warnings = ConfigStore(path).load_with_warnings()
    assert settings.provider == AppSettings().provider
    assert len(warnings) == 2


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "nested" / "settings.json"
    original = AppSettings(provider="llamacpp", output_width=256, default_formats=["svg"])
    store = ConfigStore(path)
    store.save(original)
    assert store.load() == original


# --- single source of truth for choices ----------------------------------

def test_cli_reuses_the_schema_choice_lists():
    import shapearator as cli

    assert cli.PROVIDER_CHOICES == list(schema.PROVIDERS)
    assert cli.CANVAS_MODE_CHOICES == list(schema.CANVAS_MODES)
    assert cli.BITMAP_EXPORT_MODE_CHOICES == list(schema.BITMAP_EXPORT_MODES)
    assert cli.FORMAT_CHOICES == list(schema.FORMATS)
