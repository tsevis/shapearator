"""The pre-flight rules both interfaces now share.

Order matters here: the first problem found is the one a user is shown, so
these tests pin which rule wins when a request breaks several at once.
"""
from __future__ import annotations

import pytest

from services.request_validation import validate_extraction_request
from services.settings_schema import AppSettings


@pytest.fixture
def sheet(tmp_path):
    path = tmp_path / "sheet.png"
    path.write_bytes(b"not really a png, but it exists")
    return path


def settings(**overrides) -> AppSettings:
    base = AppSettings()
    for field, value in overrides.items():
        setattr(base, field, value)
    return base


# --- a request that can run -----------------------------------------------

def test_a_well_formed_request_has_no_problem(sheet):
    assert validate_extraction_request(settings(), sheet, {"png"}) is None


def test_an_svg_sheet_is_equally_acceptable(tmp_path):
    svg = tmp_path / "sheet.svg"
    svg.write_text("<svg/>")
    assert validate_extraction_request(settings(), svg, {"svg"}) is None


def test_the_suffix_check_ignores_case(tmp_path):
    shouty = tmp_path / "SHEET.PNG"
    shouty.write_bytes(b"x")
    assert validate_extraction_request(settings(), shouty, {"png"}) is None


# --- the input ------------------------------------------------------------

def test_a_missing_input_is_named_in_the_message(tmp_path):
    missing = tmp_path / "absent.png"
    issue = validate_extraction_request(settings(), missing, {"png"})
    assert issue is not None
    assert issue.title == "Missing Input"
    assert str(missing) in issue.message


def test_an_unsupported_suffix_is_rejected(tmp_path):
    """The GUI used to accept this and fail later inside the extractor."""
    jpeg = tmp_path / "sheet.jpg"
    jpeg.write_bytes(b"x")

    issue = validate_extraction_request(settings(), jpeg, {"png"})

    assert issue is not None
    assert issue.title == "Unsupported Input"


def test_a_missing_file_is_reported_before_its_suffix(tmp_path):
    issue = validate_extraction_request(settings(), tmp_path / "absent.jpg", {"png"})
    assert issue.title == "Missing Input"


# --- the request ----------------------------------------------------------

def test_no_export_format_is_rejected(sheet):
    issue = validate_extraction_request(settings(), sheet, set())
    assert issue is not None
    assert issue.title == "No Export Format"


@pytest.mark.parametrize(
    "width,height",
    [(0, 512), (512, 0), (-1, 512), (512, -1), (0, 0)],
)
def test_a_canvas_without_positive_dimensions_is_rejected(sheet, width, height):
    issue = validate_extraction_request(
        settings(output_width=width, output_height=height), sheet, {"png"}
    )
    assert issue is not None
    assert issue.title == "Canvas Size"


# --- the backend ----------------------------------------------------------

def test_a_remote_ollama_endpoint_is_refused(sheet):
    issue = validate_extraction_request(
        settings(provider="ollama", ollama_url="http://example.com:11434"),
        sheet,
        {"png"},
    )
    assert issue is not None
    assert issue.title == "Local Only"
    assert "Ollama" in issue.message


def test_a_remote_llamacpp_endpoint_is_refused(sheet):
    issue = validate_extraction_request(
        settings(provider="llamacpp", llamacpp_url="http://example.com:8080"),
        sheet,
        {"png"},
    )
    assert issue is not None
    assert issue.title == "Local Only"
    assert "llama.cpp" in issue.message


def test_a_remote_endpoint_for_an_inactive_provider_is_not_the_run_s_problem(sheet):
    """Only the selected provider's endpoint gates the run."""
    assert (
        validate_extraction_request(
            settings(provider="geometry", ollama_url="http://example.com:11434"),
            sheet,
            {"png"},
        )
        is None
    )


def test_the_formats_argument_may_be_any_iterable(sheet):
    assert validate_extraction_request(settings(), sheet, ["png", "svg"]) is None
