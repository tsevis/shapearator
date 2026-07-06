"""Tests for the multi-provider (Ollama + llama.cpp) vision layer."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import shapearator as cli
from services.config_store import AppSettings
from services.llamacpp_client import LlamaCppVisionClient
from services.model_registry import ModelRegistry, classify_vision_model
from services.ollama_client import OllamaVisionClient
from services import vision


# --- shared helpers -------------------------------------------------------

def test_is_local_url_accepts_local_hosts():
    assert vision.is_local_url("http://127.0.0.1:8080")
    assert vision.is_local_url("http://localhost:11434")
    assert vision.is_local_url("http://[::1]:8080")


def test_is_local_url_rejects_remote_hosts():
    assert not vision.is_local_url("http://8.8.8.8:8080")
    assert not vision.is_local_url("https://example.com")


def test_parse_semantic_response_extracts_and_clamps():
    result = vision.parse_semantic_response('here you go {"label":"Light Bulb","tags":["Idea"],"confidence":1.7}')
    # label is hyphenated for filesystem safety; confidence clamped to [0, 1]
    assert result == {"label": "light-bulb", "tags": ["idea"], "confidence": 1.0}


def test_parse_semantic_response_caps_rambling_label():
    rambling = "the image shows a screenshot of a web page with a form for uploading files."
    label = vision.parse_semantic_response(rambling)["label"]
    assert label.count("-") <= 3 and len(label) <= 60 and "image-shows" not in label


def test_parse_semantic_response_caps_sentence_inside_json():
    label = vision.parse_semantic_response('{"label":"a hand drawn smiley face with two round eyes"}')["label"]
    assert label.count("-") <= 3


def test_parse_semantic_response_falls_back_to_first_line():
    result = vision.parse_semantic_response("magnifier glass\nextra")
    assert result["label"] == "magnifier-glass"
    assert result["tags"] == []


@pytest.mark.parametrize(
    "model,expected_marker",
    [
        ("moondream2-q4.gguf", "Return only JSON"),
        ("minicpm-v:latest", "most likely simple icon"),
        ("llava-v1.6.gguf", "concise and generic"),
        ("qwen2.5-vl-3b.gguf", "short-lowercase-filename-label"),
    ],
)
def test_build_icon_prompt_matches_family(model, expected_marker):
    assert expected_marker in vision.build_icon_prompt(model)


def test_image_mime_type_by_suffix():
    assert vision.image_mime_type(Path("a.png")) == "image/png"
    assert vision.image_mime_type(Path("a.jpg")) == "image/jpeg"
    assert vision.image_mime_type(Path("a.unknown")) == "image/png"


# --- factory selection ----------------------------------------------------

def test_factory_selects_llamacpp():
    settings = AppSettings(provider="llamacpp", llamacpp_url="http://127.0.0.1:8080")
    client = vision.build_vision_client(settings)
    assert isinstance(client, LlamaCppVisionClient)


def test_factory_selects_ollama():
    settings = AppSettings(provider="ollama")
    assert isinstance(vision.build_vision_client(settings), OllamaVisionClient)


def test_active_model_and_semantic_gate():
    llamacpp = AppSettings(provider="llamacpp", llamacpp_model="qwen", semantic_naming=True)
    assert vision.active_vision_model(llamacpp) == "qwen"
    assert vision.semantic_naming_enabled(llamacpp)

    geometry = AppSettings(provider="geometry", semantic_naming=True)
    assert not vision.semantic_naming_enabled(geometry)


# --- llama.cpp client -----------------------------------------------------

def test_llamacpp_client_rejects_remote():
    with pytest.raises(ValueError):
        LlamaCppVisionClient("http://example.com:8080")


def test_llamacpp_client_builds_openai_payload(tmp_path):
    image = tmp_path / "icon.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfakepngbytes")

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": '{"label":"heart","tags":["love"],"confidence":0.9}'}}]
    }
    fake_response.raise_for_status.return_value = None

    with patch("services.llamacpp_client.requests.post", return_value=fake_response) as mock_post:
        client = LlamaCppVisionClient("http://127.0.0.1:8080")
        result = client.identify_icon("qwen2.5-vl", image)

    assert result == {"label": "heart", "tags": ["love"], "confidence": 0.9}
    url, kwargs = mock_post.call_args[0][0], mock_post.call_args[1]
    assert url == "http://127.0.0.1:8080/v1/chat/completions"
    content = kwargs["json"]["messages"][0]["content"]
    text_part = next(p for p in content if p["type"] == "text")
    image_part = next(p for p in content if p["type"] == "image_url")
    assert "JSON" in text_part["text"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_llamacpp_client_handles_list_content(tmp_path):
    image = tmp_path / "icon.png"
    image.write_bytes(b"bytes")
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": [{"type": "text", "text": '{"label":"folder"}'}]}}]
    }
    fake_response.raise_for_status.return_value = None
    with patch("services.llamacpp_client.requests.post", return_value=fake_response):
        client = LlamaCppVisionClient("http://127.0.0.1:8080")
        assert client.identify_icon("m", image)["label"] == "folder"


# --- registry discovery ---------------------------------------------------

@pytest.mark.parametrize(
    "name,priority",
    [
        ("Qwen2.5-VL-3B-Instruct-Q4.gguf", 1),
        ("qwen2.5vl:3b", 1),
        ("minicpm-v:latest", 2),
        ("moondream2.gguf", 3),
        ("llava-v1.6.gguf", 4),
        ("random-model.gguf", 500),
    ],
)
def test_classify_vision_model(name, priority):
    assert classify_vision_model(name)[0] == priority


def test_list_llamacpp_models_parses_v1_models():
    fake_response = MagicMock()
    fake_response.json.return_value = {"data": [{"id": "Qwen2.5-VL-3B.gguf"}, {"id": ""}]}
    fake_response.raise_for_status.return_value = None
    with patch("services.model_registry.requests.get", return_value=fake_response):
        models = ModelRegistry().list_llamacpp_models("http://127.0.0.1:8080")
    assert len(models) == 1
    assert models[0].source == "llamacpp"
    assert models[0].supports_vision


def test_list_llamacpp_models_returns_empty_when_server_down():
    with patch("services.model_registry.requests.get", side_effect=OSError("refused")):
        assert ModelRegistry().list_llamacpp_models("http://127.0.0.1:8080") == []


# --- CLI validation -------------------------------------------------------

def test_cli_rejects_remote_llamacpp_endpoint(tmp_path):
    png = tmp_path / "sheet.png"
    png.write_bytes(b"data")
    settings = AppSettings(provider="llamacpp", llamacpp_url="http://8.8.8.8:8080")
    with pytest.raises(SystemExit, match="llama.cpp provider requires a local endpoint"):
        cli.validate_settings(settings, png, {"png"})


def test_cli_accepts_local_llamacpp_endpoint(tmp_path):
    png = tmp_path / "sheet.png"
    png.write_bytes(b"data")
    settings = AppSettings(provider="llamacpp", llamacpp_url="http://127.0.0.1:8080")
    cli.validate_settings(settings, png, {"png"})  # should not raise
