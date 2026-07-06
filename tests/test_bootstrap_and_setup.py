"""Tests for the model catalog, installer, preflight, and first-run flow."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from services import first_run as fr
from services import model_bootstrap as mb
from services import vision
from services.config_store import AppSettings
from services.llamacpp_client import LlamaCppVisionClient
from services.model_catalog import (
    classify_model_name,
    default_install_specs,
    spec_by_key,
    spec_for_model_name,
)
from services.ollama_client import OllamaVisionClient


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------

def test_catalog_has_default_install_model():
    assert default_install_specs(), "at least one model must be default-installed"


@pytest.mark.parametrize(
    "name,key",
    [
        ("qwen2.5vl:3b", "qwen2.5-vl"),
        ("Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "qwen2.5-vl"),
        ("minicpm-v:latest", "minicpm-v"),
        ("moondream2-text-model-f16.gguf", "moondream"),
        ("llava:7b", "llava"),
    ],
)
def test_spec_for_model_name(name, key):
    spec = spec_for_model_name(name)
    assert spec is not None and spec.key == key


def test_classify_unknown_model():
    priority, _, vision_ok = classify_model_name("totally-unknown-model.gguf")
    assert priority >= 500 and vision_ok is False


# --------------------------------------------------------------------------
# Retry helper
# --------------------------------------------------------------------------

def test_request_with_retries_recovers():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("refused")
        return "ok"

    assert vision.request_with_retries(flaky, sleep=lambda _s: None) == "ok"
    assert calls["n"] == 3


def test_request_with_retries_gives_up():
    def always_fail():
        raise requests.Timeout("slow")

    with pytest.raises(requests.Timeout):
        vision.request_with_retries(always_fail, attempts=2, sleep=lambda _s: None)


def test_request_with_retries_does_not_swallow_other_errors():
    def boom():
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        vision.request_with_retries(boom, sleep=lambda _s: None)


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------

def _resp(json_body, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_body
    r.raise_for_status.return_value = None
    return r


def test_preflight_ollama_unreachable():
    with patch("services.vision.requests.get", side_effect=requests.ConnectionError()):
        result = vision.preflight(AppSettings(provider="ollama", ollama_model="qwen2.5vl:3b"))
    assert not result.ok and "not reachable" in result.message


def test_preflight_ollama_model_missing():
    body = {"models": [{"name": "llama3:8b"}]}
    with patch("services.vision.requests.get", return_value=_resp(body)):
        result = vision.preflight(AppSettings(provider="ollama", ollama_model="qwen2.5vl:3b"))
    assert not result.ok and "not pulled" in result.message


def test_preflight_ollama_vision_ready():
    body = {"models": [{"name": "qwen2.5vl:3b"}]}
    with patch("services.vision.requests.get", return_value=_resp(body)):
        result = vision.preflight(AppSettings(provider="ollama", ollama_model="qwen2.5vl:3b"))
    assert result.ok and result.vision_capable is True


def test_preflight_llamacpp_multimodal_ready():
    body = {"models": [{"id": "Qwen2.5-VL", "capabilities": ["completion", "multimodal"]}]}
    with patch("services.vision.requests.get", return_value=_resp(body)):
        result = vision.preflight(AppSettings(provider="llamacpp"))
    assert result.ok and result.vision_capable is True


def test_preflight_llamacpp_not_multimodal():
    body = {"models": [{"id": "llama-text", "capabilities": ["completion"]}]}
    with patch("services.vision.requests.get", return_value=_resp(body)):
        result = vision.preflight(AppSettings(provider="llamacpp"))
    assert not result.ok and "not multimodal" in result.message


def test_preflight_geometry_needs_nothing():
    assert vision.preflight(AppSettings(provider="geometry")).ok


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def test_ollama_presence_matches_bare_tag():
    body = {"models": [{"name": "qwen2.5vl:3b"}]}
    with patch("services.model_bootstrap.requests.get", return_value=_resp(body)):
        assert mb.is_ollama_model_present("qwen2.5vl:latest", "http://127.0.0.1:11434")
        assert not mb.is_ollama_model_present("moondream:latest", "http://127.0.0.1:11434")


def test_consume_ollama_pull_stream_reports_and_errors():
    events = []
    lines = [
        b'{"status":"pulling manifest"}',
        b'{"status":"downloading","completed":50,"total":100}',
        b'{"status":"success"}',
    ]
    mb._consume_ollama_pull_stream(lines, "moondream:latest", events.append)
    assert any(e.completed == 50 and e.total == 100 for e in events)

    with pytest.raises(mb.BootstrapError):
        mb._consume_ollama_pull_stream([b'{"error":"no such model"}'], "x", None)


def test_resolve_repo_files_picks_weights_and_mmproj():
    spec = spec_by_key("qwen2.5-vl")
    files = [
        "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
        "README.md",
    ]
    with patch("huggingface_hub.list_repo_files", return_value=files):
        main, mmproj = mb.resolve_repo_files(spec)
    assert main == "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
    assert "mmproj" in mmproj


def test_download_file_streams_and_is_idempotent(tmp_path):
    dest = tmp_path / "weights.gguf"
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"Content-Length": "6"}
    resp.iter_content.return_value = [b"abc", b"def"]
    resp.raise_for_status.return_value = None
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    with patch("services.model_bootstrap.requests.get", return_value=ctx) as get:
        path = mb._download_file("http://x/y", dest, "k", None)
        assert path.read_bytes() == b"abcdef"
        # Second call short-circuits (already present) and makes no new request.
        mb._download_file("http://x/y", dest, "k", None)
    assert get.call_count == 1


def test_find_local_llamacpp_files(tmp_path):
    spec = spec_by_key("qwen2.5-vl")
    target = mb.llamacpp_target_dir(str(tmp_path), spec)
    target.mkdir(parents=True)
    (target / "Qwen2.5-VL-Q4.gguf").write_bytes(b"w")
    assert mb.find_local_llamacpp_files(str(tmp_path), spec) is None  # mmproj missing
    (target / "mmproj-Qwen2.5-VL.gguf").write_bytes(b"m")
    files = mb.find_local_llamacpp_files(str(tmp_path), spec)
    assert files is not None and files.mmproj_path.name.startswith("mmproj")


# --------------------------------------------------------------------------
# First-run orchestration
# --------------------------------------------------------------------------

def test_build_candidates_marks_defaults():
    settings = AppSettings()
    status = fr.BackendStatus(ollama_reachable=True, llamacpp_binary=True)
    with patch("services.first_run.mb.is_ollama_model_present", return_value=False), patch(
        "services.first_run.mb.is_llamacpp_model_present", return_value=False
    ):
        candidates = fr.build_candidates(settings, status)
    assert candidates
    # With Ollama running, the default model is pre-selected on Ollama, not llama.cpp.
    default_ollama = [c for c in candidates if c.default_selected]
    assert default_ollama and all(c.backend == "ollama" for c in default_ollama)


def test_apply_active_model_configures_provider():
    settings = AppSettings(provider="geometry", semantic_naming=False)
    cand = fr.SetupCandidate(spec_by_key("qwen2.5-vl"), "ollama", False, 3.2, True)
    fr.apply_active_model(settings, cand)
    assert settings.provider == "ollama"
    assert settings.ollama_model == "qwen2.5vl:3b"
    assert settings.semantic_naming is True


def test_needs_first_run_logic(tmp_path):
    settings = AppSettings(models_root=str(tmp_path))
    with patch("services.first_run.is_setup_marked", return_value=True):
        assert fr.needs_first_run(settings) is False
    with patch("services.first_run.is_setup_marked", return_value=False), patch(
        "services.first_run.any_vision_model_available", return_value=True
    ):
        assert fr.needs_first_run(settings) is False
    with patch("services.first_run.is_setup_marked", return_value=False), patch(
        "services.first_run.any_vision_model_available", return_value=False
    ):
        assert fr.needs_first_run(settings) is True


# --------------------------------------------------------------------------
# Cross-provider parity contract
# --------------------------------------------------------------------------

def _client_and_patch(kind):
    if kind == "ollama":
        return (
            OllamaVisionClient("http://127.0.0.1:11434"),
            "services.ollama_client.requests.post",
            {"response": '{"label":"Light Bulb","tags":["Idea"],"confidence":1.7}'},
        )
    return (
        LlamaCppVisionClient("http://127.0.0.1:8080"),
        "services.llamacpp_client.requests.post",
        {"choices": [{"message": {"content": '{"label":"Light Bulb","tags":["Idea"],"confidence":1.7}'}}]},
    )


@pytest.mark.parametrize("kind", ["ollama", "llamacpp"])
def test_providers_return_identical_shape(kind, tmp_path):
    img = tmp_path / "icon.png"
    img.write_bytes(b"pngdata")
    client, patch_target, body = _client_and_patch(kind)
    with patch(patch_target, return_value=_resp(body)):
        result = client.identify_icon("model", img)
    assert result == {"label": "light-bulb", "tags": ["idea"], "confidence": 1.0}


@pytest.mark.parametrize("kind", ["ollama", "llamacpp"])
def test_providers_raise_alike_on_http_error(kind, tmp_path):
    img = tmp_path / "icon.png"
    img.write_bytes(b"pngdata")
    client, patch_target, _ = _client_and_patch(kind)
    err_resp = MagicMock()
    err_resp.status_code = 500
    err_resp.raise_for_status.side_effect = requests.HTTPError("500")
    with patch(patch_target, return_value=err_resp):
        with pytest.raises(requests.HTTPError):
            client.identify_icon("model", img)
