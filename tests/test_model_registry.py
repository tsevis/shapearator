"""Discovering usable vision models across Ollama, llama.cpp and a directory.

The subprocess and the HTTP call are mocked; the model catalog is not, so the
priority ordering these tests assert is the real one.
"""
from __future__ import annotations

import subprocess

import pytest

from services import model_registry as mr
from services.model_registry import (
    ModelDescriptor,
    ModelRegistry,
    classify_vision_model,
)

OLLAMA_HEADER = "NAME                ID              SIZE      MODIFIED"


@pytest.fixture
def registry():
    return ModelRegistry()


def ollama_output(monkeypatch, stdout=None, raises=None):
    """Patch `ollama list` to return canned output or fail."""

    def fake_run(*_args, **_kwargs):
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(args=["ollama", "list"], returncode=0, stdout=stdout)

    monkeypatch.setattr(mr.subprocess, "run", fake_run)


def models_endpoint(monkeypatch, payload=None, raises=None, status_error=None):
    """Patch the llama.cpp /v1/models probe."""
    calls = []

    class Response:
        def raise_for_status(self):
            if status_error is not None:
                raise status_error

        def json(self):
            if isinstance(payload, Exception):
                raise payload
            return payload

    def fake_get(url, timeout=None):
        calls.append((url, timeout))
        if raises is not None:
            raise raises
        return Response()

    monkeypatch.setattr(mr.requests, "get", fake_get)
    return calls


# --- classification wrapper -----------------------------------------------

def test_classification_defers_to_the_shared_catalog():
    priority, recommendation, supports_vision = classify_vision_model("qwen2.5vl:3b")
    assert (priority, supports_vision) == (2, True)
    assert recommendation


def test_a_model_outside_the_catalog_is_ranked_last_and_not_a_vision_model():
    priority, _recommendation, supports_vision = classify_vision_model("mistral:7b")
    assert priority == 500
    assert supports_vision is False


# --- ollama ---------------------------------------------------------------

def test_installed_ollama_models_are_listed(monkeypatch, registry):
    ollama_output(
        monkeypatch,
        stdout=f"{OLLAMA_HEADER}\nqwen2.5vl:3b   abc   3.2 GB   2 days ago\n",
    )

    models = registry.list_ollama_models()

    assert [m.name for m in models] == ["qwen2.5vl:3b"]
    assert models[0].source == "ollama"
    assert models[0].location == "local ollama"
    assert models[0].supports_vision is True


def test_the_header_row_is_not_mistaken_for_a_model(monkeypatch, registry):
    ollama_output(monkeypatch, stdout=f"{OLLAMA_HEADER}\n")
    assert registry.list_ollama_models() == []


def test_blank_lines_are_skipped(monkeypatch, registry):
    ollama_output(
        monkeypatch,
        stdout=f"{OLLAMA_HEADER}\n\n   \nllava:7b   abc   4 GB   1 day ago\n",
    )
    assert [m.name for m in registry.list_ollama_models()] == ["llava:7b"]


def test_cloud_models_are_left_out(monkeypatch, registry):
    """Only locally served models can name icons on this machine."""
    ollama_output(
        monkeypatch,
        stdout=(
            f"{OLLAMA_HEADER}\n"
            "qwen2.5vl:3b        abc   3.2 GB   2 days ago\n"
            "gpt-oss:120b:cloud  def   -        -\n"
        ),
    )
    assert [m.name for m in registry.list_ollama_models()] == ["qwen2.5vl:3b"]


def test_models_come_back_best_first(monkeypatch, registry):
    ollama_output(
        monkeypatch,
        stdout=(
            f"{OLLAMA_HEADER}\n"
            "llava:7b        a   4 GB     1 day ago\n"
            "mistral:7b      b   4 GB     1 day ago\n"
            "qwen3-vl:8b     c   6 GB     1 day ago\n"
            "qwen2.5vl:3b    d   3.2 GB   1 day ago\n"
        ),
    )

    assert [m.name for m in registry.list_ollama_models()] == [
        "qwen3-vl:8b",
        "qwen2.5vl:3b",
        "llava:7b",
        "mistral:7b",
    ]


def test_equal_ranking_is_broken_by_name(monkeypatch, registry):
    ollama_output(
        monkeypatch,
        stdout=f"{OLLAMA_HEADER}\nzephyr:7b  a  4 GB  1 day ago\nMistral:7b  b  4 GB  1 day ago\n",
    )
    assert [m.name for m in registry.list_ollama_models()] == ["Mistral:7b", "zephyr:7b"]


def test_a_missing_ollama_binary_yields_no_models(monkeypatch, registry):
    ollama_output(monkeypatch, raises=FileNotFoundError("ollama"))
    assert registry.list_ollama_models() == []


def test_an_ollama_that_exits_nonzero_yields_no_models(monkeypatch, registry):
    ollama_output(
        monkeypatch,
        raises=subprocess.CalledProcessError(returncode=1, cmd=["ollama", "list"]),
    )
    assert registry.list_ollama_models() == []


# --- llama.cpp ------------------------------------------------------------

def test_a_served_model_is_listed(monkeypatch, registry):
    models_endpoint(monkeypatch, payload={"data": [{"id": "qwen2.5-vl"}]})

    models = registry.list_llamacpp_models("http://127.0.0.1:8080")

    assert [m.name for m in models] == ["qwen2.5-vl"]
    assert models[0].source == "llamacpp"
    assert models[0].location == "local llama.cpp"


def test_anything_llama_server_loaded_counts_as_vision_capable(monkeypatch, registry):
    """It is loaded and ready, even if it is outside the curated set."""
    models_endpoint(monkeypatch, payload={"data": [{"id": "some-unknown-model"}]})

    models = registry.list_llamacpp_models("http://127.0.0.1:8080")

    assert models[0].supports_vision is True
    assert models[0].priority == 500


def test_the_model_key_is_accepted_when_there_is_no_id(monkeypatch, registry):
    models_endpoint(monkeypatch, payload={"data": [{"model": "llava:7b"}]})
    assert [m.name for m in registry.list_llamacpp_models("http://x")] == ["llava:7b"]


def test_entries_without_a_usable_name_are_skipped(monkeypatch, registry):
    models_endpoint(
        monkeypatch,
        payload={"data": [{"id": ""}, {"id": "   "}, {}, {"id": "llava:7b"}]},
    )
    assert [m.name for m in registry.list_llamacpp_models("http://x")] == ["llava:7b"]


def test_non_object_entries_are_skipped(monkeypatch, registry):
    models_endpoint(monkeypatch, payload={"data": ["a string", 42, {"id": "llava:7b"}]})
    assert [m.name for m in registry.list_llamacpp_models("http://x")] == ["llava:7b"]


def test_a_payload_without_a_data_list_yields_no_models(monkeypatch, registry):
    models_endpoint(monkeypatch, payload={"object": "list"})
    assert registry.list_llamacpp_models("http://x") == []


def test_a_trailing_slash_does_not_double_the_path(monkeypatch, registry):
    calls = models_endpoint(monkeypatch, payload={"data": []})
    registry.list_llamacpp_models("http://127.0.0.1:8080/")
    assert calls[0][0] == "http://127.0.0.1:8080/v1/models"


def test_an_unreachable_server_yields_no_models(monkeypatch, registry):
    models_endpoint(monkeypatch, raises=ConnectionError("refused"))
    assert registry.list_llamacpp_models("http://127.0.0.1:8080") == []


def test_an_error_status_yields_no_models(monkeypatch, registry):
    models_endpoint(monkeypatch, payload={"data": []}, status_error=RuntimeError("503"))
    assert registry.list_llamacpp_models("http://127.0.0.1:8080") == []


def test_an_unparseable_body_yields_no_models(monkeypatch, registry):
    models_endpoint(monkeypatch, payload=ValueError("not json"))
    assert registry.list_llamacpp_models("http://127.0.0.1:8080") == []


# --- picking a recommendation ---------------------------------------------

def descriptor(name, priority, supports_vision):
    return ModelDescriptor(
        name=name,
        source="ollama",
        location="local ollama",
        priority=priority,
        supports_vision=supports_vision,
    )


def test_nothing_installed_means_no_recommendation(registry):
    assert registry.recommended_ollama_model([]) is None
    assert registry.recommended_llamacpp_model([]) is None


def test_the_best_ranked_vision_model_is_recommended(registry):
    models = [
        descriptor("llava:7b", 5, True),
        descriptor("qwen3-vl:8b", 1, True),
    ]
    assert registry.recommended_ollama_model(models).name == "qwen3-vl:8b"


def test_a_vision_model_wins_even_when_something_else_ranks_higher(registry):
    """Naming icons needs sight; a better-ranked text model is no use."""
    models = [
        descriptor("text-model", 1, False),
        descriptor("smolvlm:500m", 9, True),
    ]
    assert registry.recommended_ollama_model(models).name == "smolvlm:500m"


def test_with_no_vision_model_at_all_the_best_of_the_rest_is_offered(registry):
    models = [descriptor("mistral:7b", 500, False), descriptor("llama3:8b", 400, False)]
    assert registry.recommended_ollama_model(models).name == "llama3:8b"


def test_a_ranking_tie_is_broken_by_name(registry):
    models = [descriptor("beta", 2, True), descriptor("Alpha", 2, True)]
    assert registry.recommended_llamacpp_model(models).name == "Alpha"


# --- a plain directory ----------------------------------------------------

@pytest.mark.parametrize("root", ["", "   ", None])
def test_an_unset_directory_yields_no_models(registry, root):
    assert registry.list_directory_models(root) == []


def test_a_directory_that_does_not_exist_yields_no_models(registry, tmp_path):
    assert registry.list_directory_models(str(tmp_path / "absent")) == []


def test_files_are_named_by_stem_and_folders_by_name(registry, tmp_path):
    (tmp_path / "moondream2.gguf").write_bytes(b"weights")
    (tmp_path / "qwen-vl").mkdir()

    models = registry.list_directory_models(str(tmp_path))

    assert [(m.name, m.source) for m in models] == [
        ("moondream2", "directory"),
        ("qwen-vl", "directory"),
    ]


def test_hidden_entries_are_ignored(registry, tmp_path):
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "model.gguf").write_bytes(b"weights")

    assert [m.name for m in registry.list_directory_models(str(tmp_path))] == ["model"]


def test_entries_are_listed_in_case_insensitive_name_order(registry, tmp_path):
    for name in ("zeta.gguf", "Alpha.gguf", "middle.gguf"):
        (tmp_path / name).write_bytes(b"w")

    assert [m.name for m in registry.list_directory_models(str(tmp_path))] == [
        "Alpha",
        "middle",
        "zeta",
    ]


def test_the_full_path_is_recorded_as_the_location(registry, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"w")
    model = registry.list_directory_models(str(tmp_path))[0]
    assert model.location == str(tmp_path / "model.gguf")


def test_a_home_relative_directory_is_expanded(registry, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model.gguf").write_bytes(b"w")

    assert [m.name for m in registry.list_directory_models("~/models")] == ["model"]


def test_a_dangling_symlink_is_neither_file_nor_folder_and_is_skipped(registry, tmp_path):
    """A link to deleted weights should not be offered as a usable model."""
    (tmp_path / "model.gguf").write_bytes(b"w")
    (tmp_path / "broken.gguf").symlink_to(tmp_path / "gone.gguf")

    assert [m.name for m in registry.list_directory_models(str(tmp_path))] == ["model"]
