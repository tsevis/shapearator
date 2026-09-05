"""First-run detection and the install orchestration behind it.

`tests/test_bootstrap_and_setup.py` already covers candidate building with both
backends up, the Ollama side of `apply_active_model`, and the `needs_first_run`
decision table. This file covers the rest: the label a candidate presents, the
partial-backend cases, the install paths, and the setup marker.

The marker is a real file in `config/`. Every test that touches it redirects
`setup_state_path` at a temporary directory, so running this suite never
creates or overwrites the marker on the machine it runs on.
"""
from __future__ import annotations

import json

import pytest

from services import first_run as fr
from services.model_catalog import CATALOG, spec_by_key
from services.model_registry import ModelDescriptor
from services.settings_schema import AppSettings


@pytest.fixture
def marker(tmp_path, monkeypatch):
    """Point the setup marker at a throwaway path and hand it back."""
    path = tmp_path / "config" / "setup_state.json"
    monkeypatch.setattr(fr, "setup_state_path", lambda: path)
    return path


def settings(**overrides) -> AppSettings:
    base = AppSettings()
    for field, value in overrides.items():
        setattr(base, field, value)
    return base


def candidate(backend="ollama", installed=False, approx_gb=3.2, default_selected=False):
    return fr.SetupCandidate(
        spec=spec_by_key("qwen2.5-vl"),
        backend=backend,
        installed=installed,
        approx_gb=approx_gb,
        default_selected=default_selected,
    )


def descriptor(name="qwen2.5vl:3b", supports_vision=True):
    return ModelDescriptor(
        name=name,
        source="ollama",
        location="local ollama",
        supports_vision=supports_vision,
    )


# --- how a candidate presents itself --------------------------------------

def test_a_downloadable_candidate_shows_its_size():
    label = candidate(installed=False, approx_gb=3.2).label
    assert "~3.2 GB download" in label
    assert "Ollama" in label


def test_an_installed_candidate_says_so_instead_of_a_size():
    label = candidate(installed=True, approx_gb=3.2).label
    assert "installed" in label
    assert "download" not in label


def test_a_candidate_of_unknown_size_is_described_as_small():
    """approx_gb is 0 for models whose size the catalog does not record."""
    assert "~small download" in candidate(installed=False, approx_gb=0).label


def test_the_llamacpp_backend_is_named_in_the_label():
    assert "llama.cpp" in candidate(backend="llamacpp").label


def test_the_label_leads_with_the_model_name():
    assert candidate().label.startswith(spec_by_key("qwen2.5-vl").display_name)


# --- which backends are available -----------------------------------------

def test_both_backends_can_be_detected(monkeypatch):
    monkeypatch.setattr(fr.mb, "ollama_reachable", lambda _url: True)
    monkeypatch.setattr(fr, "find_llama_server_binary", lambda: "/bin/llama-server")

    status = fr.detect_backends(settings())

    assert status == fr.BackendStatus(ollama_reachable=True, llamacpp_binary=True)


def test_neither_backend_present_is_reported_plainly(monkeypatch):
    monkeypatch.setattr(fr.mb, "ollama_reachable", lambda _url: False)
    monkeypatch.setattr(fr, "find_llama_server_binary", lambda: None)

    status = fr.detect_backends(settings())

    assert status == fr.BackendStatus(ollama_reachable=False, llamacpp_binary=False)


def test_backends_are_detected_when_none_were_supplied(monkeypatch):
    """build_candidates probes for itself if the caller did not."""
    monkeypatch.setattr(fr.mb, "ollama_reachable", lambda _url: False)
    monkeypatch.setattr(fr, "find_llama_server_binary", lambda: None)

    assert fr.build_candidates(settings()) == []


# --- what gets offered ----------------------------------------------------

def offered(monkeypatch, *, ollama, llamacpp):
    monkeypatch.setattr(fr.mb, "is_ollama_model_present", lambda _tag, _url: False)
    monkeypatch.setattr(fr.mb, "is_llamacpp_model_present", lambda _root, _spec: False)
    status = fr.BackendStatus(ollama_reachable=ollama, llamacpp_binary=llamacpp)
    return fr.build_candidates(settings(), status)


def test_with_only_llamacpp_every_option_is_a_llamacpp_one(monkeypatch):
    candidates = offered(monkeypatch, ollama=False, llamacpp=True)

    assert candidates
    assert {c.backend for c in candidates} == {"llamacpp"}


def test_with_only_ollama_nothing_llamacpp_is_offered(monkeypatch):
    candidates = offered(monkeypatch, ollama=True, llamacpp=False)

    assert candidates
    assert {c.backend for c in candidates} == {"ollama"}


def test_the_default_model_is_preselected_on_ollama_when_it_is_running(monkeypatch):
    selected = [c for c in offered(monkeypatch, ollama=True, llamacpp=True) if c.default_selected]

    assert [c.backend for c in selected] == ["ollama"], "one click on the preferred backend"


def test_the_default_falls_to_llamacpp_when_ollama_is_not_running(monkeypatch):
    selected = [c for c in offered(monkeypatch, ollama=False, llamacpp=True) if c.default_selected]

    assert [c.backend for c in selected] == ["llamacpp"]


def test_a_model_without_an_ollama_tag_is_not_offered_for_ollama(monkeypatch):
    """Some catalog entries are GGUF-only."""
    candidates = offered(monkeypatch, ollama=True, llamacpp=False)
    offered_keys = {c.spec.key for c in candidates}
    taggable = {spec.key for spec in CATALOG if spec.ollama_tag}

    assert offered_keys == taggable


# --- installing -----------------------------------------------------------

def test_an_ollama_candidate_is_pulled(monkeypatch):
    pulled = []
    monkeypatch.setattr(fr.mb, "pull_ollama_model", lambda tag, url, cb: pulled.append(tag))
    monkeypatch.setattr(
        fr.mb, "download_llamacpp_model", lambda *_a: pytest.fail("wrong backend")
    )

    fr.install_candidate(settings(), candidate(backend="ollama"))

    assert pulled == [spec_by_key("qwen2.5-vl").ollama_tag]


def test_a_llamacpp_candidate_is_downloaded(monkeypatch):
    downloaded = []
    monkeypatch.setattr(
        fr.mb, "download_llamacpp_model", lambda spec, root, cb: downloaded.append(spec.key)
    )
    monkeypatch.setattr(fr.mb, "pull_ollama_model", lambda *_a: pytest.fail("wrong backend"))

    fr.install_candidate(settings(), candidate(backend="llamacpp"))

    assert downloaded == ["qwen2.5-vl"]


def test_the_progress_callback_is_handed_to_the_downloader(monkeypatch):
    seen = []
    monkeypatch.setattr(fr.mb, "pull_ollama_model", lambda _t, _u, cb: seen.append(cb))

    sentinel = object()
    fr.install_candidate(settings(), candidate(), progress_callback=sentinel)

    assert seen == [sentinel]


def test_a_selection_installs_only_what_is_missing(monkeypatch):
    installed = []
    monkeypatch.setattr(
        fr, "install_candidate", lambda _s, cand, _cb=None: installed.append(cand.backend)
    )

    fr.install_selection(
        settings(),
        [
            candidate(backend="ollama", installed=True),
            candidate(backend="llamacpp", installed=False),
        ],
    )

    assert installed == ["llamacpp"], "an installed model must not be re-downloaded"


def test_installing_an_empty_selection_does_nothing(monkeypatch):
    monkeypatch.setattr(fr, "install_candidate", lambda *_a, **_k: pytest.fail("nothing to do"))
    fr.install_selection(settings(), [])


# --- pointing the app at what was installed -------------------------------

def test_choosing_a_llamacpp_model_switches_provider_and_enables_naming():
    result = fr.apply_active_model(settings(semantic_naming=False), candidate(backend="llamacpp"))

    assert result.provider == "llamacpp"
    assert result.llamacpp_model == spec_by_key("qwen2.5-vl").display_name
    assert result.semantic_naming is True


# --- the setup marker -----------------------------------------------------

def test_no_marker_means_setup_has_not_run(marker):
    assert fr.is_setup_marked() is False


def test_writing_the_marker_records_when_it_happened(marker):
    fr.mark_setup_complete()

    assert fr.is_setup_marked() is True
    assert "completed_at" in json.loads(marker.read_text())


def test_the_marker_directory_is_created_if_missing(marker):
    assert not marker.parent.exists()

    fr.mark_setup_complete()

    assert marker.exists()


def test_extra_details_are_recorded_alongside_the_timestamp(marker):
    fr.mark_setup_complete({"installed": ["qwen2.5vl:3b"]})

    payload = json.loads(marker.read_text())
    assert payload["installed"] == ["qwen2.5vl:3b"]
    assert "completed_at" in payload


def test_marking_twice_overwrites_rather_than_appends(marker):
    fr.mark_setup_complete({"run": "first"})
    fr.mark_setup_complete({"run": "second"})

    assert json.loads(marker.read_text())["run"] == "second"


# --- is any model already usable ------------------------------------------

def test_an_installed_ollama_vision_model_counts(monkeypatch):
    monkeypatch.setattr(fr.ModelRegistry, "list_ollama_models", lambda self: [descriptor()])
    monkeypatch.setattr(fr.mb, "is_llamacpp_model_present", lambda *_a: False)

    assert fr.any_vision_model_available(settings()) is True


def test_an_ollama_model_that_cannot_see_does_not_count(monkeypatch):
    """A text-only model is no use for naming icons."""
    monkeypatch.setattr(
        fr.ModelRegistry,
        "list_ollama_models",
        lambda self: [descriptor("mistral:7b", supports_vision=False)],
    )
    monkeypatch.setattr(fr.mb, "is_llamacpp_model_present", lambda *_a: False)

    assert fr.any_vision_model_available(settings()) is False


def test_a_downloaded_gguf_counts_even_with_no_ollama(monkeypatch):
    monkeypatch.setattr(fr.ModelRegistry, "list_ollama_models", lambda self: [])
    monkeypatch.setattr(fr.mb, "is_llamacpp_model_present", lambda *_a: True)

    assert fr.any_vision_model_available(settings()) is True


def test_nothing_installed_anywhere_means_nothing_is_available(monkeypatch):
    monkeypatch.setattr(fr.ModelRegistry, "list_ollama_models", lambda self: [])
    monkeypatch.setattr(fr.mb, "is_llamacpp_model_present", lambda *_a: False)

    assert fr.any_vision_model_available(settings()) is False


# --- the decision that ties it together -----------------------------------

def test_setup_is_offered_when_unmarked_and_nothing_is_installed(marker, monkeypatch):
    monkeypatch.setattr(fr.ModelRegistry, "list_ollama_models", lambda self: [])
    monkeypatch.setattr(fr.mb, "is_llamacpp_model_present", lambda *_a: False)

    assert fr.needs_first_run(settings()) is True


def test_setup_is_not_offered_again_once_marked(marker, monkeypatch):
    monkeypatch.setattr(fr.ModelRegistry, "list_ollama_models", lambda self: [])
    monkeypatch.setattr(fr.mb, "is_llamacpp_model_present", lambda *_a: False)
    fr.mark_setup_complete()

    assert fr.needs_first_run(settings()) is False
