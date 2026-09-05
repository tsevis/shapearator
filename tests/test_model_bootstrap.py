"""Model downloading: reachability, the Ollama pull stream, and resumable HTTP.

`tests/test_bootstrap_and_setup.py` covers the catalog and the retry helper.
This file covers the downloader itself — most importantly `_download_file`,
which resumes a partial transfer and has to cope with a server that ignores the
Range header it was given.

Nothing here reaches the network or Hugging Face.
"""
from __future__ import annotations

import json

import pytest

from services import model_bootstrap as mb
from services.model_bootstrap import (
    BootstrapError,
    BootstrapProgress,
    _consume_ollama_pull_stream,
    _download_file,
    is_ollama_model_present,
    ollama_installed_tags,
    ollama_reachable,
)
from services.model_catalog import spec_by_key

SPEC = spec_by_key("qwen2.5-vl")


class FakeResponse:
    """Enough of a requests.Response for the streaming download path."""

    def __init__(self, status_code=200, headers=None, chunks=(), raises=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks
        self._raises = raises

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def raise_for_status(self):
        if self._raises is not None:
            raise self._raises

    def iter_content(self, chunk_size=None):
        yield from self._chunks

    def json(self):
        return self._json

    def with_json(self, payload):
        self._json = payload
        return self


def http(monkeypatch, response=None, raises=None):
    """Patch requests.get and record the calls made."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if raises is not None:
            raise raises
        return response

    monkeypatch.setattr(mb.requests, "get", fake_get)
    return calls


def collector():
    events = []
    return events, events.append


# --- progress arithmetic --------------------------------------------------

def test_progress_with_an_unknown_total_reports_nothing_done():
    """A stream without Content-Length must not divide by zero."""
    assert BootstrapProgress("llamacpp", "k", "download", 5, 0, "").fraction == 0.0


def test_progress_is_the_completed_share_of_the_total():
    assert BootstrapProgress("llamacpp", "k", "download", 1, 4, "").fraction == 0.25


def test_progress_never_exceeds_one():
    assert BootstrapProgress("llamacpp", "k", "download", 9, 4, "").fraction == 1.0


# --- is Ollama there ------------------------------------------------------

def test_a_responding_daemon_is_reachable(monkeypatch):
    calls = http(monkeypatch, FakeResponse(status_code=200))
    assert ollama_reachable("http://127.0.0.1:11434") is True
    assert calls[0][0] == "http://127.0.0.1:11434/api/version"


def test_a_trailing_slash_does_not_double_the_path(monkeypatch):
    calls = http(monkeypatch, FakeResponse(status_code=200))
    ollama_reachable("http://127.0.0.1:11434/")
    assert calls[0][0] == "http://127.0.0.1:11434/api/version"


def test_an_error_status_means_not_reachable(monkeypatch):
    http(monkeypatch, FakeResponse(status_code=500))
    assert ollama_reachable("http://127.0.0.1:11434") is False


def test_a_refused_connection_means_not_reachable(monkeypatch):
    http(monkeypatch, raises=ConnectionError("refused"))
    assert ollama_reachable("http://127.0.0.1:11434") is False


# --- what Ollama already has ----------------------------------------------

def tags_response(monkeypatch, payload, raises=None):
    response = FakeResponse(raises=raises).with_json(payload)
    return http(monkeypatch, response)


def test_installed_tags_are_collected(monkeypatch):
    tags_response(monkeypatch, {"models": [{"name": "qwen2.5vl:3b"}, {"name": "llava:7b"}]})
    assert ollama_installed_tags("http://x") == {"qwen2.5vl:3b", "llava:7b"}


def test_malformed_entries_are_ignored(monkeypatch):
    tags_response(monkeypatch, {"models": ["a string", {}, {"name": ""}, {"name": "llava:7b"}]})
    assert ollama_installed_tags("http://x") == {"llava:7b"}


def test_a_payload_without_models_yields_nothing(monkeypatch):
    tags_response(monkeypatch, {})
    assert ollama_installed_tags("http://x") == set()


def test_an_unreachable_daemon_yields_no_tags(monkeypatch):
    http(monkeypatch, raises=ConnectionError("refused"))
    assert ollama_installed_tags("http://x") == set()


def test_an_error_status_yields_no_tags(monkeypatch):
    tags_response(monkeypatch, {"models": []}, raises=RuntimeError("500"))
    assert ollama_installed_tags("http://x") == set()


# --- is a particular model installed --------------------------------------

def test_an_empty_tag_is_never_present():
    assert is_ollama_model_present("", "http://x") is False


def test_an_exactly_matching_tag_is_present(monkeypatch):
    monkeypatch.setattr(mb, "ollama_installed_tags", lambda _url: {"qwen2.5vl:3b"})
    assert is_ollama_model_present("qwen2.5vl:3b", "http://x") is True


def test_the_implicit_latest_suffix_is_tolerated(monkeypatch):
    """`ollama pull llava` installs `llava:latest`; asking for `llava` must match."""
    monkeypatch.setattr(mb, "ollama_installed_tags", lambda _url: {"llava:latest"})
    assert is_ollama_model_present("llava", "http://x") is True


def test_a_different_model_is_not_a_match(monkeypatch):
    monkeypatch.setattr(mb, "ollama_installed_tags", lambda _url: {"mistral:7b"})
    assert is_ollama_model_present("llava:7b", "http://x") is False


# --- reading the pull stream ----------------------------------------------

def lines(*events):
    return [json.dumps(e).encode() for e in events]


def test_each_stream_event_becomes_progress():
    events, callback = collector()

    _consume_ollama_pull_stream(
        lines({"status": "pulling", "completed": 1, "total": 4}), "llava:7b", callback
    )

    assert [(e.completed, e.total, e.message) for e in events] == [(1, 4, "pulling")]


def test_an_event_without_a_status_still_names_the_model():
    events, callback = collector()

    _consume_ollama_pull_stream(lines({"completed": 0}), "llava:7b", callback)

    assert events[0].message == "Pulling llava:7b"


def test_blank_lines_are_skipped():
    events, callback = collector()
    _consume_ollama_pull_stream([b"", b""], "llava:7b", callback)
    assert events == []


def test_unparseable_lines_are_skipped():
    """A keep-alive or truncated frame must not abort a long pull."""
    events, callback = collector()

    _consume_ollama_pull_stream([b"not json", *lines({"status": "ok"})], "llava:7b", callback)

    assert [e.message for e in events] == ["ok"]


def test_an_error_event_stops_the_pull():
    with pytest.raises(BootstrapError, match="no space"):
        _consume_ollama_pull_stream(lines({"error": "no space left"}), "llava:7b", None)


def test_a_stream_without_a_callback_is_fine():
    _consume_ollama_pull_stream(lines({"status": "ok"}), "llava:7b", None)


# --- downloading a file ---------------------------------------------------

def test_a_fresh_download_is_written_and_reported(monkeypatch, tmp_path):
    http(monkeypatch, FakeResponse(headers={"Content-Length": "6"}, chunks=[b"abc", b"def"]))
    events, callback = collector()
    dest = tmp_path / "model.gguf"

    result = _download_file("http://h/f", dest, "qwen2.5-vl", callback)

    assert result == dest
    assert dest.read_bytes() == b"abcdef"
    assert [e.completed for e in events] == [3, 6]
    assert events[-1].total == 6


def test_an_already_downloaded_file_is_not_fetched_again(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mb.requests, "get", lambda *_a, **_k: pytest.fail("should not have downloaded")
    )
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"already here")

    assert _download_file("http://h/f", dest, "k", None) == dest


def test_an_empty_file_is_downloaded_again(monkeypatch, tmp_path):
    """A zero-byte leftover is a failed attempt, not a finished download."""
    http(monkeypatch, FakeResponse(headers={"Content-Length": "3"}, chunks=[b"abc"]))
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"")

    assert _download_file("http://h/f", dest, "k", None).read_bytes() == b"abc"


def test_a_partial_transfer_resumes_where_it_stopped(monkeypatch, tmp_path):
    calls = http(monkeypatch, FakeResponse(status_code=206, headers={"Content-Length": "3"}, chunks=[b"def"]))
    dest = tmp_path / "model.gguf"
    dest.with_name("model.gguf.part").write_bytes(b"abc")

    result = _download_file("http://h/f", dest, "k", None)

    assert calls[0][1]["headers"] == {"Range": "bytes=3-"}
    assert result.read_bytes() == b"abcdef", "resumed bytes appended, not overwritten"


def test_a_server_ignoring_the_range_header_starts_over(monkeypatch, tmp_path):
    """A 200 to a Range request means the whole file is coming again."""
    http(monkeypatch, FakeResponse(status_code=200, headers={"Content-Length": "6"}, chunks=[b"abcdef"]))
    dest = tmp_path / "model.gguf"
    dest.with_name("model.gguf.part").write_bytes(b"XXX")

    result = _download_file("http://h/f", dest, "k", None)

    assert result.read_bytes() == b"abcdef", "stale partial discarded rather than appended"


def test_a_range_already_satisfied_completes_the_file(monkeypatch, tmp_path):
    """416 means the .part is already the whole file."""
    http(monkeypatch, FakeResponse(status_code=416))
    dest = tmp_path / "model.gguf"
    dest.with_name("model.gguf.part").write_bytes(b"complete")

    result = _download_file("http://h/f", dest, "k", None)

    assert result.read_bytes() == b"complete"
    assert not dest.with_name("model.gguf.part").exists()


def test_empty_chunks_do_not_count_towards_progress(monkeypatch, tmp_path):
    http(monkeypatch, FakeResponse(headers={"Content-Length": "3"}, chunks=[b"", b"abc", b""]))
    events, callback = collector()

    _download_file("http://h/f", tmp_path / "m.gguf", "k", callback)

    assert [e.completed for e in events] == [3]


def test_the_partial_file_is_only_renamed_once_complete(monkeypatch, tmp_path):
    http(monkeypatch, FakeResponse(headers={"Content-Length": "3"}, chunks=[b"abc"]))
    dest = tmp_path / "model.gguf"

    _download_file("http://h/f", dest, "k", None)

    assert dest.exists()
    assert not dest.with_name("model.gguf.part").exists()


def test_a_failed_download_is_reported_and_raised(monkeypatch, tmp_path):
    http(monkeypatch, raises=ConnectionError("connection reset"))
    events, callback = collector()

    with pytest.raises(BootstrapError, match="Download failed"):
        _download_file("http://h/f", tmp_path / "model.gguf", "qwen2.5-vl", callback)

    assert [e.phase for e in events] == ["error"]
    assert "connection reset" in events[0].message


def test_a_failed_download_leaves_no_finished_file(monkeypatch, tmp_path):
    """A half-written file must never be mistaken for a usable model."""
    http(monkeypatch, FakeResponse(raises=RuntimeError("503")))
    dest = tmp_path / "model.gguf"

    with pytest.raises(BootstrapError):
        _download_file("http://h/f", dest, "k", None)

    assert not dest.exists()


def test_the_destination_directory_is_created(monkeypatch, tmp_path):
    http(monkeypatch, FakeResponse(headers={"Content-Length": "3"}, chunks=[b"abc"]))
    dest = tmp_path / "nested" / "deeper" / "model.gguf"

    assert _download_file("http://h/f", dest, "k", None).exists()


# --- finding what is already downloaded -----------------------------------

def install(root, spec, weights="model-Q4_K_M.gguf", mmproj="mmproj-f16.gguf"):
    target = mb.llamacpp_target_dir(str(root), spec)
    target.mkdir(parents=True, exist_ok=True)
    for name in filter(None, (weights, mmproj)):
        (target / name).write_bytes(b"weights")
    return target


def test_nothing_downloaded_is_reported_as_absent(tmp_path):
    assert mb.find_local_llamacpp_files(str(tmp_path), SPEC) is None
    assert mb.is_llamacpp_model_present(str(tmp_path), SPEC) is False


def test_both_files_present_makes_a_model_usable(tmp_path):
    install(tmp_path, SPEC)

    files = mb.find_local_llamacpp_files(str(tmp_path), SPEC)

    assert files is not None
    assert "mmproj" in files.mmproj_path.name
    assert "mmproj" not in files.gguf_path.name
    assert mb.is_llamacpp_model_present(str(tmp_path), SPEC) is True


def test_weights_without_a_projector_are_not_usable(tmp_path):
    """A vision model needs its mmproj; weights alone cannot see."""
    install(tmp_path, SPEC, mmproj=None)

    assert mb.find_local_llamacpp_files(str(tmp_path), SPEC) is None


def test_a_projector_without_weights_is_not_usable(tmp_path):
    install(tmp_path, SPEC, weights=None)

    assert mb.find_local_llamacpp_files(str(tmp_path), SPEC) is None


def test_downloaded_models_are_listed_best_first(tmp_path):
    from services.model_catalog import CATALOG

    two = sorted(CATALOG, key=lambda s: s.priority)[:2]
    for spec in reversed(two):
        install(tmp_path, spec)

    listed = mb.list_downloaded_llamacpp(str(tmp_path))

    assert [spec.key for spec, _files in listed] == [s.key for s in two]


def test_an_already_installed_model_is_not_downloaded_again(tmp_path):
    install(tmp_path, SPEC)
    events, callback = collector()

    result = mb.download_llamacpp_model(SPEC, str(tmp_path), callback)

    assert result.gguf_path.exists()
    assert [e.phase for e in events] == ["done"]
    assert "already installed" in events[0].message


# --- pulling from Ollama --------------------------------------------------

class FakePost(FakeResponse):
    def __init__(self, lines=(), **kwargs):
        super().__init__(**kwargs)
        self._lines = lines

    def iter_lines(self):
        yield from self._lines


def post(monkeypatch, response=None, raises=None):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if raises is not None:
            raise raises
        return response

    monkeypatch.setattr(mb.requests, "post", fake_post)
    return calls


def test_a_pull_reports_resolve_then_progress_then_done(monkeypatch):
    post(monkeypatch, FakePost(lines=lines({"status": "downloading", "completed": 1, "total": 2})))
    events, callback = collector()

    mb.pull_ollama_model("llava:7b", "http://127.0.0.1:11434", callback)

    assert [e.phase for e in events] == ["resolve", "download", "done"]
    assert events[-1].message == "llava:7b ready"


def test_a_pull_asks_ollama_to_stream(monkeypatch):
    calls = post(monkeypatch, FakePost())

    mb.pull_ollama_model("llava:7b", "http://127.0.0.1:11434/")

    url, kwargs = calls[0]
    assert url == "http://127.0.0.1:11434/api/pull"
    assert kwargs["json"] == {"model": "llava:7b", "stream": True}
    assert kwargs["stream"] is True


def test_pulling_without_a_tag_is_refused_before_any_request(monkeypatch):
    monkeypatch.setattr(mb.requests, "post", lambda *_a, **_k: pytest.fail("no request expected"))

    with pytest.raises(BootstrapError, match="No Ollama tag"):
        mb.pull_ollama_model("", "http://x")


def test_a_failed_pull_is_reported_and_wrapped(monkeypatch):
    post(monkeypatch, raises=ConnectionError("daemon went away"))
    events, callback = collector()

    with pytest.raises(BootstrapError, match="Ollama pull failed"):
        mb.pull_ollama_model("llava:7b", "http://x", callback)

    assert [e.phase for e in events] == ["resolve", "error"]


def test_an_error_inside_the_stream_is_not_rewrapped(monkeypatch):
    """The server's own message is more useful than a generic wrapper."""
    post(monkeypatch, FakePost(lines=lines({"error": "no space left"})))

    with pytest.raises(BootstrapError, match="no space left"):
        mb.pull_ollama_model("llava:7b", "http://x")


# --- resolving filenames in a Hugging Face repo ---------------------------

def repo_listing(monkeypatch, files, raises=None):
    import huggingface_hub

    def fake_list(repo):
        if raises is not None:
            raise raises
        return files

    monkeypatch.setattr(huggingface_hub, "list_repo_files", fake_list)


def test_the_requested_quantization_is_preferred(monkeypatch):
    repo_listing(
        monkeypatch,
        ["Model-Q8_0.gguf", "Model-Q4_K_M.gguf", "mmproj-f16.gguf", "README.md"],
    )

    main, mmproj = mb.resolve_repo_files(SPEC)

    assert main == "Model-Q4_K_M.gguf"
    assert mmproj == "mmproj-f16.gguf"


def test_an_unavailable_quantization_falls_back_to_a_smaller_file(monkeypatch):
    """Upstream renames quants; a missing one must not block the download."""
    repo_listing(monkeypatch, ["Model-Q8_0.gguf", "Model-f16-huge.gguf", "mmproj.gguf"])

    main, _mmproj = mb.resolve_repo_files(SPEC)

    assert main == "Model-Q8_0.gguf"


def test_a_repo_without_a_projector_is_refused(monkeypatch):
    repo_listing(monkeypatch, ["Model-Q4_K_M.gguf"])

    with pytest.raises(BootstrapError, match="mmproj"):
        mb.resolve_repo_files(SPEC)


def test_a_repo_without_weights_is_refused(monkeypatch):
    repo_listing(monkeypatch, ["mmproj-f16.gguf"])

    with pytest.raises(BootstrapError, match="weights or mmproj"):
        mb.resolve_repo_files(SPEC)


def test_an_unreachable_repo_is_reported_with_its_name(monkeypatch):
    repo_listing(monkeypatch, [], raises=OSError("offline"))

    with pytest.raises(BootstrapError, match=SPEC.hf_repo):
        mb.resolve_repo_files(SPEC)


# --- the full llama.cpp download ------------------------------------------

def test_a_download_fetches_weights_then_projector(monkeypatch, tmp_path):
    repo_listing(monkeypatch, ["Model-Q4_K_M.gguf", "mmproj-f16.gguf"])
    fetched = []

    def fake_download(url, dest, key, callback, **_kw):
        fetched.append(dest.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"weights")
        return dest

    monkeypatch.setattr(mb, "_download_file", fake_download)
    events, callback = collector()

    files = mb.download_llamacpp_model(SPEC, str(tmp_path), callback)

    assert fetched == ["Model-Q4_K_M.gguf", "mmproj-f16.gguf"]
    assert files.gguf_path.exists() and files.mmproj_path.exists()
    assert [e.phase for e in events] == ["resolve", "done"]
    assert events[-1].message.endswith("ready")
