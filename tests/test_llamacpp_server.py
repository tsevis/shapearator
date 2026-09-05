"""Launching, probing and stopping a local llama-server.

Every subprocess, socket and sleep is mocked: no server is started, nothing is
downloaded, and the suite does not wait on a clock.

Managers built here do register their `stop` with atexit, as they do in the
app. That is left alone rather than patched out: `atexit` is a shared module,
so neutralising it would also swallow registrations made by pytest and its
plugins. The handlers are harmless — every process here is a fake, and `stop`
on a manager that launched nothing returns immediately.
"""
from __future__ import annotations

import subprocess

import pytest

from services import llamacpp_server as ls
from services.llamacpp_server import (
    LlamaCppServerManager,
    find_llama_server_binary,
    is_server_healthy,
)


class FakeProcess:
    """A Popen stand-in whose exit state the test drives."""

    def __init__(self, exit_code=None, wait_times_out=False):
        self.exit_code = exit_code          # None means still running
        self.wait_times_out = wait_times_out
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True
        if not self.wait_times_out:
            self.exit_code = 0

    def wait(self, timeout=None):
        if self.wait_times_out:
            raise subprocess.TimeoutExpired(cmd="llama-server", timeout=timeout)
        return self.exit_code

    def kill(self):
        self.killed = True
        self.exit_code = -9


def responder(monkeypatch, **by_endpoint):
    """Patch requests.get so each endpoint returns a status or raises.

    Values are status codes, or exceptions to raise. Records the calls made.
    """
    calls = []

    def fake_get(url, timeout=None):
        calls.append((url, timeout))
        endpoint = "health" if url.endswith("/health") else "models"
        outcome = by_endpoint.get(endpoint, 404)
        if isinstance(outcome, Exception):
            raise outcome
        response = type("Response", (), {"status_code": outcome})()
        return response

    monkeypatch.setattr(ls.requests, "get", fake_get)
    return calls


# --- finding the binary ---------------------------------------------------

def test_the_binary_is_looked_up_on_path(monkeypatch):
    monkeypatch.setattr(ls.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    assert find_llama_server_binary() == "/usr/local/bin/llama-server"


def test_a_missing_binary_reports_nothing(monkeypatch):
    monkeypatch.setattr(ls.shutil, "which", lambda _name: None)
    assert find_llama_server_binary() is None


# --- health probing -------------------------------------------------------

def test_a_healthy_health_endpoint_settles_it(monkeypatch):
    calls = responder(monkeypatch, health=200)
    assert is_server_healthy("http://127.0.0.1:8080") is True
    assert len(calls) == 1, "the fallback endpoint should not be needed"


def test_a_build_without_health_falls_back_to_the_models_endpoint(monkeypatch):
    """Some llama.cpp builds expose only /v1/models."""
    responder(monkeypatch, health=404, models=200)
    assert is_server_healthy("http://127.0.0.1:8080") is True


def test_a_refused_health_probe_still_tries_the_fallback(monkeypatch):
    responder(monkeypatch, health=ConnectionError("refused"), models=200)
    assert is_server_healthy("http://127.0.0.1:8080") is True


def test_nothing_listening_is_not_healthy(monkeypatch):
    responder(
        monkeypatch,
        health=ConnectionError("refused"),
        models=ConnectionError("refused"),
    )
    assert is_server_healthy("http://127.0.0.1:8080") is False


def test_both_endpoints_answering_badly_is_not_healthy(monkeypatch):
    responder(monkeypatch, health=500, models=503)
    assert is_server_healthy("http://127.0.0.1:8080") is False


def test_a_trailing_slash_does_not_produce_a_doubled_path(monkeypatch):
    calls = responder(monkeypatch, health=200)
    is_server_healthy("http://127.0.0.1:8080/")
    assert calls[0][0] == "http://127.0.0.1:8080/health"


def test_the_probe_timeout_is_passed_through(monkeypatch):
    calls = responder(monkeypatch, health=200)
    is_server_healthy("http://127.0.0.1:8080", timeout=0.5)
    assert calls[0][1] == 0.5


# --- is_running -----------------------------------------------------------

def test_a_manager_that_has_launched_nothing_is_not_running():
    assert LlamaCppServerManager().is_running is False


def test_a_live_process_counts_as_running():
    manager = LlamaCppServerManager()
    manager._process = FakeProcess(exit_code=None)
    assert manager.is_running is True


def test_a_process_that_has_exited_is_not_running():
    manager = LlamaCppServerManager()
    manager._process = FakeProcess(exit_code=0)
    assert manager.is_running is False


# --- what start() and start_hf() ask for ----------------------------------

def test_local_weights_are_launched_with_model_and_projector(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        LlamaCppServerManager,
        "_launch",
        lambda self, args, url, **kw: captured.update(args=args, url=url, **kw),
    )
    files = type(
        "Files",
        (),
        {"gguf_path": tmp_path / "model.gguf", "mmproj_path": tmp_path / "mmproj.gguf"},
    )()

    LlamaCppServerManager().start(files, "http://127.0.0.1:8080")

    assert captured["args"] == [
        "-m",
        str(tmp_path / "model.gguf"),
        "--mmproj",
        str(tmp_path / "mmproj.gguf"),
    ]
    assert captured["wait_seconds"] == 120.0


def test_a_hugging_face_ref_is_launched_with_hf(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        LlamaCppServerManager,
        "_launch",
        lambda self, args, url, **kw: captured.update(args=args, url=url, **kw),
    )

    LlamaCppServerManager().start_hf("ggml-org/Qwen2.5-VL:Q4_K_M", "http://127.0.0.1:8080")

    assert captured["args"] == ["-hf", "ggml-org/Qwen2.5-VL:Q4_K_M"]


def test_a_hugging_face_launch_waits_longer_for_a_first_time_pull(monkeypatch):
    """The weights may still need downloading, unlike local files."""
    captured = {}
    monkeypatch.setattr(
        LlamaCppServerManager,
        "_launch",
        lambda self, args, url, **kw: captured.update(**kw),
    )

    LlamaCppServerManager().start_hf("repo:quant", "http://127.0.0.1:8080")

    assert captured["wait_seconds"] == 300.0


# --- launching ------------------------------------------------------------

def launched(monkeypatch, *, healthy_before=False, becomes_healthy=True, binary="/bin/llama-server"):
    """Patch out the world around _launch and record the spawned command."""
    spawned = []
    states = iter([healthy_before] + [becomes_healthy] * 50)
    monkeypatch.setattr(ls, "is_server_healthy", lambda *_a, **_k: next(states))
    monkeypatch.setattr(ls, "find_llama_server_binary", lambda: binary)
    monkeypatch.setattr(ls.time, "sleep", lambda _s: None)

    def fake_popen(command, **_kwargs):
        spawned.append(command)
        return FakeProcess(exit_code=None)

    monkeypatch.setattr(ls.subprocess, "Popen", fake_popen)
    return spawned


def test_an_already_serving_endpoint_is_reused_rather_than_duplicated(monkeypatch):
    spawned = launched(monkeypatch, healthy_before=True)
    manager = LlamaCppServerManager()

    manager._launch(["-hf", "r:q"], "http://127.0.0.1:8080", context_size=4096, wait_seconds=1)

    assert spawned == []
    assert manager.is_running is False


def test_a_missing_binary_is_reported_before_anything_is_spawned(monkeypatch):
    spawned = launched(monkeypatch, binary=None)

    with pytest.raises(RuntimeError, match="PATH"):
        LlamaCppServerManager()._launch(
            ["-hf", "r:q"], "http://127.0.0.1:8080", context_size=4096, wait_seconds=1
        )

    assert spawned == []


def test_the_command_carries_the_host_port_and_context_from_the_caller(monkeypatch):
    spawned = launched(monkeypatch)

    LlamaCppServerManager()._launch(
        ["-hf", "r:q"], "http://192.168.1.5:9001", context_size=8192, wait_seconds=1
    )

    assert spawned == [
        [
            "/bin/llama-server",
            "-hf",
            "r:q",
            "--host",
            "192.168.1.5",
            "--port",
            "9001",
            "-c",
            "8192",
        ]
    ]


def test_a_url_without_a_port_falls_back_to_the_llama_cpp_default(monkeypatch):
    spawned = launched(monkeypatch)

    LlamaCppServerManager()._launch(
        ["-hf", "r:q"], "http://localhost", context_size=4096, wait_seconds=1
    )

    command = spawned[0]
    assert command[command.index("--host") + 1] == "localhost"
    assert command[command.index("--port") + 1] == "8080"


def test_a_server_that_never_comes_up_is_stopped_and_reported(monkeypatch):
    launched(monkeypatch, becomes_healthy=False)
    monkeypatch.setattr(ls.time, "time", iter([0.0, 0.0, 99.0]).__next__)
    manager = LlamaCppServerManager()

    with pytest.raises(RuntimeError, match="healthy"):
        manager._launch(
            ["-hf", "r:q"], "http://127.0.0.1:8080", context_size=4096, wait_seconds=1
        )

    assert manager._process is None, "a stuck server must not be left running"


# --- waiting for health ---------------------------------------------------

def test_waiting_ends_as_soon_as_the_server_answers(monkeypatch):
    monkeypatch.setattr(ls, "is_server_healthy", lambda *_a, **_k: True)
    monkeypatch.setattr(ls.time, "sleep", lambda _s: None)
    manager = LlamaCppServerManager()
    manager._process = FakeProcess(exit_code=None)

    assert manager._wait_until_healthy("http://127.0.0.1:8080", 5.0) is True


def test_a_process_that_dies_early_ends_the_wait(monkeypatch):
    """No point waiting out the deadline for a server that already exited."""
    monkeypatch.setattr(ls, "is_server_healthy", lambda *_a, **_k: False)
    monkeypatch.setattr(ls.time, "sleep", lambda _s: None)
    manager = LlamaCppServerManager()
    manager._process = FakeProcess(exit_code=1)

    assert manager._wait_until_healthy("http://127.0.0.1:8080", 5.0) is False


def test_the_wait_gives_up_at_the_deadline(monkeypatch):
    monkeypatch.setattr(ls, "is_server_healthy", lambda *_a, **_k: False)
    monkeypatch.setattr(ls.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ls.time, "time", iter([0.0, 1.0, 99.0]).__next__)
    manager = LlamaCppServerManager()
    manager._process = FakeProcess(exit_code=None)

    assert manager._wait_until_healthy("http://127.0.0.1:8080", 10.0) is False


# --- stopping -------------------------------------------------------------

def test_stopping_a_manager_that_never_started_is_harmless():
    LlamaCppServerManager().stop()


def test_a_running_server_is_terminated_and_forgotten():
    manager = LlamaCppServerManager()
    process = FakeProcess(exit_code=None)
    manager._process = process

    manager.stop()

    assert process.terminated is True
    assert process.killed is False
    assert manager._process is None


def test_a_server_ignoring_terminate_is_killed():
    manager = LlamaCppServerManager()
    process = FakeProcess(exit_code=None, wait_times_out=True)
    manager._process = process

    manager.stop()

    assert process.terminated is True
    assert process.killed is True
    assert manager._process is None


def test_an_already_exited_server_is_only_forgotten():
    manager = LlamaCppServerManager()
    process = FakeProcess(exit_code=0)
    manager._process = process

    manager.stop()

    assert process.terminated is False
    assert manager._process is None


def test_stopping_twice_is_harmless():
    manager = LlamaCppServerManager()
    manager._process = FakeProcess(exit_code=None)
    manager.stop()
    manager.stop()


# --- the exit safety net --------------------------------------------------

def test_a_new_manager_registers_its_own_shutdown(monkeypatch):
    """An app-launched server must never outlive the app."""
    registered = []
    monkeypatch.setattr(ls.atexit, "register", registered.append)

    manager = LlamaCppServerManager()

    assert registered == [manager.stop]
