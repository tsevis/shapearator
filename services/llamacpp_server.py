"""Optionally launch a local ``llama-server`` for downloaded GGUF vision models.

Unlike Ollama (a always-on daemon), llama.cpp needs a server process bound to a
specific model. When the user picks the llama.cpp provider and no server is
running, the app can start one itself against locally downloaded weights so the
experience matches Ollama's "it just works".
"""
from __future__ import annotations

import atexit
import shutil
import subprocess
import time
from urllib.parse import urlparse

import requests

from .model_bootstrap import LlamaCppModelFiles


def find_llama_server_binary() -> str | None:
    return shutil.which("llama-server")


def is_server_healthy(base_url: str, timeout: float = 2.0) -> bool:
    base = base_url.rstrip("/")
    try:
        response = requests.get(f"{base}/health", timeout=timeout)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    # Some builds only expose /v1/models.
    try:
        return requests.get(f"{base}/v1/models", timeout=timeout).status_code == 200
    except Exception:
        return False


class LlamaCppServerManager:
    """Start and stop a llama-server subprocess owned by this app."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        # Safety net: never leave an app-launched server orphaned on exit.
        atexit.register(self.stop)

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(
        self,
        files: LlamaCppModelFiles,
        base_url: str,
        *,
        context_size: int = 4096,
        wait_seconds: float = 120.0,
    ) -> None:
        """Launch llama-server for locally downloaded weights + projector."""
        self._launch(
            ["-m", str(files.gguf_path), "--mmproj", str(files.mmproj_path)],
            base_url,
            context_size=context_size,
            wait_seconds=wait_seconds,
        )

    def start_hf(
        self,
        hf_ref: str,
        base_url: str,
        *,
        context_size: int = 4096,
        wait_seconds: float = 300.0,
    ) -> None:
        """Launch llama-server for a Hugging Face ref (``repo:quant``).

        Reuses llama.cpp's own cache; only downloads if the model is not present.
        The longer default wait accommodates a first-time pull.
        """
        self._launch(
            ["-hf", hf_ref],
            base_url,
            context_size=context_size,
            wait_seconds=wait_seconds,
        )

    def _launch(
        self,
        model_args: list[str],
        base_url: str,
        *,
        context_size: int,
        wait_seconds: float,
    ) -> None:
        if is_server_healthy(base_url):
            return  # something is already serving here; reuse it
        binary = find_llama_server_binary()
        if binary is None:
            raise RuntimeError("llama-server was not found on PATH. Install llama.cpp first.")
        parsed = urlparse(base_url)
        host = parsed.hostname or "127.0.0.1"
        port = str(parsed.port or 8080)
        command = [binary, *model_args, "--host", host, "--port", port, "-c", str(context_size)]
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not self._wait_until_healthy(base_url, wait_seconds):
            self.stop()
            raise RuntimeError("llama-server did not become healthy in time.")

    def _wait_until_healthy(self, base_url: str, wait_seconds: float) -> bool:
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                return False  # process exited early
            if is_server_healthy(base_url):
                return True
            time.sleep(1.0)
        return False

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
