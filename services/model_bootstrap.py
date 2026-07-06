"""Download local vision models on demand for either backend.

* Ollama    -> streams ``POST /api/pull`` and reports byte progress.
* llama.cpp -> resolves the GGUF weights + ``mmproj`` projector in a Hugging Face
  repo and stream-downloads both (with resume) into the app's models directory,
  so the app can point ``llama-server`` at self-contained local files.

Everything here degrades gracefully: a missing backend, an offline machine, or a
partial download is reported, never crashed on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import requests

from .model_catalog import VisionModelSpec
from .paths import resolve_models_root

ProgressCallback = Callable[["BootstrapProgress"], None]

CHUNK_SIZE = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class BootstrapProgress:
    backend: str  # "ollama" | "llamacpp"
    model_key: str
    phase: str  # resolve | download | verify | done | error | skip
    completed: int
    total: int
    message: str

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.completed / self.total))


@dataclass(frozen=True)
class LlamaCppModelFiles:
    gguf_path: Path
    mmproj_path: Path


class BootstrapError(RuntimeError):
    pass


def _emit(callback: ProgressCallback | None, progress: BootstrapProgress) -> None:
    if callback is not None:
        callback(progress)


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------

def ollama_reachable(base_url: str, timeout: float = 3.0) -> bool:
    """True when an Ollama daemon answers at ``base_url``."""
    try:
        return requests.get(f"{base_url.rstrip('/')}/api/version", timeout=timeout).status_code == 200
    except Exception:
        return False


def ollama_installed_tags(base_url: str, timeout: float = 5.0) -> set[str]:
    """Return the set of model tags Ollama already has locally (empty if down)."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return set()
    tags: set[str] = set()
    for item in data.get("models", []):
        if isinstance(item, dict) and item.get("name"):
            tags.add(str(item["name"]))
    return tags


def is_ollama_model_present(tag: str, base_url: str) -> bool:
    if not tag:
        return False
    installed = ollama_installed_tags(base_url)
    if tag in installed:
        return True
    # Tolerate the implicit ":latest" suffix Ollama applies.
    bare = tag.split(":")[0]
    return any(name == tag or name.split(":")[0] == bare for name in installed)


def pull_ollama_model(
    tag: str,
    base_url: str,
    progress_callback: ProgressCallback | None = None,
    timeout: float = 600.0,
) -> None:
    """Pull an Ollama model, streaming byte-level progress."""
    if not tag:
        raise BootstrapError("No Ollama tag configured for this model.")
    _emit(progress_callback, BootstrapProgress("ollama", tag, "resolve", 0, 0, f"Pulling {tag}"))
    try:
        with requests.post(
            f"{base_url.rstrip('/')}/api/pull",
            json={"model": tag, "stream": True},
            stream=True,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            _consume_ollama_pull_stream(response.iter_lines(), tag, progress_callback)
    except BootstrapError:
        raise
    except Exception as exc:  # network / server errors
        _emit(progress_callback, BootstrapProgress("ollama", tag, "error", 0, 0, str(exc)))
        raise BootstrapError(f"Ollama pull failed for {tag}: {exc}") from exc
    _emit(progress_callback, BootstrapProgress("ollama", tag, "done", 1, 1, f"{tag} ready"))


def _consume_ollama_pull_stream(
    lines: Iterable[bytes],
    tag: str,
    progress_callback: ProgressCallback | None,
) -> None:
    for raw in lines:
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except Exception:
            continue
        if event.get("error"):
            raise BootstrapError(str(event["error"]))
        status = str(event.get("status", ""))
        completed = int(event.get("completed", 0) or 0)
        total = int(event.get("total", 0) or 0)
        _emit(
            progress_callback,
            BootstrapProgress("ollama", tag, "download", completed, total, status or f"Pulling {tag}"),
        )


# --------------------------------------------------------------------------
# llama.cpp (Hugging Face GGUF)
# --------------------------------------------------------------------------

def resolve_repo_files(spec: VisionModelSpec) -> tuple[str, str]:
    """Resolve (main_gguf, mmproj_gguf) filenames within the model's HF repo.

    Uses the repo listing so we tolerate upstream filename changes instead of
    hard-coding brittle names.
    """
    from huggingface_hub import list_repo_files

    try:
        files = [f for f in list_repo_files(spec.hf_repo) if f.lower().endswith(".gguf")]
    except Exception as exc:
        raise BootstrapError(f"Could not list files for {spec.hf_repo}: {exc}") from exc

    mmproj = next((f for f in files if "mmproj" in f.lower()), None)
    weights = [f for f in files if "mmproj" not in f.lower()]
    quant = spec.gguf_quant.lower()
    main = next((f for f in weights if quant in f.lower()), None)
    if main is None and weights:
        # Fall back to the smallest weights file by name heuristic (Q4 < Q8 < f16).
        main = sorted(weights, key=lambda f: (len(f), f.lower()))[0]
    if main is None or mmproj is None:
        raise BootstrapError(
            f"{spec.hf_repo} is missing a GGUF weights or mmproj file (found: {files or 'none'})."
        )
    return main, mmproj


def llamacpp_target_dir(models_root: str | None, spec: VisionModelSpec) -> Path:
    return resolve_models_root(models_root) / "llamacpp" / spec.key


def find_local_llamacpp_files(models_root: str | None, spec: VisionModelSpec) -> LlamaCppModelFiles | None:
    """Return already-downloaded (gguf, mmproj) for this model, if both exist."""
    target = llamacpp_target_dir(models_root, spec)
    if not target.exists():
        return None
    ggufs = list(target.glob("*.gguf"))
    mmproj = next((f for f in ggufs if "mmproj" in f.name.lower()), None)
    weights = next((f for f in ggufs if "mmproj" not in f.name.lower()), None)
    if weights is not None and mmproj is not None:
        return LlamaCppModelFiles(gguf_path=weights, mmproj_path=mmproj)
    return None


def is_llamacpp_model_present(models_root: str | None, spec: VisionModelSpec) -> bool:
    return find_local_llamacpp_files(models_root, spec) is not None


def download_llamacpp_model(
    spec: VisionModelSpec,
    models_root: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> LlamaCppModelFiles:
    """Download the GGUF weights + mmproj for ``spec`` into the models directory."""
    existing = find_local_llamacpp_files(models_root, spec)
    if existing is not None:
        _emit(progress_callback, BootstrapProgress("llamacpp", spec.key, "done", 1, 1, f"{spec.display_name} already installed"))
        return existing

    from huggingface_hub import hf_hub_url

    _emit(progress_callback, BootstrapProgress("llamacpp", spec.key, "resolve", 0, 0, f"Resolving {spec.hf_repo}"))
    main_file, mmproj_file = resolve_repo_files(spec)
    target = llamacpp_target_dir(models_root, spec)

    gguf_path = _download_file(
        hf_hub_url(repo_id=spec.hf_repo, filename=main_file),
        target / Path(main_file).name,
        spec.key,
        progress_callback,
    )
    mmproj_path = _download_file(
        hf_hub_url(repo_id=spec.hf_repo, filename=mmproj_file),
        target / Path(mmproj_file).name,
        spec.key,
        progress_callback,
    )
    _emit(progress_callback, BootstrapProgress("llamacpp", spec.key, "done", 1, 1, f"{spec.display_name} ready"))
    return LlamaCppModelFiles(gguf_path=gguf_path, mmproj_path=mmproj_path)


def _download_file(
    url: str,
    dest: Path,
    model_key: str,
    progress_callback: ProgressCallback | None,
    timeout: float = 60.0,
) -> Path:
    """Stream a file to ``dest`` with resume support and byte progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    part = dest.with_name(dest.name + ".part")
    resume = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume}-"} if resume else {}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as response:
            if response.status_code == 416:  # requested range already satisfied
                part.rename(dest)
                return dest
            # If the server ignored our Range header, start over cleanly.
            if resume and response.status_code == 200:
                resume = 0
            response.raise_for_status()
            remaining = int(response.headers.get("Content-Length", 0) or 0)
            total = resume + remaining
            done = resume
            mode = "ab" if resume else "wb"
            with open(part, mode) as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    done += len(chunk)
                    _emit(
                        progress_callback,
                        BootstrapProgress("llamacpp", model_key, "download", done, total, f"Downloading {dest.name}"),
                    )
    except Exception as exc:
        _emit(progress_callback, BootstrapProgress("llamacpp", model_key, "error", 0, 0, str(exc)))
        raise BootstrapError(f"Download failed for {dest.name}: {exc}") from exc
    part.rename(dest)
    return dest
