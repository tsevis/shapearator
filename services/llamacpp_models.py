"""Discover llama.cpp vision models that are ready to serve locally.

Two sources are unified into a single list of *startable* models:

1. Models the app downloaded itself into the models directory (``-m`` + ``--mmproj``).
2. Models already in llama.cpp's own cache (pulled via ``-hf`` outside the app),
   discovered with ``llama-cli --cache-list`` and launched by their ``-hf`` ref.

Only recognized vision families (per the model catalog) are offered, so text-only
or embedding models in the cache are filtered out.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

from .model_bootstrap import LlamaCppModelFiles, list_downloaded_llamacpp
from .model_catalog import VisionModelSpec, spec_for_model_name

_CACHE_LINE = re.compile(r"^\s*\d+\.\s+(\S+)\s*$")


@dataclass(frozen=True)
class StartableLlamaModel:
    display_name: str
    priority: int
    source: str  # "downloaded" | "cache"
    files: LlamaCppModelFiles | None = None
    hf_ref: str | None = None


def list_cache_refs() -> list[str]:
    """Return raw ``repo:quant`` refs from ``llama-cli --cache-list`` (empty if unavailable)."""
    binary = shutil.which("llama-cli")
    if binary is None:
        return []
    try:
        result = subprocess.run(
            [binary, "--cache-list"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return []
    refs: list[str] = []
    for line in result.stdout.splitlines():
        match = _CACHE_LINE.match(line)
        if match:
            refs.append(match.group(1))
    return refs


def cached_vision_models() -> list[tuple[str, VisionModelSpec]]:
    """Cached refs that map to a known vision family, best-first."""
    matched: list[tuple[str, VisionModelSpec]] = []
    seen: set[str] = set()
    for ref in list_cache_refs():
        spec = spec_for_model_name(ref)
        if spec is not None and spec.key not in seen:
            matched.append((ref, spec))
            seen.add(spec.key)
    return sorted(matched, key=lambda item: item[1].priority)


def available_startable_models(models_root: str | None) -> list[StartableLlamaModel]:
    """All llama.cpp vision models that can be served right now, best-first.

    Locally downloaded models take precedence over cache refs for the same family.
    """
    startable: list[StartableLlamaModel] = []
    seen_keys: set[str] = set()

    for spec, files in list_downloaded_llamacpp(models_root):
        startable.append(
            StartableLlamaModel(spec.display_name, spec.priority, "downloaded", files=files)
        )
        seen_keys.add(spec.key)

    for ref, spec in cached_vision_models():
        if spec.key in seen_keys:
            continue
        startable.append(
            StartableLlamaModel(spec.display_name, spec.priority, "cache", hf_ref=ref)
        )
        seen_keys.add(spec.key)

    return sorted(startable, key=lambda model: model.priority)
