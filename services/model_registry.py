from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests

from .model_catalog import classify_model_name


@dataclass(frozen=True)
class ModelDescriptor:
    name: str
    source: str
    location: str
    priority: int = 999
    recommendation: str = ""
    supports_vision: bool = False


def classify_vision_model(name: str) -> tuple[int, str, bool]:
    """Return (priority, recommendation, supports_vision) for a raw model name.

    Thin wrapper over the shared model catalog so registry, installer, and UI
    all agree on which models are recommended.
    """
    return classify_model_name(name)


class ModelRegistry:
    def list_ollama_models(self) -> list[ModelDescriptor]:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return []

        models: list[ModelDescriptor] = []
        for line in result.stdout.splitlines()[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            name = stripped.split()[0]
            if name.endswith(":cloud"):
                continue
            priority, recommendation, supports_vision = classify_vision_model(name)
            models.append(
                ModelDescriptor(
                    name=name,
                    source="ollama",
                    location="local ollama",
                    priority=priority,
                    recommendation=recommendation,
                    supports_vision=supports_vision,
                )
            )
        return sorted(models, key=lambda item: (item.priority, item.name.lower()))

    def recommended_ollama_model(self, models: list[ModelDescriptor]) -> ModelDescriptor | None:
        return self._best_model(models)

    def list_llamacpp_models(self, base_url: str) -> list[ModelDescriptor]:
        """Discover models served by a running llama.cpp server via ``/v1/models``.

        llama-server typically hosts a single model, so this usually returns one
        descriptor. Returns an empty list if the server is unreachable.
        """
        endpoint = base_url.rstrip("/")
        try:
            response = requests.get(f"{endpoint}/v1/models", timeout=5)
            response.raise_for_status()
            data = response.json()
        except Exception:
            return []

        models: list[ModelDescriptor] = []
        for item in data.get("data", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("id") or item.get("model") or "").strip()
            if not name:
                continue
            priority, recommendation, supports_vision = classify_vision_model(name)
            models.append(
                ModelDescriptor(
                    name=name,
                    source="llamacpp",
                    location="local llama.cpp",
                    priority=priority,
                    # A model served by llama-server is loaded and ready; treat it
                    # as usable even if it isn't in our curated recommendation set.
                    recommendation=recommendation,
                    supports_vision=True,
                )
            )
        return sorted(models, key=lambda item: (item.priority, item.name.lower()))

    def recommended_llamacpp_model(self, models: list[ModelDescriptor]) -> ModelDescriptor | None:
        return self._best_model(models)

    @staticmethod
    def _best_model(models: list[ModelDescriptor]) -> ModelDescriptor | None:
        if not models:
            return None
        vision_models = [model for model in models if model.supports_vision]
        return min(vision_models or models, key=lambda item: (item.priority, item.name.lower()))

    def list_directory_models(self, root_dir: str) -> list[ModelDescriptor]:
        if not root_dir or not root_dir.strip():
            return []
        root = Path(root_dir).expanduser()
        if not root.exists():
            return []
        models: list[ModelDescriptor] = []
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if child.name.startswith("."):
                continue
            if child.is_dir() or child.is_file():
                models.append(
                    ModelDescriptor(
                        name=child.stem if child.is_file() else child.name,
                        source="directory",
                        location=str(child),
                    )
                )
        return models
