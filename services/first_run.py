"""First-run detection and model-install orchestration, shared by GUI and CLI.

This layer is UI-agnostic: it decides *what* to offer and performs installs with
progress callbacks. Presentation (a dialog or a terminal progress bar) lives in
the caller.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from . import model_bootstrap as mb
from .config_store import AppSettings
from .model_catalog import CATALOG, VisionModelSpec
from .llamacpp_server import find_llama_server_binary
from .model_registry import ModelRegistry
from .paths import setup_state_path


@dataclass(frozen=True)
class BackendStatus:
    ollama_reachable: bool
    llamacpp_binary: bool


@dataclass(frozen=True)
class SetupCandidate:
    spec: VisionModelSpec
    backend: str  # "ollama" | "llamacpp"
    installed: bool
    approx_gb: float
    default_selected: bool

    @property
    def label(self) -> str:
        where = "Ollama" if self.backend == "ollama" else "llama.cpp"
        size = f"{self.approx_gb:.1f} GB" if self.approx_gb else "small"
        state = "installed" if self.installed else f"~{size} download"
        return f"{self.spec.display_name} · {where} · {state}"


def detect_backends(settings: AppSettings) -> BackendStatus:
    return BackendStatus(
        ollama_reachable=mb.ollama_reachable(settings.ollama_url),
        llamacpp_binary=find_llama_server_binary() is not None,
    )


def build_candidates(settings: AppSettings, status: BackendStatus | None = None) -> list[SetupCandidate]:
    """Every install option for the currently available backends.

    The default model is pre-selected on the preferred backend (Ollama if it is
    running, otherwise llama.cpp), so the common path is one click.
    """
    status = status or detect_backends(settings)
    prefer_ollama = status.ollama_reachable
    candidates: list[SetupCandidate] = []
    for spec in CATALOG:
        if status.ollama_reachable and spec.ollama_tag:
            candidates.append(
                SetupCandidate(
                    spec=spec,
                    backend="ollama",
                    installed=mb.is_ollama_model_present(spec.ollama_tag, settings.ollama_url),
                    approx_gb=spec.approx_ollama_gb,
                    default_selected=spec.default_install and prefer_ollama,
                )
            )
        if status.llamacpp_binary:
            candidates.append(
                SetupCandidate(
                    spec=spec,
                    backend="llamacpp",
                    installed=mb.is_llamacpp_model_present(settings.models_root, spec),
                    approx_gb=spec.approx_llamacpp_gb,
                    default_selected=spec.default_install and not prefer_ollama,
                )
            )
    return candidates


def install_candidate(
    settings: AppSettings,
    candidate: SetupCandidate,
    progress_callback: mb.ProgressCallback | None = None,
) -> None:
    if candidate.backend == "ollama":
        mb.pull_ollama_model(candidate.spec.ollama_tag, settings.ollama_url, progress_callback)
    else:
        mb.download_llamacpp_model(candidate.spec, settings.models_root, progress_callback)


def install_selection(
    settings: AppSettings,
    candidates: list[SetupCandidate],
    progress_callback: mb.ProgressCallback | None = None,
) -> None:
    for candidate in candidates:
        if candidate.installed:
            continue
        install_candidate(settings, candidate, progress_callback)


def apply_active_model(settings: AppSettings, candidate: SetupCandidate) -> AppSettings:
    """Point the app at a freshly installed model and enable naming."""
    settings.semantic_naming = True
    if candidate.backend == "ollama":
        settings.provider = "ollama"
        settings.ollama_model = candidate.spec.ollama_tag
    else:
        settings.provider = "llamacpp"
        settings.llamacpp_model = candidate.spec.display_name
    return settings


# --- setup-state marker ---------------------------------------------------

def is_setup_marked() -> bool:
    return setup_state_path().exists()


def mark_setup_complete(details: dict | None = None) -> None:
    path = setup_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"completed_at": datetime.now(timezone.utc).isoformat()}
    if details:
        payload.update(details)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def any_vision_model_available(settings: AppSettings) -> bool:
    """True if a usable vision model already exists on any backend."""
    registry = ModelRegistry()
    if any(model.supports_vision for model in registry.list_ollama_models()):
        return True
    return any(mb.is_llamacpp_model_present(settings.models_root, spec) for spec in CATALOG)


def needs_first_run(settings: AppSettings) -> bool:
    """Offer setup only if it has never run and no vision model is present yet."""
    if is_setup_marked():
        return False
    return not any_vision_model_available(settings)
