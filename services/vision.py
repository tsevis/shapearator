from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, TypeVar
from urllib.parse import urlparse

import requests

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

DEFAULT_REQUEST_TIMEOUT = 120.0

T = TypeVar("T")

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class VisionClient(Protocol):
    """Common surface shared by the local vision providers."""

    def identify_icon(self, model: str, image_path: Path) -> dict[str, object]:
        ...

    def caption_icon(self, model: str, image_path: Path) -> str:
        ...


def is_local_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return parsed.hostname in LOCAL_HOSTS


def image_mime_type(image_path: Path) -> str:
    return _MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/png")


def build_icon_prompt(model: str) -> str:
    """Build the icon-identification prompt, tuned per known vision-model family.

    Matching is substring-based so it works for both Ollama tags (``moondream:latest``)
    and llama.cpp GGUF names (``moondream2-q4.gguf``).
    """
    name = (model or "").lower()
    base = (
        "This is a single hand-drawn UI or symbolic icon on a transparent or white background. "
        "Return valid compact JSON only with this shape: "
        "{\"label\":\"short-lowercase-filename-label\",\"tags\":[\"tag1\",\"tag2\"],\"confidence\":0.0}. "
        "Use 1 to 3 words in the label joined with hyphens. "
        "Prefer common UI nouns like lightbulb, folder, upload, heart, magnifier, grid, smiley."
    )
    if "moondream" in name:
        return (
            "Return only JSON like {\"label\":\"lightbulb\",\"tags\":[\"idea\",\"lamp\"],\"confidence\":0.82} for this hand-drawn icon. "
            "Keep tags short and common."
        )
    if "minicpm" in name:
        return base + " If uncertain, choose the most likely simple icon category."
    if "llava" in name:
        return base + " Keep it concise and generic rather than descriptive."
    return base


# Lead-ins that weaker vision models prepend to descriptions; stripped so a
# fallback label starts at the actual subject.
_DESCRIPTION_LEAD_INS = (
    "the image shows",
    "this image shows",
    "the icon shows",
    "this is an image of",
    "this is a picture of",
    "this is an icon of",
    "this is a",
    "this is an",
    "it appears to be",
    "the image is",
    "a picture of",
    "an image of",
)

# A filename label should be a couple of words, never a sentence. These caps keep
# a model that ignores the JSON instruction from producing an unusable filename.
_MAX_FALLBACK_WORDS = 4
_MAX_LABEL_CHARS = 60


def _clean_fallback_label(text: str) -> str:
    """Turn a free-form first line into a short, filesystem-safe label."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    line = line.strip().lower().strip('"').strip("'").strip()
    for lead in _DESCRIPTION_LEAD_INS:
        if line.startswith(lead):
            line = line[len(lead):].strip()
            break
    # Keep only the first clause, before any sentence/punctuation break.
    for stop in (".", ",", ";", ":", " - ", " that ", " which ", " with "):
        idx = line.find(stop)
        if idx > 0:
            line = line[:idx]
            break
    words = [w for w in line.replace("-", " ").split() if w]
    label = "-".join(words[:_MAX_FALLBACK_WORDS])
    return label[:_MAX_LABEL_CHARS].strip("-")


def parse_semantic_response(text: str) -> dict[str, object]:
    """Parse a model's free-form answer into a normalized semantic payload."""
    text = (text or "").strip().lower()
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            payload = json.loads(text[start : end + 1])
            label = str(payload.get("label", "")).strip().lower()
            # Even inside JSON, a verbose model may stuff a sentence into label.
            if len(label) > _MAX_LABEL_CHARS or label.count(" ") >= _MAX_FALLBACK_WORDS:
                label = _clean_fallback_label(label)
            else:
                label = label.replace(" ", "-")
            tags = payload.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            clean_tags = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
            confidence = payload.get("confidence")
            try:
                confidence_value = float(confidence)
            except Exception:
                confidence_value = None
            if confidence_value is not None:
                confidence_value = max(0.0, min(1.0, confidence_value))
            return {"label": label, "tags": clean_tags[:6], "confidence": confidence_value}
    except Exception:
        pass
    return {"label": _clean_fallback_label(text), "tags": [], "confidence": None}


def build_vision_client(settings) -> "VisionClient":
    """Return the local vision client for the active provider.

    Clients are imported lazily to avoid an import cycle with this module,
    which they depend on for shared prompt/parse helpers.
    """
    if settings.provider == "llamacpp":
        from .llamacpp_client import LlamaCppVisionClient

        return LlamaCppVisionClient(settings.llamacpp_url)
    from .ollama_client import OllamaVisionClient

    return OllamaVisionClient(settings.ollama_url)


def active_vision_model(settings) -> str:
    """Return the model name configured for the active provider."""
    if settings.provider == "llamacpp":
        return settings.llamacpp_model
    return settings.ollama_model


def semantic_naming_enabled(settings) -> bool:
    """True when the current provider can produce semantic names and it is enabled."""
    return bool(settings.semantic_naming) and settings.provider in {"ollama", "llamacpp"}


# Cold model loads and transient hiccups show up as connection/timeout errors;
# a couple of backed-off retries makes the first request of a session reliable.
RETRYABLE_EXCEPTIONS = (requests.ConnectionError, requests.Timeout)


def request_with_retries(
    send: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.8,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``send`` with exponential backoff on transient network errors."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return send()
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            sleep(base_delay * (2 ** attempt))
    assert last_error is not None
    raise last_error


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    provider: str
    message: str
    model: str | None = None
    vision_capable: bool | None = None


def preflight(settings, timeout: float = 5.0) -> PreflightResult:
    """Check that the active provider is ready to caption icons.

    Returns a structured result so callers can show a clear, actionable message
    instead of failing part-way through a naming run.
    """
    provider = settings.provider
    if provider not in {"ollama", "llamacpp"}:
        return PreflightResult(True, provider, "No local model needed for this provider.")
    if provider == "ollama":
        return _preflight_ollama(settings, timeout)
    return _preflight_llamacpp(settings, timeout)


def _preflight_ollama(settings, timeout: float) -> PreflightResult:
    from .model_catalog import spec_for_model_name

    url = settings.ollama_url.rstrip("/")
    if not is_local_url(url):
        return PreflightResult(False, "ollama", "Ollama URL must be a local endpoint (127.0.0.1).")
    try:
        response = requests.get(f"{url}/api/tags", timeout=timeout)
        response.raise_for_status()
        tags = {str(m.get("name", "")) for m in response.json().get("models", [])}
    except Exception:
        return PreflightResult(
            False, "ollama",
            f"Ollama is not reachable at {settings.ollama_url}. Start Ollama, then retry.",
        )
    model = settings.ollama_model
    if not model:
        return PreflightResult(False, "ollama", "No Ollama model selected in Settings.")
    bare = model.split(":")[0]
    if model not in tags and not any(t.split(":")[0] == bare for t in tags):
        return PreflightResult(
            False, "ollama", f"Model '{model}' is not pulled. Run the first-run setup or `ollama pull {model}`.",
            model=model,
        )
    vision = spec_for_model_name(model) is not None
    if not vision:
        return PreflightResult(
            True, "ollama",
            f"Using '{model}'. It is not a known vision model; labels may be poor.",
            model=model, vision_capable=False,
        )
    return PreflightResult(True, "ollama", f"Ollama ready with vision model '{model}'.", model=model, vision_capable=True)


def _preflight_llamacpp(settings, timeout: float) -> PreflightResult:
    url = settings.llamacpp_url.rstrip("/")
    if not is_local_url(url):
        return PreflightResult(False, "llamacpp", "llama.cpp URL must be a local endpoint (127.0.0.1).")
    try:
        response = requests.get(f"{url}/v1/models", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return PreflightResult(
            False, "llamacpp",
            f"No llama.cpp server at {settings.llamacpp_url}. Start llama-server (or use auto-start), then retry.",
        )
    entries = payload.get("models") or payload.get("data") or []
    if not entries:
        return PreflightResult(False, "llamacpp", "llama.cpp server is running but no model is loaded.")
    first = entries[0] if isinstance(entries[0], dict) else {}
    model = str(first.get("id") or first.get("name") or settings.llamacpp_model or "loaded model")
    capabilities = first.get("capabilities")
    if isinstance(capabilities, list):
        if "multimodal" in capabilities or "vision" in capabilities:
            return PreflightResult(True, "llamacpp", f"llama.cpp ready with vision model '{model}'.", model=model, vision_capable=True)
        return PreflightResult(
            False, "llamacpp",
            f"Loaded model '{model}' is not multimodal. Start llama-server with a vision model and its --mmproj.",
            model=model, vision_capable=False,
        )
    # Older servers omit capabilities; assume usable but note it.
    return PreflightResult(True, "llamacpp", f"llama.cpp ready with model '{model}'.", model=model)
