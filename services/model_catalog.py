"""Single source of truth for the local vision models Shapearator recommends.

Every recommended icon-captioning model is described once here, with the details
each backend needs:

* Ollama  -> a model tag that ``ollama pull`` understands.
* llama.cpp -> a Hugging Face GGUF repo plus a quantization hint; the actual
  weight and ``mmproj`` (vision projector) filenames are resolved at download
  time so the catalog stays robust to upstream renames.

The model registry, the first-run installer, and the recommendation UI all read
from this table instead of hard-coding model names in several places.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionModelSpec:
    key: str
    display_name: str
    priority: int
    recommendation: str
    # Ollama
    ollama_tag: str
    approx_ollama_gb: float
    # llama.cpp (Hugging Face GGUF repo)
    hf_repo: str
    gguf_quant: str  # substring used to pick the main weights file, e.g. "Q4_K_M"
    approx_llamacpp_gb: float
    # Whether this model is downloaded by default during first-run setup.
    default_install: bool = False

    @property
    def family_aliases(self) -> tuple[str, ...]:
        """Lowercase substrings that identify this family in a raw model name.

        Used to map an arbitrary Ollama tag or GGUF filename back to this spec.
        """
        base = self.key.replace("-", "")
        return tuple({self.key, base, self.ollama_tag.split(":")[0]})


# Ordered best-first. Priority mirrors list order but is stored explicitly so
# callers can sort a mixed set of discovered models deterministically.
CATALOG: tuple[VisionModelSpec, ...] = (
    VisionModelSpec(
        key="qwen2.5-vl",
        display_name="Qwen2.5-VL 3B",
        priority=1,
        recommendation="Best overall for this app: strongest balance of icon understanding, stable short labels, and local speed.",
        ollama_tag="qwen2.5vl:3b",
        approx_ollama_gb=3.2,
        hf_repo="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
        gguf_quant="Q4_K_M",
        approx_llamacpp_gb=3.0,
        default_install=True,
    ),
    VisionModelSpec(
        key="minicpm-v",
        display_name="MiniCPM-V 2.6",
        priority=2,
        recommendation="Great backup when you want a second opinion on hand-drawn marks and ambiguous symbols.",
        ollama_tag="minicpm-v:latest",
        approx_ollama_gb=5.5,
        hf_repo="openbmb/MiniCPM-V-2_6-gguf",
        gguf_quant="Q4_K_M",
        approx_llamacpp_gb=5.5,
    ),
    VisionModelSpec(
        key="moondream",
        display_name="moondream2",
        priority=3,
        recommendation="Fastest lightweight option for quick local naming passes.",
        ollama_tag="moondream:latest",
        approx_ollama_gb=1.7,
        hf_repo="ggml-org/moondream2-20250414-GGUF",
        gguf_quant="text-model-f16",
        approx_llamacpp_gb=1.9,
    ),
    VisionModelSpec(
        key="llava",
        display_name="LLaVA 1.6 (7B)",
        priority=4,
        recommendation="General fallback vision model when the preferred local icon models are unavailable.",
        ollama_tag="llava:7b",
        approx_ollama_gb=4.7,
        hf_repo="ggml-org/llava-1.6-mistral-7b-gguf",
        gguf_quant="Q4_K_M",
        approx_llamacpp_gb=4.4,
    ),
    VisionModelSpec(
        key="smolvlm-500m",
        display_name="SmolVLM 500M (ultra-light)",
        priority=8,
        recommendation="Tiny footprint for constrained machines or a fast smoke test; lower accuracy on ambiguous marks.",
        ollama_tag="",  # not published as a first-class Ollama vision tag
        approx_ollama_gb=0.0,
        hf_repo="ggml-org/SmolVLM-500M-Instruct-GGUF",
        gguf_quant="Q8_0",
        approx_llamacpp_gb=0.6,
    ),
)

_BY_KEY: dict[str, VisionModelSpec] = {spec.key: spec for spec in CATALOG}

UNKNOWN_PRIORITY = 500
UNKNOWN_RECOMMENDATION = "Available locally, but not one of the app's primary icon-focused recommendations."


def all_specs() -> tuple[VisionModelSpec, ...]:
    return CATALOG


def spec_by_key(key: str) -> VisionModelSpec | None:
    return _BY_KEY.get(key)


def default_install_specs() -> list[VisionModelSpec]:
    return [spec for spec in CATALOG if spec.default_install]


def normalize_name(name: str) -> str:
    """Collapse a model name to lowercase alphanumerics for tolerant matching.

    ``qwen2.5vl:3b``, ``Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf`` and ``qwen2.5-vl``
    all normalize to a string containing ``qwen25vl``.
    """
    return "".join(char for char in name.lower() if char.isalnum())


def spec_for_model_name(name: str) -> VisionModelSpec | None:
    """Best-effort mapping from a raw backend model name to a catalog spec."""
    normalized = normalize_name(name)
    for spec in CATALOG:
        for alias in spec.family_aliases:
            if normalize_name(alias) and normalize_name(alias) in normalized:
                return spec
    return None


def classify_model_name(name: str) -> tuple[int, str, bool]:
    """Return (priority, recommendation, supports_vision) for a raw model name."""
    spec = spec_for_model_name(name)
    if spec is not None:
        return spec.priority, spec.recommendation, True
    return UNKNOWN_PRIORITY, UNKNOWN_RECOMMENDATION, False
