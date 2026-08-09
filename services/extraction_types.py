"""Data carried between the extractor and its callers (GUI, CLI, tests)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Per-icon semantic naming outcome.
NAMING_NOT_REQUESTED = "not_requested"  # naming was off, or the run was downgraded
NAMING_NAMED = "named"  # the model returned a label and the files were renamed
NAMING_FAILED = "failed"  # naming was attempted for this icon and did not succeed


@dataclass(frozen=True)
class ExtractedIcon:
    index: int
    stem: str
    outputs: dict[str, Path]
    preview_path: Path | None
    canvas_size: tuple[int, int]
    source_size: tuple[int, int]
    source_bounds: tuple[int, int, int, int]
    semantic_label: str | None = None
    semantic_tags: list[str] | None = None
    semantic_confidence: float | None = None
    vector_mode: str | None = None
    metadata_path: Path | None = None
    naming_status: str = NAMING_NOT_REQUESTED
    naming_error: str | None = None


@dataclass(frozen=True)
class NamingSummary:
    """What semantic naming was asked to do, and what it actually achieved."""

    requested: bool = False
    provider: str | None = None
    model: str | None = None
    named: int = 0
    failed: int = 0
    errors: tuple[str, ...] = ()

    @property
    def attempted(self) -> int:
        return self.named + self.failed

    def describe(self) -> str:
        if not self.requested:
            return "Semantic naming: not requested."
        if self.attempted == 0:
            return f"Semantic naming: requested ({self.model or 'no model'}) but no icon was named."
        detail = f"Semantic naming: {self.named} named"
        if self.failed:
            detail += f", {self.failed} failed"
        return f"{detail} via {self.provider}/{self.model}."


@dataclass(frozen=True)
class ExtractionResult:
    input_path: Path
    output_dir: Path
    icons: list[ExtractedIcon]
    provider_summary: str
    naming: NamingSummary = field(default_factory=NamingSummary)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractionProgress:
    phase: str
    current: int
    total: int
    message: str

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.current / self.total))
