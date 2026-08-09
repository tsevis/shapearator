"""Data carried between the extractor and its callers (GUI, CLI, tests)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
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


@dataclass
class ExtractionResult:
    input_path: Path
    output_dir: Path
    icons: list[ExtractedIcon]
    provider_summary: str


@dataclass
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
