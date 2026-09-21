"""Extracting every sheet in a folder in one run.

`IconExtractor` takes one sheet and knows nothing about its neighbours. A
folder adds three questions it cannot answer on its own:

* which files in there are sheets, and which are notes, previews, or the
  output of an earlier run;
* where each sheet's icons go, given that every run numbers its icons from
  `icon_001` and two sheets writing one folder would overwrite each other;
* what a run does when sheet four of nine cannot be read.

The third is the reason this module exists rather than a loop at each call
site. A batch that stops at the first bad file wastes the work already done
and tells the user nothing about the sheets it never reached, so a failure is
recorded against its sheet and the run continues.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from services.extraction_types import ExtractedIcon, ExtractionProgress, ExtractionResult
from services.extractor import IconExtractor
from services.sheets import find_sheets
from services.semantic_naming import SemanticPreflightError
from services.settings_schema import AppSettings


class NoSheetsFound(RuntimeError):
    """Raised before anything is written, when a folder holds no sheet."""


@dataclass(frozen=True)
class SheetOutcome:
    """What became of one sheet in a folder run."""

    input_path: Path
    output_dir: Path
    result: ExtractionResult | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.result is None

    @property
    def icons(self) -> list[ExtractedIcon]:
        return list(self.result.icons) if self.result else []


@dataclass(frozen=True)
class BatchOutcome:
    """Every sheet's fate, plus the totals an interface reports."""

    folder: Path
    output_root: Path
    sheets: tuple[SheetOutcome, ...]

    @property
    def icons(self) -> list[ExtractedIcon]:
        return [icon for sheet in self.sheets for icon in sheet.icons]

    @property
    def failed_count(self) -> int:
        return sum(1 for sheet in self.sheets if sheet.failed)

    @property
    def extracted_count(self) -> int:
        return len(self.sheets) - self.failed_count

    @property
    def warnings(self) -> tuple[str, ...]:
        """A sheet's own warnings, and one line per sheet that did not run.

        Each is prefixed with the file it came from: in a folder run an
        unattributed warning is not actionable, because the user cannot tell
        which of nine sheets it describes.
        """
        collected: list[str] = []
        for sheet in self.sheets:
            name = sheet.input_path.name
            if sheet.error is not None:
                collected.append(f"{name}: not extracted -- {sheet.error}")
            elif sheet.result is not None:
                collected.extend(f"{name}: {warning}" for warning in sheet.result.warnings)
        return tuple(collected)

    def summary(self) -> str:
        """One line for a status bar or the end of a CLI run."""
        detail = f"{len(self.icons)} icons from {self.extracted_count} of {len(self.sheets)} sheets"
        if self.failed_count:
            detail += f"; {self.failed_count} could not be read"
        return detail


def plan_output_dirs(sheets: list[Path], output_root: Path) -> dict[Path, Path]:
    """Map each sheet to the folder its icons belong in.

    Named after the sheet, because a person looking for the icons of
    `Many3.svg` looks for `Many3`. Two sheets can share a stem -- `logo.png`
    and `logo.svg` are different artwork -- and sharing a folder would have
    the second overwrite the first's `icon_001`, so a clash takes the suffix
    too.
    """
    planned: dict[Path, Path] = {}
    taken: set[str] = set()
    for sheet in sheets:
        name = sheet.stem
        if name.casefold() in taken:
            name = f"{sheet.stem}-{sheet.suffix.lstrip('.').lower()}"
        suffix_index = 2
        while name.casefold() in taken:
            name = f"{sheet.stem}-{suffix_index}"
            suffix_index += 1
        taken.add(name.casefold())
        planned[sheet] = output_root / name
    return planned


def extract_folder(
    settings: AppSettings,
    folder: Path,
    output_root: Path,
    formats: set[str],
    progress_callback: Callable[[ExtractionProgress], None] | None = None,
    allow_unnamed: bool = False,
) -> BatchOutcome:
    """Extract every sheet in ``folder`` into its own subfolder of ``output_root``.

    Raises :class:`NoSheetsFound` before creating anything when the folder
    holds nothing to extract, so an empty run leaves no empty directories
    behind to explain.
    """
    sheets = find_sheets(folder)
    if not sheets:
        raise NoSheetsFound(
            f"No .png or .svg sheets in {folder}. "
            "Choose a folder that contains the artwork, not one above it."
        )

    planned = plan_output_dirs(sheets, output_root)
    outcomes: list[SheetOutcome] = []

    for index, sheet in enumerate(sheets, start=1):
        destination = planned[sheet]
        _emit(progress_callback, "sheet", index, len(sheets), f"Sheet {index} of {len(sheets)}: {sheet.name}")

        def forward(progress: ExtractionProgress, _name: str = sheet.name) -> None:
            # The bar counts sheets; the message carries what is happening
            # inside the current one, which is the part that takes the time.
            _emit(progress_callback, progress.phase, index, len(sheets), f"{_name} -- {progress.message}")

        try:
            result = IconExtractor(settings).extract(
                sheet, destination, formats,
                progress_callback=forward, allow_unnamed=allow_unnamed,
            )
        except SemanticPreflightError:
            # Not this sheet's problem, and skipping it fixes nothing: the
            # vision backend is unreachable for every sheet in the folder.
            # Recording it per sheet turns one recoverable failure into nine
            # unactionable ones and silences the "export with generic names?"
            # offer that both the CLI and the app build on this exception.
            _remove_if_empty(destination)
            raise
        except Exception as exc:  # one unreadable sheet must not end the run
            _remove_if_empty(destination)
            outcomes.append(SheetOutcome(
                input_path=sheet, output_dir=destination,
                error=str(exc) or exc.__class__.__name__,
            ))
            continue
        outcomes.append(SheetOutcome(input_path=sheet, output_dir=destination, result=result))

    return BatchOutcome(folder=folder, output_root=output_root, sheets=tuple(outcomes))


def _remove_if_empty(directory: Path) -> None:
    """Take back the folder a failed sheet had already created.

    An empty folder named after a sheet reads as "extracted, found nothing",
    which is a different and much quieter failure than "could not be read".
    Only an empty one goes: anything the user already had there stays.
    """
    try:
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    except OSError:
        pass  # a folder we cannot tidy is not a reason to fail the run


def _emit(
    callback: Callable[[ExtractionProgress], None] | None,
    phase: str,
    current: int,
    total: int,
    message: str,
) -> None:
    if callback is not None:
        callback(ExtractionProgress(phase=phase, current=current, total=total, message=message))
