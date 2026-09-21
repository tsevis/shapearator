"""What counts as a sheet, and how to find them in a folder.

One definition, imported by the validation rules and by the folder runner.
They ask the same question -- "is there anything here to extract?" -- and an
answer that differed between them would let a folder pass validation and then
produce nothing.
"""
from __future__ import annotations

from pathlib import Path

#: The artwork formats the extractor reads.
SUPPORTED_SUFFIXES = (".png", ".svg")


def is_sheet(path: Path) -> bool:
    """True for a file the extractor can read. A directory is never one."""
    return path.is_file() and path.suffix.lower() in {s.lower() for s in SUPPORTED_SUFFIXES}


def find_sheets(folder: Path) -> list[Path]:
    """Every sheet directly inside ``folder``, in name order.

    Deliberately not recursive. An export folder sitting inside the input
    folder is full of files with supported suffixes, and descending into one
    would feed a previous run's icons back through the extractor.
    """
    return sorted((entry for entry in folder.iterdir() if is_sheet(entry)),
                  key=lambda path: path.name)
