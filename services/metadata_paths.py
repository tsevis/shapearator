"""Keep the local filesystem out of exported metadata.

Icons and their metadata are deliverables: they get zipped, committed, and
handed to other people. An absolute path embedded in one discloses the account
name and directory layout of the machine that produced it, so everything bound
for an exported file is reduced to a portable, non-identifying form first.
"""
from __future__ import annotations

from pathlib import Path

HOME_PLACEHOLDER = "~"


def portable_output_path(path: Path, output_dir: Path) -> str:
    """Describe ``path`` relative to the export folder, using forward slashes.

    Relative paths are both safe to publish and more useful than absolute
    ones: they keep resolving after the folder is moved or shared.
    """
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        # Outside the export folder entirely; the name alone is still safe.
        return path.name


def scrub_local_paths(text: str | None, home: Path | None = None) -> str | None:
    """Replace the user's home directory with ``~`` in free-form text.

    Error messages from the filesystem and from model backends quote the paths
    they failed on, and those messages are recorded in metadata.
    """
    if not text:
        return text
    home_prefix = str(home or Path.home())
    return text.replace(home_prefix, HOME_PLACEHOLDER)
