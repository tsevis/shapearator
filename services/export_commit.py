"""Staged exports committed against a run manifest.

A run writes into a staging directory inside the output folder and is moved
into place only once extraction, naming, and metadata have all succeeded. The
manifest records exactly which files Shapearator wrote, and it is the *only*
thing a later run is allowed to delete: anything else in the folder belongs to
the user and is left alone.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .metadata_paths import scrub_local_paths
from .settings_schema import FORMATS

MANIFEST_NAME = ".shapearator-manifest.json"
STAGING_NAME = ".shapearator-staging"
MANIFEST_VERSION = 1
REPLACED_NAME = "_replaced"

# Only these directories are ever written or removed by a commit. Anything the
# extractor leaves outside them (the _work_* intermediates) stays in staging
# and is discarded with it.
MANAGED_DIRECTORIES = tuple(FORMATS) + ("metadata",)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommitReport:
    """What a commit changed, for the CLI and GUI to report."""

    written: int = 0
    replaced: int = 0
    preserved: int = 0
    warnings: tuple[str, ...] = ()


def read_manifest(output_dir: Path) -> dict | None:
    """Return the previous run's manifest, or None if there isn't a usable one."""
    path = output_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable manifest at %s: %s", path, exc)
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        logger.warning("Ignoring malformed manifest at %s", path)
        return None
    return manifest


def managed_files(manifest: dict | None, output_dir: Path) -> list[Path]:
    """Resolve a manifest's file list to paths that are safe to delete.

    Entries are dropped unless they are relative and stay inside the output
    directory, so a hand-edited or corrupted manifest cannot direct the app to
    delete something elsewhere on disk.
    """
    if manifest is None:
        return []
    resolved_root = output_dir.resolve()
    safe: list[Path] = []
    for entry in manifest["files"]:
        if not isinstance(entry, str) or not entry:
            continue
        candidate = Path(entry)
        if candidate.is_absolute():
            logger.warning("Manifest entry %r is absolute; ignoring.", entry)
            continue
        target = (output_dir / candidate).resolve()
        if not target.is_relative_to(resolved_root):
            logger.warning("Manifest entry %r escapes the output directory; ignoring.", entry)
            continue
        safe.append(target)
    return safe


class ExportStaging:
    """A scratch copy of the output tree, promoted atomically on success.

    Use as a context manager: leaving the block removes the staging directory
    whether or not the commit ran, so a failed run never leaves debris behind.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.staging_dir = output_dir / STAGING_NAME

    def __enter__(self) -> "ExportStaging":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # A leftover directory means a previous process died mid-run.
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        self.staging_dir.mkdir(parents=True)
        return self

    def __exit__(self, *_exc_info) -> None:
        shutil.rmtree(self.staging_dir, ignore_errors=True)

    def final_path(self, staged_path: Path) -> Path:
        """Translate a staged path to where it will live after the commit."""
        return self.output_dir / staged_path.relative_to(self.staging_dir)

    def staged_files(self) -> list[Path]:
        """Every file that will be committed, in a stable order."""
        found: list[Path] = []
        for name in MANAGED_DIRECTORIES:
            directory = self.staging_dir / name
            if directory.is_dir():
                found.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
        return found

    def commit(self, input_path: Path, formats: set[str], icon_count: int, app_version: str) -> CommitReport:
        """Replace the previously-managed files with this run's output.

        Prior files are moved aside rather than deleted outright, so a failure
        partway through moving the new ones can put everything back.
        """
        prior = read_manifest(self.output_dir)
        to_replace = [path for path in managed_files(prior, self.output_dir) if path.is_file()]
        staged = self.staged_files()
        warnings = self._describe_unmanaged(prior, staged)

        trash_root = self.staging_dir / REPLACED_NAME
        moved_aside: list[tuple[Path, Path]] = []
        moved_in: list[tuple[Path, Path]] = []
        try:
            for original in to_replace:
                parked = trash_root / original.relative_to(self.output_dir)
                parked.parent.mkdir(parents=True, exist_ok=True)
                original.replace(parked)
                moved_aside.append((original, parked))

            for staged_path in staged:
                destination = self.final_path(staged_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                staged_path.replace(destination)
                moved_in.append((staged_path, destination))
        except OSError:
            self._roll_back(moved_in, moved_aside)
            raise

        self._prune_empty_managed_dirs()
        self._write_manifest(input_path, formats, icon_count, app_version, staged)
        return CommitReport(
            written=len(staged),
            replaced=len(to_replace),
            preserved=self._count_preserved(),
            warnings=warnings,
        )

    # --- internals --------------------------------------------------------

    def _roll_back(self, moved_in: list[tuple[Path, Path]], moved_aside: list[tuple[Path, Path]]) -> None:
        """Undo a partial commit, newest move first."""
        for staged_path, destination in reversed(moved_in):
            try:
                destination.replace(staged_path)
            except OSError:
                logger.exception("Could not return %s to staging during rollback", destination)
        for original, parked in reversed(moved_aside):
            try:
                original.parent.mkdir(parents=True, exist_ok=True)
                parked.replace(original)
            except OSError:
                logger.exception("Could not restore %s during rollback", original)

    def _describe_unmanaged(self, prior: dict | None, staged: list[Path]) -> tuple[str, ...]:
        """Warn when a folder holds files this run did not write and cannot track."""
        if prior is not None:
            return ()
        incoming = {self.final_path(path) for path in staged}
        existing = [
            path
            for name in MANAGED_DIRECTORIES
            for path in (self.output_dir / name).rglob("*")
            if (self.output_dir / name).is_dir() and path.is_file() and path not in incoming
        ]
        if not existing:
            return ()
        sample = ", ".join(sorted(path.name for path in existing)[:3])
        return (
            f"{len(existing)} pre-existing file(s) in this folder were not written by Shapearator "
            f"and were left in place ({sample}{', ...' if len(existing) > 3 else ''}). "
            "They will not be cleaned up by future runs.",
        )

    def _count_preserved(self) -> int:
        tracked = {path for path in managed_files(read_manifest(self.output_dir), self.output_dir)}
        return sum(
            1
            for name in MANAGED_DIRECTORIES
            if (self.output_dir / name).is_dir()
            for path in (self.output_dir / name).rglob("*")
            if path.is_file() and path not in tracked
        )

    def _prune_empty_managed_dirs(self) -> None:
        for name in MANAGED_DIRECTORIES:
            directory = self.output_dir / name
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()

    def _write_manifest(
        self,
        input_path: Path,
        formats: set[str],
        icon_count: int,
        app_version: str,
        staged: list[Path],
    ) -> None:
        payload = {
            "manifest_version": MANIFEST_VERSION,
            "app_version": app_version,
            # Local bookkeeping, but it travels if the folder is shared: keep
            # enough to identify the source sheet, minus the account name.
            "input": scrub_local_paths(str(input_path)),
            "input_name": input_path.name,
            "formats": sorted(formats),
            "icon_count": icon_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": [
                self.final_path(path).relative_to(self.output_dir).as_posix() for path in staged
            ],
        }
        (self.output_dir / MANIFEST_NAME).write_text(
            json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
        )
