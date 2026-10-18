"""File handling helpers for safe financial-document ingestion."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Iterable, List

from config.settings import SUPPORTED_EXTENSIONS


class UnsupportedFileTypeError(ValueError):
    """Raised when an unsupported document type is provided."""


def ensure_supported(file_path: Path) -> None:
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: {file_path.suffix or '(no extension)'}"
        )


def ensure_storage(storage_dir: Path) -> Path:
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def _same_content(left: Path, right: Path) -> bool:
    """Compare file contents without relying on filenames or file sizes."""
    if not right.is_file() or left.stat().st_size != right.stat().st_size:
        return False

    with left.open("rb") as left_file, right.open("rb") as right_file:
        while True:
            left_chunk = left_file.read(1024 * 1024)
            right_chunk = right_file.read(1024 * 1024)

            if left_chunk != right_chunk:
                return False

            if not left_chunk:
                return True


def _collision_destination(source: Path, storage_dir: Path) -> Path:
    """Generate a deterministic, source-specific collision-safe filename."""
    source_id = hashlib.sha256(
        str(source.resolve()).encode("utf-8")
    ).hexdigest()[:10]

    content_hash = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            content_hash.update(chunk)

    content_id = content_hash.hexdigest()[:10]
    return storage_dir / (
        f"{source.stem}__{source_id}_{content_id}{source.suffix}"
    )


def store_files(input_files: Iterable[Path], storage_dir: Path) -> List[Path]:
    """Store supported files without silently overwriting different documents."""
    ensure_storage(storage_dir)
    stored_paths: List[Path] = []

    for source in input_files:
        if not source.is_file():
            raise FileNotFoundError(f"Input file not found: {source}")

        ensure_supported(source)

        destination = storage_dir / source.name

        # Do not copy a file onto itself.
        if source.resolve() == destination.resolve():
            stored_paths.append(destination)
            continue

        # Preserve the original filename when it is available.
        if not destination.exists():
            shutil.copy2(source, destination)
            stored_paths.append(destination)
            continue

        # Re-ingesting identical content is safe and idempotent.
        if _same_content(source, destination):
            stored_paths.append(destination)
            continue

        # A different document has the same filename. Never overwrite it.
        destination = _collision_destination(source, storage_dir)

        if destination.exists():
            if not _same_content(source, destination):
                raise FileExistsError(
                    "Collision-safe destination already exists with "
                    f"different content: {destination}"
                )
        else:
            shutil.copy2(source, destination)

        stored_paths.append(destination)

    return stored_paths
