"""File handling helpers for ingestion."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List

from config.settings import SUPPORTED_EXTENSIONS


class UnsupportedFileTypeError(ValueError):
    pass


def ensure_supported(file_path: Path) -> None:
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Unsupported file type: {file_path.suffix}")


def ensure_storage(storage_dir: Path) -> Path:
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def store_files(input_files: Iterable[Path], storage_dir: Path) -> List[Path]:
    ensure_storage(storage_dir)
    stored_paths: List[Path] = []

    for source in input_files:
        ensure_supported(source)
        destination = storage_dir / source.name
        # Keep exact bytes to preserve source-of-truth values.
        if source.resolve() == destination.resolve():
            stored_paths.append(destination)
            continue

        shutil.copy2(source, destination)
        stored_paths.append(destination)

    return stored_paths
