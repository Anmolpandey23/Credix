"""Data ingestion service."""

from pathlib import Path
from typing import Iterable, List

from config.settings import DEFAULT_STORAGE_DIR, SUPPORTED_EXTENSIONS
from utils.file_utils import store_files


class IngestionService:
    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir or DEFAULT_STORAGE_DIR

    def ingest_files(self, file_paths: Iterable[str | Path]) -> List[Path]:
        normalized_paths = [Path(path) for path in file_paths]
        resolved_paths: List[Path] = []

        for source in normalized_paths:
            if source.exists() and source.is_dir():
                supported = [
                    child
                    for child in sorted(source.iterdir())
                    if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
                ]
                resolved_paths.extend(supported)
                continue

            if source.exists() and source.is_file():
                resolved_paths.append(source)
                continue

            inferred = self._infer_supported_file(source)
            if inferred is not None:
                resolved_paths.append(inferred)
                continue

            raise FileNotFoundError(f"Input path not found: {source}")

        if not resolved_paths:
            raise ValueError("No supported input files found")

        return store_files(resolved_paths, self.storage_dir)

    def _infer_supported_file(self, source: Path) -> Path | None:
        """Infer a single supported file when user passes a shorthand path."""
        if source.suffix:
            return None

        parent = source.parent
        if not parent.exists() or not parent.is_dir():
            return None

        candidates: List[Path] = []

        for extension in sorted(SUPPORTED_EXTENSIONS):
            direct = source.with_suffix(extension)
            if direct.exists() and direct.is_file():
                candidates.append(direct)

            pattern_matches = [
                path
                for path in parent.glob(f"{source.name}*{extension}")
                if path.is_file()
            ]
            candidates.extend(pattern_matches)

        deduped = sorted({path.resolve() for path in candidates})
        if len(deduped) == 1:
            return deduped[0]
        return None
