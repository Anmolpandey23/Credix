from pathlib import Path

import pytest

from ingestion.service import IngestionService
from utils.file_utils import UnsupportedFileTypeError, store_files


def test_same_filename_does_not_overwrite_existing_document(tmp_path: Path):
    first = tmp_path / "company_a" / "annual_report.pdf"
    second = tmp_path / "company_b" / "annual_report.pdf"

    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"company A financial report")
    second.write_bytes(b"company B financial report")

    storage = tmp_path / "storage"

    first_stored = store_files([first], storage)[0]
    second_stored = store_files([second], storage)[0]

    assert first_stored.name == "annual_report.pdf"
    assert second_stored.name != first_stored.name
    assert first_stored.read_bytes() == b"company A financial report"
    assert second_stored.read_bytes() == b"company B financial report"
    assert len(list(storage.iterdir())) == 2


def test_reingesting_same_file_does_not_create_another_copy(tmp_path: Path):
    source = tmp_path / "annual_report.json"
    source.write_text('{"revenue": 100}', encoding="utf-8")

    storage = tmp_path / "storage"
    service = IngestionService(storage_dir=storage)

    first = service.ingest_files([source])
    second = service.ingest_files([source])

    assert first == second
    assert len(list(storage.iterdir())) == 1


def test_collision_destination_is_stable_on_reingestion(tmp_path: Path):
    first = tmp_path / "a" / "report.csv"
    second = tmp_path / "b" / "report.csv"
    first.parent.mkdir()
    second.parent.mkdir()

    first.write_text("revenue\n100", encoding="utf-8")
    second.write_text("revenue\n200", encoding="utf-8")

    storage = tmp_path / "storage"
    store_files([first], storage)
    first_collision = store_files([second], storage)[0]
    second_collision = store_files([second], storage)[0]

    assert first_collision == second_collision
    assert first_collision.read_text(encoding="utf-8") == "revenue\n200"
    assert len(list(storage.iterdir())) == 2


def test_unsupported_file_type_is_rejected(tmp_path: Path):
    source = tmp_path / "report.xlsx"
    source.write_bytes(b"not supported by current configuration")

    with pytest.raises(UnsupportedFileTypeError):
        store_files([source], tmp_path / "storage")


def test_missing_input_file_is_rejected(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        store_files([tmp_path / "missing.pdf"], tmp_path / "storage")
