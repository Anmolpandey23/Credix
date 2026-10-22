import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ai-brain"))

import main as main_module


class FakePipeline:
    last_kwargs = None

    def __init__(self, storage_dir=None):
        self.storage_dir = storage_dir

    def run(self, **kwargs):
        FakePipeline.last_kwargs = kwargs
        return {"status": "success", "inspection": kwargs["inspection_data"]}


def test_cli_passes_inspection_json_to_pipeline(tmp_path, monkeypatch, capsys):
    inspection = {
        "capacity_utilization": 0.78,
        "machinery": "Operational",
        "workforce": 85,
        "expected_workforce": 90,
        "findings": [],
    }
    inspection_file = tmp_path / "inspection.json"
    inspection_file.write_text(json.dumps(inspection), encoding="utf-8")

    monkeypatch.setattr(main_module, "AIBrainPipeline", FakePipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--company",
            "Example Ltd",
            "--files",
            "financials.pdf",
            "--inspection-json",
            str(inspection_file),
            "--storage-dir",
            str(tmp_path / "storage"),
        ],
    )

    main_module.main()

    assert FakePipeline.last_kwargs["inspection_data"] == inspection
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "success"
    assert output["inspection"] == inspection


def test_cli_rejects_malformed_inspection_json(tmp_path, monkeypatch, capsys):
    inspection_file = tmp_path / "inspection.json"
    inspection_file.write_text('{"capacity_utilization": ', encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--company",
            "Example Ltd",
            "--files",
            "financials.pdf",
            "--inspection-json",
            str(inspection_file),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main_module.main()

    assert exc.value.code == 2
    assert "Could not read inspection JSON" in capsys.readouterr().err


def test_cli_rejects_non_object_inspection_json(tmp_path, monkeypatch, capsys):
    inspection_file = tmp_path / "inspection.json"
    inspection_file.write_text('[{"capacity_utilization": 0.78}]', encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--company",
            "Example Ltd",
            "--files",
            "financials.pdf",
            "--inspection-json",
            str(inspection_file),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main_module.main()

    assert exc.value.code == 2
    assert "--inspection-json must contain a JSON object" in capsys.readouterr().err
