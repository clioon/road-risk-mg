import json
import os
import subprocess
import sys

import yaml

from road_accidents.cli import main
from tests.fixtures.prf_factory import prf_bytes


def test_import_dry_run_does_not_persist(tmp_path, capsys):
    original = tmp_path / "source.csv"
    original.write_bytes(prf_bytes(2026))
    root = tmp_path / "work"
    code = main(
        [
            "--work-dir",
            str(root),
            "--result-format",
            "json",
            "--dry-run",
            "sources",
            "import-prf",
            "--file",
            str(original),
            "--year",
            "2026",
            "--source-url",
            "https://example.org/2026.csv",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result["status"] == "succeeded"
    assert not root.exists()


def test_dvc_import_stdout_is_one_json_document(tmp_path, project_root):
    root = tmp_path / "work"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True, capture_output=True)
    environment = os.environ.copy()
    environment["DVC_NO_ANALYTICS"] = "true"
    environment["PYTHONPATH"] = str(project_root / "src")
    subprocess.run(
        [sys.executable, "-m", "dvc", "init", "--quiet"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
    )
    original = tmp_path / "source.csv"
    original.write_bytes(prf_bytes(2026))
    execution = subprocess.run(
        [
            sys.executable,
            "-m",
            "road_accidents.cli",
            "--work-dir",
            str(root),
            "--result-format",
            "json",
            "sources",
            "import-prf",
            "--file",
            str(original),
            "--year",
            "2026",
            "--source-url",
            "https://example.org/2026.csv",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert execution.returncode == 0, execution.stderr
    result = json.loads(execution.stdout)
    assert result["status"] == "succeeded"
    assert len(result["outputs"]["snapshot_ids"]) == 1
    assert len(list((root / "data").rglob("*.dvc"))) == 3
    assert original.read_bytes() == prf_bytes(2026)


def test_dataset_commands_publish_structure_without_learned_inputs(synthetic_sources, capsys):
    from road_accidents.data.storage import write_json

    inputs = synthetic_sources
    root = inputs["root"]
    write_json(
        root / "prf-manifest.json",
        {
            "schema_version": "3.0.0",
            "sources": [snapshot.model_dump(mode="json") for snapshot in inputs["snapshots"]],
        },
    )
    write_json(
        root / "osm-manifest.json",
        {"schema_version": "3.0.0", "sources": [inputs["osm"].model_dump(mode="json")]},
    )
    (root / "dataset.yaml").write_text(yaml.safe_dump(inputs["config"].model_dump(mode="json")))
    (root / "coverage.yaml").write_text(
        yaml.safe_dump({"profile": "smoke_only", "sources": inputs["coverage_documents"]})
    )
    code = main(
        [
            "--work-dir",
            str(root),
            "--result-format",
            "json",
            "dataset",
            "build",
            "--config",
            str(root / "dataset.yaml"),
            "--prf-manifest",
            str(root / "prf-manifest.json"),
            "--osm-manifest",
            str(root / "osm-manifest.json"),
            "--coverage-manifest",
            str(root / "coverage.yaml"),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0, result
    assert result["outputs"]["graph_manifest"]
    version = result["outputs"]["dataset_version_id"]
    code = main(
        [
            "--work-dir",
            str(root),
            "--result-format",
            "json",
            "dataset",
            "validate",
            "--dataset-version",
            version,
        ]
    )
    assert code == 0, capsys.readouterr().out
    assert not list(root.rglob("*.pt")) and not list(root.rglob("*.pkl"))


def test_fetch_prf_catalog_reuses_eight_selected_local_sources(tmp_path, capsys):
    root = tmp_path / "managed"
    years = {}
    for year in range(2019, 2027):
        original = tmp_path / f"{year}.csv"
        original.write_bytes(prf_bytes(year))
        years[str(year)] = {"url": f"https://example.org/{year}.csv", "local_file": str(original)}
    config = tmp_path / "sources.yaml"
    config.write_text(yaml.safe_dump({"profile": "smoke_only", "years": years}))
    arguments = [
        "--work-dir",
        str(root),
        "--result-format",
        "json",
        "sources",
        "fetch-prf",
        "--config",
        str(config),
        "--years",
        "2019:2026",
        "--output",
        "custom/prf",
    ]
    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert len(first["outputs"]["snapshot_ids"]) == 8
    assert len(list((root / "custom/prf").glob("*/source.bin"))) == 8
    assert main(arguments) == 0
    second = json.loads(capsys.readouterr().out)
    assert first["outputs"] == second["outputs"]


def test_osm_dry_run_never_downloads_or_persists(tmp_path, capsys):
    root = tmp_path / "managed"
    arguments = [
        "--work-dir",
        str(root),
        "--result-format",
        "json",
        "--dry-run",
        "sources",
        "fetch-osm",
        "--state",
        "MG",
        "--effective-at",
        "2018-12-31T23:59:59Z",
    ]
    assert main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["outputs"]["dry_run"]
    assert not root.exists()
