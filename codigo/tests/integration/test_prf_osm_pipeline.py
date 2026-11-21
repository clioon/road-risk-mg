import json


def test_public_smoke_integrates_sources_models_report_and_package(tmp_path, project_root, capsys):
    from road_accidents.cli import main

    root = tmp_path / "smoke"
    args = [
        "--work-dir",
        str(root),
        "--result-format",
        "json",
        "pipeline",
        "smoke",
        "--config",
        str(project_root / "configs/experiments/smoke.yaml"),
    ]
    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "succeeded"
    assert len(first["outputs"]["evaluation_ids"]) == 2
    assert first["outputs"]["profile"] == "smoke_only"
    assert (root / first["outputs"]["package_manifest"]).is_file()
    assert first["outputs"]["reproduction"]["status"] == "succeeded"
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["outputs"]["bundle_id"] == first["outputs"]["bundle_id"]
