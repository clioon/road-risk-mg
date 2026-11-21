import pandas as pd
import pytest

from road_accidents.config import PipelineError


def test_coverage_cannot_invent_unloaded_history(synthetic_sources):
    from road_accidents.data.coverage import CoverageLedger
    from road_accidents.data.pipeline import build_dataset
    from tests.fixtures.prf_factory import coverage_document

    inputs = synthetic_sources
    extra = coverage_document(2026, "sha256:" + "f" * 64)
    ledger = CoverageLedger.from_documents(
        inputs["coverage_documents"] + [extra], profile="smoke_only"
    )
    with pytest.raises(PipelineError, match="snapshot"):
        build_dataset(inputs["root"], inputs["config"], inputs["snapshots"], inputs["osm"], ledger)


def test_corrupt_dataset_cannot_be_sealed(synthetic_sources):
    from road_accidents.data.pipeline import build_dataset, validate_dataset

    inputs = synthetic_sources
    version = build_dataset(
        inputs["root"], inputs["config"], inputs["snapshots"], inputs["osm"], inputs["coverage"]
    )
    path = inputs["root"] / version.partition_manifests["test_2026"]["features"]
    path.write_bytes(b"corrupted fixture")
    with pytest.raises(PipelineError, match="integrity"):
        validate_dataset(inputs["root"], version.dataset_version_id)


def test_dataset_seals_idempotently_without_training(synthetic_sources):
    from road_accidents.data.pipeline import build_dataset, validate_dataset

    inputs = synthetic_sources
    first = build_dataset(
        inputs["root"], inputs["config"], inputs["snapshots"], inputs["osm"], inputs["coverage"]
    )
    assert first.status == "validated"
    sealed = validate_dataset(inputs["root"], first.dataset_version_id)
    assert sealed.status == "sealed"
    second = build_dataset(
        inputs["root"], inputs["config"], inputs["snapshots"], inputs["osm"], inputs["coverage"]
    )
    assert second.dataset_version_id == first.dataset_version_id
    assert second.development_digest == first.development_digest
    assert set(first.partition_manifests) == {"train", "validation", "test_2025", "test_2026"}
    for partition in first.partition_manifests.values():
        labels = pd.read_parquet(inputs["root"] / partition["labels"])
        features = pd.read_parquet(inputs["root"] / partition["features"])
        assert not features.empty and set(labels.label) == {0, 1}
        assert not {"label", "event_count", "causa_acidente", "degree"} & set(features)
        assert len(labels) == len(features)


def test_revised_test_keeps_development_digest(synthetic_sources):
    from road_accidents.data.coverage import CoverageLedger
    from road_accidents.data.pipeline import build_dataset
    from road_accidents.data.prf import ingest_prf
    from tests.fixtures.prf_factory import coverage_document, prf_bytes

    inputs = synthetic_sources
    first = build_dataset(
        inputs["root"], inputs["config"], inputs["snapshots"], inputs["osm"], inputs["coverage"]
    )
    revised = ingest_prf(
        prf_bytes(2026, revision=True),
        inputs["root"],
        year=2026,
        source_url="https://example.org/2026.csv",
        profile="smoke_only",
    )
    snapshots = inputs["snapshots"][:-1] + [revised]
    coverage = CoverageLedger.from_documents(
        [coverage_document(snapshot.nominal_year, snapshot.snapshot_id) for snapshot in snapshots],
        profile="smoke_only",
    )
    second = build_dataset(inputs["root"], inputs["config"], snapshots, inputs["osm"], coverage)
    assert first.dataset_version_id != second.dataset_version_id
    assert first.development_digest == second.development_digest


def test_dvc_prepares_selected_sources_offline(synthetic_sources, project_root):
    import configparser
    import os
    import shutil
    import subprocess
    import sys

    import yaml
    from dvc.repo import Repo

    from road_accidents.data.storage import write_json

    inputs = synthetic_sources
    root = inputs["root"]
    Repo.init(str(root), no_scm=True)
    dvc_config = configparser.ConfigParser()
    dvc_config.read([root / ".dvc/config", project_root / ".dvc/config"])
    with (root / ".dvc/config").open("w") as output:
        dvc_config.write(output)
    for filename in ("dvc.yaml", "pyproject.toml", "uv.lock"):
        shutil.copyfile(project_root / filename, root / filename)
    shutil.copytree(
        project_root / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    (root / "configs").mkdir()
    (root / "configs/dataset.yaml").write_text(
        yaml.safe_dump(inputs["config"].model_dump(mode="json"))
    )
    (root / "configs/coverage.yaml").write_text(
        yaml.safe_dump({"profile": "smoke_only", "sources": inputs["coverage_documents"]})
    )
    write_json(
        root / "data/manifests/prf-selected.json",
        {
            "schema_version": "3.0.0",
            "sources": [source.model_dump(mode="json") for source in inputs["snapshots"]],
        },
    )
    write_json(
        root / "data/manifests/osm-selected.json",
        {"schema_version": "3.0.0", "sources": [inputs["osm"].model_dump(mode="json")]},
    )
    environment = dict(os.environ)
    environment["PATH"] = (
        str(__import__("pathlib").Path(sys.executable).parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    result = subprocess.run(
        [sys.executable, "-m", "dvc", "repro", "prepare"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list((root / "data/processed/datasets").glob("*/sealed.json"))
    assert (root / "data/manifests/prepared.json").is_file()
    stages = yaml.safe_load((root / "dvc.yaml").read_text())["stages"]
    assert set(stages) == {"prepare", "experiment", "report"}
    from road_accidents.experiments.schemas import FAMILIES
    from road_accidents.reporting.packaging import verify_package

    (root / "configs/models").mkdir()
    (root / "configs/experiments").mkdir()
    models = {}
    for family in FAMILIES:
        parameters = (
            {"n_estimators": 4, "max_depth": 2}
            if family in {"random_forest", "xgboost"}
            else {"epochs": 1, "hidden_channels": 8, "batch_size": 32, "dropout": 0.0}
        )
        model_path = f"configs/models/{family}.yaml"
        (root / model_path).write_text(
            yaml.safe_dump({"model_family": family, "parameters": parameters})
        )
        models[family] = model_path
    (root / "configs/experiments/full.yaml").write_text(
        yaml.safe_dump(
            {"profile": "smoke_only", "models": models, "seeds": [101], "feature_sets": ["full"]}
        )
    )
    completed = subprocess.run(
        [sys.executable, "-m", "dvc", "repro", "report"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    package_manifest = next((root / "artifacts/packages").glob("*/package-manifest.json"))
    assert verify_package(package_manifest).profile == "smoke_only"
    with Repo(str(root)) as repository:
        assert repository.status() == {}
