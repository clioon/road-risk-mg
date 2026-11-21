from pathlib import Path

import pytest


@pytest.fixture
def synthetic_bundle(synthetic_sources):
    import yaml

    from road_accidents.data.pipeline import build_dataset, validate_dataset
    from road_accidents.experiments.freeze import freeze_experiment
    from road_accidents.experiments.schemas import FAMILIES, ExperimentConfig
    from road_accidents.experiments.training import run_experiment

    inputs = synthetic_sources
    version = build_dataset(
        inputs["root"], inputs["config"], inputs["snapshots"], inputs["osm"], inputs["coverage"]
    )
    validate_dataset(inputs["root"], version.dataset_version_id)
    paths = {}
    for family in FAMILIES:
        config = inputs["root"] / f"{family}.yaml"
        parameters = (
            {"n_estimators": 4, "max_depth": 2}
            if family in {"random_forest", "xgboost"}
            else {"epochs": 1, "hidden_channels": 8, "batch_size": 16, "dropout": 0.0}
        )
        config.write_text(yaml.safe_dump({"model_family": family, "parameters": parameters}))
        paths[family] = config.name
    config = ExperimentConfig(
        profile="smoke_only", models=paths, seeds=[101], feature_sets=["full"]
    )
    group = run_experiment(inputs["root"], version.dataset_version_id, config)
    bundle = freeze_experiment(inputs["root"], group["parent_run_id"])
    return inputs | {"version": version, "bundle": bundle, "experiment": group}


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("DVC_NO_ANALYTICS", "1")


@pytest.fixture
def synthetic_evaluations(synthetic_bundle):
    from road_accidents.evaluation.runner import evaluate_bundle

    inputs = synthetic_bundle
    main = evaluate_bundle(
        inputs["root"],
        inputs["bundle"].bundle_id,
        inputs["version"].dataset_version_id,
        "test_2025",
    )
    temporal = evaluate_bundle(
        inputs["root"],
        inputs["bundle"].bundle_id,
        inputs["version"].dataset_version_id,
        "test_2026",
    )
    return inputs | {"main_evaluation": main, "temporal_evaluation": temporal}


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_sources(tmp_path: Path, project_root: Path):
    import geopandas as gpd

    from road_accidents.data.coverage import CoverageLedger
    from road_accidents.data.osm import ingest_osm
    from road_accidents.data.prf import ingest_prf
    from road_accidents.data.schemas import DatasetConfig
    from tests.fixtures.prf_factory import coverage_document, prf_bytes

    root = tmp_path / "work"
    snapshots = [
        ingest_prf(
            prf_bytes(year),
            root,
            year=year,
            source_url=f"https://example.org/{year}.csv",
            profile="smoke_only",
        )
        for year in range(2019, 2027)
    ]
    osm = ingest_osm(
        (project_root / "tests/fixtures/network.osm").read_bytes(),
        root,
        effective_at="2018-12-31T23:59:59Z",
        profile="smoke_only",
    )
    boundary = root / "boundary.geojson"
    boundary.write_bytes((project_root / "tests/fixtures/mg_boundary.geojson").read_bytes())
    documents = [
        coverage_document(snapshot.nominal_year, snapshot.snapshot_id) for snapshot in snapshots
    ]
    ledger = CoverageLedger.from_documents(documents, profile="smoke_only")
    return {
        "root": root,
        "snapshots": snapshots,
        "osm": osm,
        "coverage": ledger,
        "coverage_documents": documents,
        "boundary": gpd.read_file(boundary),
        "config": DatasetConfig(profile="smoke_only", boundary_path="boundary.geojson"),
    }
