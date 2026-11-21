import json
from pathlib import Path

import pytest

from road_accidents.config import PipelineError


def test_training_never_opens_any_test_file(synthetic_sources, monkeypatch):
    from road_accidents.data.pipeline import build_dataset, validate_dataset
    from road_accidents.experiments.schemas import ModelConfig
    from road_accidents.experiments.training import train_run

    inputs = synthetic_sources
    version = build_dataset(
        inputs["root"], inputs["config"], inputs["snapshots"], inputs["osm"], inputs["coverage"]
    )
    validate_dataset(inputs["root"], version.dataset_version_id)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert not {"test_2025", "test_2026"} & set(path.parts), str(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    run = train_run(
        inputs["root"],
        version.dataset_version_id,
        ModelConfig(model_family="random_forest", parameters={"n_estimators": 4, "max_depth": 2}),
        seed=101,
    )
    assert run.status == "completed"
    assert run.validation_metrics and not list((inputs["root"] / run.directory).glob("*test*"))
    assert run.benchmark["fit_seconds"] > 0
    assert run.benchmark["validation_prediction_seconds"] > 0
    assert run.benchmark["model_bytes"] > 0
    assert run.benchmark["sampled_peak_rss_mib"] > 0
    assert (inputs["root"] / run.directory / "benchmark.json").is_file()
    predictions = __import__("pandas").read_parquet(inputs["root"] / run.validation_predictions)
    assert set(predictions.split) == {"validation"}


def test_training_rejects_unknown_model_and_future_configuration():
    from pydantic import ValidationError

    from road_accidents.experiments.schemas import ModelConfig

    with pytest.raises(ValidationError):
        ModelConfig(model_family="unknown")
    with pytest.raises(ValidationError):
        ModelConfig(model_family="random_forest", splits={"train": "2026"})


def test_training_cli_uses_same_run_idempotently(synthetic_sources, capsys):
    import yaml

    from road_accidents.cli import main
    from road_accidents.data.pipeline import build_dataset, validate_dataset

    inputs = synthetic_sources
    version = build_dataset(
        inputs["root"], inputs["config"], inputs["snapshots"], inputs["osm"], inputs["coverage"]
    )
    validate_dataset(inputs["root"], version.dataset_version_id)
    config = inputs["root"] / "rf.yaml"
    config.write_text(
        yaml.safe_dump({"model_family": "random_forest", "parameters": {"n_estimators": 4}})
    )
    arguments = [
        "--work-dir",
        str(inputs["root"]),
        "--result-format",
        "json",
        "train",
        "--dataset-version",
        version.dataset_version_id,
        "--model",
        "random_forest",
        "--model-config",
        str(config),
        "--feature-set",
        "full",
        "--seed",
        "101",
    ]
    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["outputs"] == first["outputs"]


def test_any_recorded_test_opening_blocks_new_training(tmp_path):
    from road_accidents.experiments.training import assert_training_allowed
    from road_accidents.provenance import append_event

    append_event(
        tmp_path / "artifacts/protocol/test-access.jsonl",
        {"event": "test_opened", "development_digest": "digest", "bundle_id": "bundle"},
    )
    with pytest.raises(PipelineError) as error:
        assert_training_allowed(tmp_path, "digest")
    assert error.value.exit_code == 5


def test_development_matrix_selects_all_five_models_without_test_outputs(synthetic_sources):
    import yaml

    from road_accidents.data.pipeline import build_dataset, validate_dataset
    from road_accidents.experiments.schemas import FAMILIES, ExperimentConfig
    from road_accidents.experiments.training import load_run, run_experiment

    inputs = synthetic_sources
    version = build_dataset(
        inputs["root"], inputs["config"], inputs["snapshots"], inputs["osm"], inputs["coverage"]
    )
    validate_dataset(inputs["root"], version.dataset_version_id)
    paths = {}
    for family in FAMILIES:
        path = inputs["root"] / f"{family}.yaml"
        parameters = (
            {"n_estimators": 4, "max_depth": 2}
            if family in {"random_forest", "xgboost"}
            else {"epochs": 1, "hidden_channels": 8, "batch_size": 32, "dropout": 0.0}
        )
        path.write_text(yaml.safe_dump({"model_family": family, "parameters": parameters}))
        paths[family] = path.name
    config = ExperimentConfig(
        profile="smoke_only", models=paths, seeds=[101], feature_sets=["full"]
    )
    group = run_experiment(inputs["root"], version.dataset_version_id, config)
    assert len(group["selected_run_ids"]) == 5
    assert {
        load_run(inputs["root"], identifier).model_family
        for identifier in group["selected_run_ids"]
    } == set(FAMILIES)
    assert (
        run_experiment(inputs["root"], version.dataset_version_id, config)["parent_run_id"]
        == group["parent_run_id"]
    )
