import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from filelock import FileLock

from road_accidents.config import PipelineError, load_config
from road_accidents.data.pipeline import dataset_directory
from road_accidents.data.schemas import SPLITS
from road_accidents.data.storage import (
    atomic_directory,
    read_json,
    read_parquet,
    safe_path,
    write_json,
    write_parquet,
)
from road_accidents.evaluation.ablation import feature_columns
from road_accidents.evaluation.metrics import classification_metrics, select_threshold
from road_accidents.experiments.schemas import (
    FAMILIES,
    SEEDS,
    ExperimentConfig,
    ExperimentRun,
    FeatureSet,
    ModelConfig,
)
from road_accidents.features.preprocessing import SharedPreprocessor
from road_accidents.models.baselines import fit_model, load_model, save_model, score_model
from road_accidents.provenance import (
    MemorySampler,
    RunTracker,
    canonical_hash,
    capture_code_state,
    hash_file,
)


def assert_training_allowed(root: Path, development_digest: str) -> None:
    path = root / "artifacts/protocol/test-access.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as source:
            for line in source:
                event = json.loads(line)
                if event.get("event") == "test_opened":
                    raise PipelineError(
                        "Recorded test access forbids further fitting or reselection.",
                        5,
                        "test_already_opened",
                    )


def load_development(root: Path, dataset_version_id: str) -> dict[str, Any]:
    directory = dataset_directory(root, dataset_version_id)
    if not (directory / "sealed.json").is_file():
        raise PipelineError("Training requires a sealed dataset.")
    seal = read_json(directory / "sealed.json")
    if seal.get("dataset_version_id") != dataset_version_id or seal.get(
        "development_sha256"
    ) != hash_file(directory / "development.json"):
        raise PipelineError("Development view failed integrity verification.")
    view: dict[str, Any] = read_json(directory / "development.json")
    if (
        view.get("schema_version") != "3.0.0"
        or view.get("dataset_version_id") != dataset_version_id
        or set(view["partitions"]) != {"train", "validation"}
    ):
        raise PipelineError(
            "Training accepts only the restricted development view.", 5, "temporal_integrity"
        )
    frames = {}
    for split, partition in view["partitions"].items():
        loaded = {}
        for kind in ("features", "labels"):
            path = safe_path(root, partition[kind])
            if not path.is_relative_to(directory / "segment_week" / split):
                raise PipelineError(
                    "Development path escapes its temporal partition.", 5, "temporal_integrity"
                )
            if hash_file(path) != partition[kind + "_sha256"]:
                raise PipelineError("Development file integrity failed.")
            loaded[kind] = read_parquet(path)
        features, labels = loaded["features"], loaded["labels"]
        if not features[["segment_id", "week_start"]].equals(labels[["segment_id", "week_start"]]):
            raise PipelineError(
                "Development features and labels do not align.", 7, "population_mismatch"
            )
        dates = pd.to_datetime(features.week_start).dt.date
        begin, end = SPLITS[split]
        if (
            features.empty
            or not ((dates >= begin) & (dates < end)).all()
            or set(labels.label.unique()) != {0, 1}
        ):
            raise PipelineError(
                "Development needs correct dates and both classes.", 6, "invalid_development"
            )
        frames[split] = (features, labels)
    graph_path = safe_path(root, view["graph_manifest"])
    if (
        not graph_path.is_relative_to(directory / "graph")
        or hash_file(graph_path) != view["graph_manifest_sha256"]
    ):
        raise PipelineError("Graph manifest failed integrity verification.")
    graph = read_json(graph_path)
    for kind in ("nodes", "adjacency"):
        path = safe_path(root, graph[kind])
        if (
            not path.is_relative_to(directory / "graph")
            or hash_file(path) != graph[kind + "_sha256"]
        ):
            raise PipelineError("Graph input failed integrity verification.")
        graph[kind + "_frame"] = read_parquet(path)
    view["frames"], view["graph"] = frames, graph
    return view


def archive_code(root: Path) -> dict[str, Any]:
    from road_accidents.provenance import runtime_environment

    code_root = Path(__file__).resolve().parents[3]
    state = capture_code_state(code_root)
    relative = f"artifacts/code/{state['code_hash'][7:]}"
    destination = safe_path(root, relative)
    if not destination.exists():
        with atomic_directory(root, relative) as staging:
            for entry in state["inventory"]:
                output = safe_path(staging / "files", entry["path"])
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(code_root / entry["path"], output)
                if hash_file(output) != entry["sha256"]:
                    raise PipelineError("Code changed while capturing the exact source snapshot.")
            write_json(staging / "manifest.json", state)
            write_json(staging / "environment.json", runtime_environment())
    if read_json(destination / "environment.json") != runtime_environment():
        raise PipelineError(
            "Installed environment differs from the recorded code snapshot.",
            5,
            "environment_changed",
        )
    state["environment_lock_hash"] = hash_file(code_root / "uv.lock")
    return state


def run_path(root: Path, run_id: str) -> Path:
    if (
        not run_id.startswith("sha256:")
        or len(run_id) != 71
        or any(character not in "0123456789abcdef" for character in run_id[7:])
    ):
        raise PipelineError("An exact run identifier is required.", 2, "usage_error")
    return safe_path(root, f"artifacts/runs/{run_id[7:]}")


def load_run(root: Path, run_id: str) -> ExperimentRun:
    run = ExperimentRun.model_validate(read_json(run_path(root, run_id) / "manifest.json"))
    if run.run_id != run_id or run.status != "completed":
        raise PipelineError("Only the identified completed run can be loaded.")
    for relative, digest in run.artifact_hashes.items():
        path = safe_path(root, relative)
        if not path.is_file() or hash_file(path) != digest:
            raise PipelineError(
                "Trained artifact failed integrity verification.", 5, "model_changed"
            )
    return run


def train_run(
    root: Path,
    dataset_version_id: str,
    config: ModelConfig,
    *,
    seed: int,
    feature_set: FeatureSet = "full",
    parent_run_id: str | None = None,
    force: bool = False,
) -> ExperimentRun:
    view = load_development(root, dataset_version_id)
    if seed not in SEEDS and view["profile"] != "smoke_only":
        raise PipelineError("Seed is not in the declared confirmatory protocol.", 2, "usage_error")
    if feature_set != "full" and config.model_family not in {"graphsage", "gatv2"}:
        raise PipelineError("Ablations require graph models.", 2, "usage_error")
    code = archive_code(root)
    configuration_hash = canonical_hash(config.model_dump(mode="json"))
    identity = {
        "development": view["development_digest"],
        "config": configuration_hash,
        "feature_set": feature_set,
        "seed": seed,
        "code": code["code_hash"],
        "lock": code["environment_lock_hash"],
        "parent_run_id": parent_run_id,
    }
    if force:
        identity["attempt"] = uuid4().hex
    identifier = canonical_hash(identity)
    destination = run_path(root, identifier)
    if destination.exists() and not force:
        return load_run(root, identifier)
    locks = root / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with FileLock(str(locks / "protocol.lock")), MemorySampler() as memory:
        assert_training_allowed(root, view["development_digest"])
        if destination.exists():
            return load_run(root, identifier)
        train_features, train_labels = view["frames"]["train"]
        validation_features, validation_labels = view["frames"]["validation"]
        columns = feature_columns(feature_set)
        print(
            f"Training: {config.model_family}/{feature_set}, seed {seed}.",
            file=sys.stderr,
            flush=True,
        )
        started = datetime.now(UTC)
        preprocessing_started = time.perf_counter()
        processor = SharedPreprocessor(columns).fit(train_features)
        training_matrix = processor.transform(train_features)
        validation_matrix = processor.transform(validation_features)
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        tracker = RunTracker(root)
        tracking_parameters = {
            "dataset_version_id": dataset_version_id,
            "development_digest": view["development_digest"],
            "model_family": config.model_family,
            "feature_set": feature_set,
            "seed": str(seed),
            "configuration_hash": configuration_hash,
            "code_hash": code["code_hash"],
            "environment_lock_hash": code["environment_lock_hash"],
        }
        with tracker.run(tracking_parameters) as mlflow_run_id:
            fit_started = time.perf_counter()
            if config.model_family in {"graphsage", "gatv2"}:
                from road_accidents.graph.dataset import fit_graph_model, score_graph_model

                graph = view["graph"]
                model = fit_graph_model(
                    config.model_family,
                    train_features,
                    training_matrix,
                    train_labels.label.to_numpy(),
                    graph["nodes_frame"],
                    graph["adjacency_frame"],
                    config.parameters,
                    seed,
                    without_topology=feature_set == "without_topology",
                )
                fit_seconds = time.perf_counter() - fit_started
                prediction_started = time.perf_counter()
                scores = score_graph_model(
                    model,
                    validation_features,
                    validation_matrix,
                    graph["nodes_frame"],
                    graph["adjacency_frame"],
                    without_topology=feature_set == "without_topology",
                )
            else:
                model = fit_model(
                    config.model_family,
                    training_matrix,
                    train_labels.label.to_numpy(),
                    config.parameters,
                    seed,
                )
                fit_seconds = time.perf_counter() - fit_started
                prediction_started = time.perf_counter()
                scores = score_model(model, validation_matrix)
            prediction_seconds = time.perf_counter() - prediction_started
            threshold = select_threshold(validation_labels.label.to_numpy(), scores)
            metrics = classification_metrics(validation_labels.label.to_numpy(), scores, threshold)
            prefix = destination.relative_to(root.resolve()).as_posix()
            with atomic_directory(root, prefix) as staging:
                extension = (
                    "ubj"
                    if config.model_family == "xgboost"
                    else "pt"
                    if config.model_family in {"mlp", "graphsage", "gatv2"}
                    else "pkl"
                )
                model_file = f"model.{extension}"
                save_model(model, staging / model_file, config.model_family)
                processor.save(staging / "preprocessor.pkl")
                predictions = validation_labels[["segment_id", "week_start", "label"]].copy()
                predictions["score"] = scores
                predictions["predicted_label"] = (scores >= threshold).astype(int)
                predictions["split"] = "validation"
                predictions["run_id"] = identifier
                predictions["dataset_version_id"] = dataset_version_id
                predictions["prediction_set_id"] = canonical_hash(
                    {"run_id": identifier, "split": "validation"}
                )
                write_parquet(staging / "validation-predictions.parquet", predictions)
                write_json(staging / "configuration.json", config.model_dump(mode="json"))
                write_json(staging / "metrics.json", metrics)
                benchmark = {
                    "preprocessing_seconds": preprocessing_seconds,
                    "fit_seconds": fit_seconds,
                    "validation_prediction_seconds": prediction_seconds,
                    "sampled_peak_rss_mib": memory.peak_bytes / 1024**2,
                    "memory_measurement": "parent_and_children_rss_sampled_every_50ms",
                    "training_observations": len(train_features),
                    "validation_observations": len(validation_features),
                    "input_dimensions": training_matrix.shape[1],
                    "model_bytes": (staging / model_file).stat().st_size,
                    "compute_device": "cpu",
                    "model_threads": 1,
                }
                write_json(staging / "benchmark.json", benchmark)
                artifact_hashes = {
                    f"{prefix}/{path.name}": hash_file(path)
                    for path in staging.iterdir()
                    if path.is_file()
                }
                run = ExperimentRun(
                    run_id=identifier,
                    mlflow_run_id=mlflow_run_id,
                    parent_run_id=parent_run_id,
                    profile=view["profile"],
                    dataset_version_id=dataset_version_id,
                    development_digest=view["development_digest"],
                    model_family=config.model_family,
                    feature_set=feature_set,
                    seed=seed,
                    configuration_hash=configuration_hash,
                    parameters=config.parameters,
                    git_revision=code["git_revision"],
                    code_hash=code["code_hash"],
                    code_inventory=code["inventory"],
                    environment_lock_hash=code["environment_lock_hash"],
                    started_at=started,
                    finished_at=datetime.now(UTC),
                    decision_threshold=threshold,
                    status="completed",
                    directory=prefix,
                    model_path=f"{prefix}/{model_file}",
                    preprocessor_path=f"{prefix}/preprocessor.pkl",
                    feature_names=processor.feature_names,
                    feature_columns=columns,
                    node_order_hash=view["graph"]["node_order_hash"],
                    validation_predictions=f"{prefix}/validation-predictions.parquet",
                    validation_metrics=metrics,
                    benchmark=benchmark,
                    artifact_hashes=artifact_hashes,
                )
                write_json(staging / "manifest.json", run.model_dump(mode="json"))
            tracker.log_metrics(
                mlflow_run_id,
                {
                    name: result["value"]
                    for name, result in metrics.items()
                    if result["value"] is not None
                },
            )
            tracker.log_artifact(mlflow_run_id, destination / "manifest.json")
        print(
            f"Completed: {config.model_family}/{feature_set}, seed {seed}; "
            f"fit {fit_seconds:.2f}s, validation inference {prediction_seconds:.2f}s.",
            file=sys.stderr,
            flush=True,
        )
    return run


def predict_run(
    root: Path, run: ExperimentRun, features: pd.DataFrame, graph: dict[str, Any] | None = None
) -> np.ndarray[Any, Any]:
    verified = load_run(root, run.run_id)
    processor = SharedPreprocessor.load(safe_path(root, verified.preprocessor_path))
    model = load_model(safe_path(root, verified.model_path), verified.model_family)
    if verified.model_family in {"graphsage", "gatv2"}:
        from road_accidents.graph.dataset import score_graph_model

        if graph is None or verified.node_order_hash != graph["node_order_hash"]:
            raise PipelineError(
                "Graph order differs from the trained model.", 7, "node_order_mismatch"
            )
        return score_graph_model(
            model,
            features,
            processor.transform(features),
            graph["nodes_frame"],
            graph["adjacency_frame"],
            without_topology=verified.feature_set == "without_topology",
        )
    return score_model(model, processor.transform(features))


def group_path(root: Path, parent_run_id: str) -> Path:
    run_path(root, parent_run_id)
    return safe_path(root, f"artifacts/experiments/{parent_run_id[7:]}")


def run_experiment(root: Path, dataset_version_id: str, config: ExperimentConfig) -> dict[str, Any]:
    view = load_development(root, dataset_version_id)
    if config.profile != view["profile"]:
        raise PipelineError("Experiment profile must match its dataset.")
    del view["frames"], view["graph"]
    models = {
        family: load_config(safe_path(root, path), ModelConfig)
        for family, path in config.models.items()
    }
    for family, model in models.items():
        if (
            model.model_family != family
            or len(model.candidates or [{}]) > config.max_candidates_per_variant
        ):
            raise PipelineError(
                "Model configuration violates the declared search budget.", 2, "configuration_error"
            )
    code = archive_code(root)
    identifier = canonical_hash(
        {
            "development_digest": view["development_digest"],
            "configuration": config.model_dump(mode="json"),
            "models": {family: model.model_dump(mode="json") for family, model in models.items()},
            "code_hash": code["code_hash"],
            "lock": code["environment_lock_hash"],
        }
    )
    destination = group_path(root, identifier)
    if destination.exists():
        result: dict[str, Any] = read_json(destination / "manifest.json")
        for run_id in result["selected_run_ids"]:
            load_run(root, run_id)
        return result
    assert_training_allowed(root, view["development_digest"])
    tracker = RunTracker(root)
    selected: list[ExperimentRun] = []
    all_runs: list[str] = []
    scores: dict[str, float] = {}
    with tracker.run(
        {
            "parent_run_id": identifier,
            "profile": config.profile,
            "development_digest": view["development_digest"],
        }
    ) as tracking_id:
        for family in FAMILIES:
            model = models[family]
            variants: list[FeatureSet] = (
                config.feature_sets if family in {"graphsage", "gatv2"} else ["full"]
            )
            for variant in variants:
                candidates: list[tuple[float, str, list[ExperimentRun]]] = []
                for parameters in model.candidates or [{}]:
                    candidate = ModelConfig(
                        model_family=model.model_family, parameters=model.parameters | parameters
                    )
                    runs = [
                        train_run(
                            root,
                            dataset_version_id,
                            candidate,
                            seed=seed,
                            feature_set=variant,
                            parent_run_id=identifier,
                        )
                        for seed in config.seeds
                    ]
                    values = [run.validation_metrics["auprc"]["value"] for run in runs]
                    if any(value is None for value in values):
                        raise PipelineError(
                            "Validation AP cannot be undefined during selection.",
                            6,
                            "invalid_selection",
                        )
                    candidates.append((float(np.mean(values)), runs[0].configuration_hash, runs))
                    all_runs.extend(run.run_id for run in runs)
                mean, _config_hash, chosen = sorted(
                    candidates, key=lambda item: (-item[0], item[1])
                )[0]
                selected.extend(chosen)
                if variant == "full":
                    scores[family] = mean
        reference = sorted(
            ("random_forest", "xgboost", "mlp"),
            key=lambda family: (-scores[family], ("random_forest", "xgboost", "mlp").index(family)),
        )[0]
        result = {
            "schema_version": "3.0.0",
            "parent_run_id": identifier,
            "mlflow_run_id": tracking_id,
            "dataset_version_id": dataset_version_id,
            "development_digest": view["development_digest"],
            "profile": config.profile,
            "status": "completed",
            "selected_run_ids": [run.run_id for run in selected],
            "all_run_ids": all_runs,
            "baseline_reference": reference,
            "validation_average_precision": scores,
            "seeds": config.seeds,
            "ablation_test_2026": config.ablation_test_2026,
            "seasonal_rule": config.seasonal_rule,
            "configuration": config.model_dump(mode="json"),
            "models": {family: model.model_dump(mode="json") for family, model in models.items()},
            "code_hash": code["code_hash"],
            "environment_lock_hash": code["environment_lock_hash"],
        }
        with atomic_directory(root, destination.relative_to(root.resolve())) as staging:
            write_json(staging / "manifest.json", result)
        tracker.log_artifact(tracking_id, destination / "manifest.json")
    return result
