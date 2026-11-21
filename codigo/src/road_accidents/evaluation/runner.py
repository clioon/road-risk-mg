import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from road_accidents.config import PipelineError
from road_accidents.data.pipeline import dataset_directory, load_dataset
from road_accidents.data.storage import (
    atomic_directory,
    read_json,
    read_parquet,
    safe_path,
    write_json,
    write_parquet,
)
from road_accidents.evaluation.comparison import (
    KEYS,
    cohort_manifest,
    population_keys,
    require_same_population,
    seasonal_keys,
)
from road_accidents.evaluation.metrics import METRICS, aggregate_seeds, classification_metrics
from road_accidents.experiments.freeze import load_bundle
from road_accidents.experiments.schemas import EvaluationRun
from road_accidents.experiments.training import load_run, predict_run, run_path
from road_accidents.provenance import (
    MemorySampler,
    RunTracker,
    append_event,
    canonical_hash,
    hash_file,
)


def evaluation_path(root: Path, evaluation_id: str) -> Path:
    run_path(root, evaluation_id)
    return safe_path(root, f"artifacts/evaluations/{evaluation_id[7:]}/manifest.json")


def load_evaluation(root: Path, evaluation_id: str) -> EvaluationRun:
    try:
        evaluation = EvaluationRun.model_validate(read_json(evaluation_path(root, evaluation_id)))
    except (ValueError, OSError):
        raise PipelineError(
            "Evaluation is unavailable or invalid.", 7, "invalid_evaluation"
        ) from None
    if evaluation.evaluation_id != evaluation_id or evaluation.status != "completed":
        raise PipelineError("A completed evaluation is required.", 7, "invalid_evaluation")
    for relative, digest in evaluation.artifact_hashes.items():
        path = safe_path(root, relative)
        if not path.is_file() or hash_file(path) != digest:
            raise PipelineError(
                "Evaluation artifacts failed integrity verification.", 7, "evaluation_changed"
            )
    return evaluation


def evaluate_bundle(
    root: Path, bundle_id: str, dataset_version_id: str, split: str
) -> EvaluationRun:
    if split not in {"test_2025", "test_2026"}:
        raise PipelineError("Evaluation accepts a single declared test period.", 2, "usage_error")
    root = root.resolve()
    locks = root / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with FileLock(str(locks / "protocol.lock")):
        bundle = load_bundle(root, bundle_id)
        development = read_json(dataset_directory(root, dataset_version_id) / "development.json")
        if (
            development["development_digest"] != bundle.development_digest
            or development["profile"] != bundle.profile
        ):
            raise PipelineError(
                "Test revision changes frozen development or profile.", 5, "development_changed"
            )
        identifier = canonical_hash(
            {"bundle_id": bundle_id, "dataset_version_id": dataset_version_id, "split": split}
        )
        destination = evaluation_path(root, identifier)
        if destination.exists():
            load_dataset(root, dataset_version_id)
            return load_evaluation(root, identifier)
        if split == "test_2026":
            prior_main = [
                read_json(path) for path in (root / "artifacts/evaluations").glob("*/manifest.json")
            ]
            if not any(
                evaluation.get("bundle_id") == bundle_id
                and evaluation.get("split") == "test_2025"
                and evaluation.get("status") == "completed"
                for evaluation in prior_main
            ):
                raise PipelineError(
                    "The same bundle must first complete the main 2025 evaluation.",
                    5,
                    "main_evaluation_required",
                )
        before = hash_file(root / bundle.manifest_path)
        append_event(
            root / "artifacts/protocol/test-access.jsonl",
            {
                "event": "test_opened",
                "bundle_id": bundle_id,
                "dataset_version_id": dataset_version_id,
                "development_digest": bundle.development_digest,
                "split": split,
                "opened_at": datetime.now(UTC).isoformat(),
            },
        )
        version = load_dataset(root, dataset_version_id)
        if version.status != "sealed":
            raise PipelineError("Only a sealed dataset can be evaluated.")
        graph = read_json(safe_path(root, version.graph_manifest))
        if graph["node_order_hash"] != bundle.node_order_hash:
            raise PipelineError(
                "Graph node order changed after freezing.", 5, "node_order_mismatch"
            )
        for kind in ("nodes", "adjacency"):
            graph[kind + "_frame"] = read_parquet(safe_path(root, graph[kind]))
        features = read_parquet(safe_path(root, version.partition_manifests[split]["features"]))
        labels = read_parquet(safe_path(root, version.partition_manifests[split]["labels"]))
        require_same_population(features, labels)
        keys_by_split = {
            period: population_keys(
                read_parquet(safe_path(root, version.partition_manifests[period]["features"]))
            )
            for period in ("test_2025", "test_2026")
        }
        season_2025, season_2026 = seasonal_keys(
            keys_by_split["test_2025"], keys_by_split["test_2026"]
        )
        season = season_2025 if split == "test_2025" else season_2026
        full = population_keys(labels)
        cohorts = [cohort_manifest(full, "full_period"), cohort_manifest(season, "matched_season")]
        runs = [load_run(root, run_id) for run_id in bundle.selected_run_ids]
        if split == "test_2026" and not bundle.ablation_test_2026:
            runs = [run for run in runs if run.feature_set == "full"]
        tracker = RunTracker(root)
        prefix = destination.parent.relative_to(root).as_posix()
        predictions_paths: dict[str, str] = {}
        results: dict[str, dict[str, dict[str, Any]]] = {}
        inference_benchmarks: dict[str, dict[str, Any]] = {}
        with tracker.run(
            {
                "bundle_id": bundle_id,
                "dataset_version_id": dataset_version_id,
                "split": split,
                "profile": bundle.profile,
            }
        ) as tracking_id:
            with atomic_directory(root, destination.parent.relative_to(root)) as staging:
                write_parquet(staging / "population.parquet", full)
                write_parquet(staging / "matched-season.parquet", season)
                for run in runs:
                    print(
                        f"Evaluation: {split}, {run.model_family}/{run.feature_set}, seed {run.seed}.",
                        file=sys.stderr,
                        flush=True,
                    )
                    with MemorySampler() as memory:
                        prediction_started = time.perf_counter()
                        scores = predict_run(root, run, features, graph)
                        prediction_seconds = time.perf_counter() - prediction_started
                    inference_benchmarks[run.run_id] = {
                        "prediction_seconds": prediction_seconds,
                        "observations_per_second": len(features) / prediction_seconds,
                        "sampled_peak_rss_mib": memory.peak_bytes / 1024**2,
                        "scope": "verified_model_load_preprocessing_and_inference",
                    }
                    predictions = features[KEYS].copy()
                    predictions = predictions.merge(
                        labels[KEYS + ["label"]], on=KEYS, validate="one_to_one", sort=False
                    )
                    predictions["score"] = scores
                    predictions["predicted_label"] = (scores >= run.decision_threshold).astype(int)
                    predictions["split"] = split
                    predictions["run_id"] = run.run_id
                    predictions["evaluation_id"] = identifier
                    predictions["dataset_version_id"] = dataset_version_id
                    predictions["prediction_set_id"] = canonical_hash(
                        {"evaluation_id": identifier, "run_id": run.run_id}
                    )
                    require_same_population(labels, predictions)
                    relative = f"predictions/{run.run_id[7:]}.parquet"
                    write_parquet(staging / relative, predictions)
                    predictions_paths[run.run_id] = f"{prefix}/{relative}"
                    run_results = {
                        "full_period": classification_metrics(
                            predictions.label.to_numpy(), scores, run.decision_threshold
                        )
                    }
                    if not season.empty:
                        seasonal = predictions.merge(
                            season, on=KEYS, validate="one_to_one", sort=False
                        )
                        run_results["matched_season"] = classification_metrics(
                            seasonal.label.to_numpy(),
                            seasonal.score.to_numpy(),
                            run.decision_threshold,
                        )
                    results[run.run_id] = run_results
                aggregates: dict[str, Any] = {}
                for family, variant in sorted(
                    {(run.model_family, run.feature_set) for run in runs}
                ):
                    selected = [
                        run
                        for run in runs
                        if run.model_family == family and run.feature_set == variant
                    ]
                    aggregates[family + ":" + variant] = {
                        cohort: {
                            metric: aggregate_seeds(
                                {
                                    run.seed: results[run.run_id][cohort][metric]["value"]
                                    for run in selected
                                },
                                bundle.seeds,
                            )
                            for metric in METRICS
                        }
                        for cohort in results[selected[0].run_id]
                    }
                partition = version.partition_manifests[split]
                dataset_dir = dataset_directory(root, dataset_version_id)
                coverage = read_json(dataset_dir / "coverage.json")
                nominal_year = int(split[-4:])
                source_ids = {
                    entry["snapshot_id"]
                    for entry in coverage["sources"]
                    if entry["nominal_year"] == nominal_year
                }
                records = read_parquet(dataset_dir / "accidents/records.parquet")
                period_rows = records[records.source_snapshot_id.isin(source_ids)]
                cutoff = max(
                    interval["end"]
                    for entry in coverage["sources"]
                    if entry["nominal_year"] == nominal_year
                    for interval in entry["coverage_intervals"]
                )
                period = {
                    "split": split,
                    "coverage_basis": "owner_assumption"
                    if any(
                        entry["nominal_year"] == nominal_year
                        and (entry.get("evidence") or {}).get("kind") == "assumption"
                        for entry in coverage["sources"]
                    )
                    else "recorded_evidence",
                    "start": partition["start"],
                    "end_exclusive": partition["end_exclusive"],
                    "coverage_cutoff": cutoff,
                    "partial": cutoff < f"{nominal_year}-12-31",
                    "week_count": int(partition["week_count"]),
                    "segment_count": int(full.segment_id.nunique()),
                    "observation_count": len(full),
                    "event_count": int(labels.event_count.sum()),
                    "positive_count": int(labels.label.sum()),
                    "prevalence": float(labels.label.mean()),
                    "excluded_records": period_rows.target_exclusion_reason.value_counts().to_dict(),
                    "uncertainty": "single_seed_smoke"
                    if bundle.profile == "smoke_only"
                    else "five_seed_sample_standard_deviation",
                }
                after_bundle = load_bundle(root, bundle_id)
                load_dataset(root, dataset_version_id)
                if (
                    before != hash_file(root / bundle.manifest_path)
                    or after_bundle.protocol_hash != bundle.protocol_hash
                ):
                    raise PipelineError("Bundle changed during evaluation.", 5, "bundle_changed")
                write_json(
                    staging / "metrics.json",
                    {
                        "results": results,
                        "aggregates": aggregates,
                        "period": period,
                        "cohorts": cohorts,
                        "inference_benchmarks": inference_benchmarks,
                    },
                )
                hashes = {
                    f"{prefix}/{path.relative_to(staging).as_posix()}": hash_file(path)
                    for path in staging.rglob("*")
                    if path.is_file()
                }
                evaluation = EvaluationRun.model_validate(
                    {
                        "evaluation_id": identifier,
                        "profile": bundle.profile,
                        "mlflow_run_id": tracking_id,
                        "bundle_id": bundle_id,
                        "dataset_version_id": dataset_version_id,
                        "split": split,
                        "coverage_manifest_hash": hash_file(dataset_dir / "coverage.json"),
                        "population_hash": cohorts[0]["population_hash"],
                        "cohort_manifests": cohorts,
                        "model_run_ids": [run.run_id for run in runs],
                        "bundle_before_hash": bundle.protocol_hash,
                        "bundle_after_hash": after_bundle.protocol_hash,
                        "status": "completed",
                        "manifest_path": f"{prefix}/manifest.json",
                        "predictions": predictions_paths,
                        "results": results,
                        "aggregates": aggregates,
                        "period": period,
                        "inference_benchmarks": inference_benchmarks,
                        "artifact_hashes": hashes,
                    }
                )
                write_json(staging / "manifest.json", evaluation.model_dump(mode="json"))
            tracker.log_artifact(tracking_id, destination)
        return evaluation
