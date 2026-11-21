import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd

from road_accidents.config import PipelineError
from road_accidents.data.pipeline import (
    build_dataset,
    load_dataset,
    validate_dataset,
)
from road_accidents.data.schemas import DatasetConfig, SourceSnapshot
from road_accidents.data.storage import (
    atomic_directory,
    read_json,
    read_parquet,
    safe_path,
    write_json,
)
from road_accidents.evaluation.metrics import classification_metrics
from road_accidents.evaluation.runner import evaluate_bundle, load_evaluation
from road_accidents.experiments.schemas import EvaluationRun
from road_accidents.provenance import hash_file, runtime_environment
from road_accidents.reporting.packaging import verify_package
from road_accidents.reporting.tables import generate_report, verify_report


def compare_results(expected: EvaluationRun, actual: EvaluationRun, tolerance_pp: float) -> float:
    maximum = 0.0
    if (
        expected.split != actual.split
        or expected.population_hash != actual.population_hash
        or set(expected.results) != set(actual.results)
    ):
        raise PipelineError(
            "Reproduction changed the evaluation population.", 7, "reproduction_mismatch"
        )
    for run_id, cohorts in expected.results.items():
        if set(cohorts) != set(actual.results[run_id]):
            raise PipelineError(
                "Reproduction changed seasonal cohorts.", 7, "reproduction_mismatch"
            )
        for cohort, metrics in cohorts.items():
            for name, reference in metrics.items():
                repeated = actual.results[run_id][cohort][name]
                if (
                    reference["undefined_reason"] != repeated["undefined_reason"]
                    or reference["observation_count"] != repeated["observation_count"]
                    or reference["positive_count"] != repeated["positive_count"]
                ):
                    raise PipelineError(
                        "Reproduction changed metric null reasons or population counts.",
                        7,
                        "reproduction_mismatch",
                    )
                first, second = reference["value"], repeated["value"]
                if (first is None) != (second is None):
                    raise PipelineError(
                        "Reproduction changed metric definition.", 7, "reproduction_mismatch"
                    )
                if first is not None:
                    maximum = max(maximum, abs(first - second) * 100)
    if maximum > tolerance_pp:
        raise PipelineError(
            "Reproduced metrics exceed the approved tolerance.", 7, "reproduction_mismatch"
        )
    return maximum


def reports_only(manifest: Path, destination: Path, tolerance_pp: float) -> dict[str, Any]:
    content = read_json(manifest)
    relative = content.get("manifest")
    if not isinstance(relative, str):
        raise PipelineError("Reports mode requires a report manifest.", 2, "usage_error")
    root = manifest.resolve()
    for _part in Path(relative).parts:
        root = root.parent
    if safe_path(root, relative) != manifest.resolve():
        raise PipelineError(
            "Report location is inconsistent with its manifest.", 7, "invalid_report"
        )
    report = verify_report(root, relative)
    maximum = 0.0
    for identifier in report["evaluation_ids"]:
        expected = load_evaluation(root, identifier)
        results = {}
        season = read_parquet(root / Path(expected.manifest_path).parent / "matched-season.parquet")
        for run_id, path in expected.predictions.items():
            predictions = read_parquet(safe_path(root, path))
            threshold = expected.results[run_id]["full_period"]["f1"]["decision_threshold"]
            cohorts = {"full_period": predictions}
            if not season.empty:
                cohorts["matched_season"] = predictions.merge(
                    season, on=["week_start", "segment_id"], validate="one_to_one"
                )
            results[run_id] = {
                name: classification_metrics(
                    frame.label.to_numpy(), frame.score.to_numpy(), threshold
                )
                for name, frame in cohorts.items()
            }
        actual = expected.model_copy(update={"results": results})
        maximum = max(maximum, compare_results(expected, actual, tolerance_pp))
    with atomic_directory(destination.parent, destination.name) as staging:
        presentation = staging / "report"
        presentation.mkdir()
        for path in report["artifact_hashes"]:
            shutil.copyfile(safe_path(root, path), presentation / Path(path).name)
        result = {
            "status": "succeeded",
            "mode": "reports",
            "max_difference_pp": maximum,
            "undefined_reasons_preserved": True,
            "reconstructed_dataset": False,
            "inference_replayed": False,
            "training_calls": 0,
            "source_manifest_sha256": hash_file(manifest),
        }
        write_json(staging / "reproduction.json", result)
    return result


def reproduce(
    manifest: Path, *, mode: str = "full", tolerance_pp: float = 1.0, output: Path | None = None
) -> dict[str, Any]:
    if mode not in {"full", "reports"} or not 0 <= tolerance_pp <= 1.0:
        raise PipelineError(
            "Reproduction requires full/reports and tolerance from 0 to 1 percentage point.",
            2,
            "usage_error",
        )
    manifest = manifest.resolve()
    content = read_json(manifest)
    if mode == "full" and "package_id" not in content:
        raise PipelineError(
            "Full reproduction requires a package manifest, not a report.", 2, "usage_error"
        )
    package = verify_package(manifest) if mode == "full" else None
    if package and read_json(manifest.parent / "runtime-environment.json") != runtime_environment():
        raise PipelineError(
            "Reproduction requires the recorded Python and installed package versions.",
            7,
            "environment_mismatch",
        )
    identifier = content.get("package_id", content.get("report_id", "report"))
    destination = (
        output
        or manifest.parent.parent / ("reproduction-" + identifier.replace("sha256:", "")[:16])
    ).resolve()
    if destination == manifest.parent or destination.is_relative_to(manifest.parent):
        raise PipelineError(
            "Reproduction output must not modify its input package.", 2, "usage_error"
        )
    if destination.exists():
        previous = read_json(destination / "reproduction.json")
        if (
            previous.get("source_manifest_sha256") == hash_file(manifest)
            and previous.get("mode") == mode
            and previous.get("max_difference_pp", 100) <= tolerance_pp
        ):
            return dict(previous)
        raise PipelineError(
            "Reproduction destination already contains different results.", 7, "output_conflict"
        )
    if mode == "reports":
        return reports_only(manifest, destination, tolerance_pp)
    assert package is not None
    with tempfile.TemporaryDirectory(prefix="road-risk-replay-") as temporary:
        isolated = Path(temporary)
        for component in package.inventory:
            target = safe_path(isolated, component.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(manifest.parent, component.path), target)
        shutil.copyfile(manifest, isolated / "package-manifest.json")
        verify_package(isolated / "package-manifest.json")
        write_json(
            isolated / "code/.road-risk-code.json", read_json(isolated / "code-snapshot.json")
        )
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=str(isolated / "code/src"),
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING="false",
            MLFLOW_DISABLE_AGENT_HINT="1",
            DO_NOT_TRACK="1",
            DVC_NO_ANALYTICS="1",
        )
        with atomic_directory(destination.parent, destination.name) as staging:
            command = [
                sys.executable,
                "-B",
                "-m",
                "road_accidents.reporting.reproduce",
                "--worker",
                str(isolated / "package-manifest.json"),
                "--output",
                str(staging),
                "--tolerance-pp",
                str(tolerance_pp),
            ]
            try:
                execution = subprocess.run(
                    command,
                    cwd=isolated / "code",
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=7200 if package.profile == "confirmatory" else 900,
                )
            except (OSError, subprocess.TimeoutExpired):
                raise PipelineError(
                    "Isolated reproduction did not complete.", 7, "reproduction_failed"
                ) from None
            if execution.returncode:
                failure = (
                    read_json(staging / "failure.json")
                    if (staging / "failure.json").is_file()
                    else {"message": "Isolated reproduction process failed."}
                )
                raise PipelineError(failure["message"], 7, "reproduction_failed")
            result: dict[str, Any] = read_json(staging / "reproduction.json")
            result["source_manifest_sha256"] = hash_file(manifest)
            if (
                result.get("status") != "succeeded"
                or not result.get("reconstructed_dataset")
                or not result.get("inference_replayed")
            ):
                raise PipelineError(
                    "Isolated reproduction returned incomplete evidence.", 7, "reproduction_failed"
                )
    return result


def replay_worker(manifest: Path, output: Path, tolerance_pp: float) -> None:
    from road_accidents.data.coverage import CoverageLedger
    from road_accidents.data.osm import ingest_osm
    from road_accidents.data.prf import ingest_prf
    from road_accidents.features.preprocessing import SharedPreprocessor

    package = verify_package(manifest)
    isolated = manifest.parent
    if not Path(__file__).resolve().is_relative_to((isolated / "code").resolve()):
        raise PipelineError(
            "Reproduction did not load the archived source code.", 7, "code_mismatch"
        )
    if read_json(isolated / "runtime-environment.json") != runtime_environment():
        raise PipelineError("Reproduction environment changed.", 7, "environment_mismatch")
    replay = read_json(isolated / "replay.json")
    original = isolated / "payload"
    rebuilt = isolated / "rebuilt"
    rebuilt.mkdir()
    for component in package.inventory:
        if not component.path.startswith("payload/"):
            continue
        relative = component.path[len("payload/") :]
        if relative.startswith("data/processed/") or component.component_type in {
            "dataset",
            "evaluation",
            "report",
        }:
            continue
        target = safe_path(rebuilt, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_path(isolated, component.path), target)
    configuration = DatasetConfig.model_validate(replay["configuration"])
    snapshots = [SourceSnapshot.model_validate(source) for source in replay["sources"]]
    prf: list[SourceSnapshot] = []
    history: list[SourceSnapshot] = []
    osm = None
    expected_version = load_dataset(original, package.dataset_version_id)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise PipelineError(
            "Training and network access are forbidden during reproduction.",
            7,
            "forbidden_reproduction_action",
        )

    with (
        patch.object(SharedPreprocessor, "fit", forbidden),
        patch("socket.socket.connect", forbidden),
        patch("socket.create_connection", forbidden),
    ):
        for source in snapshots:
            payload = safe_path(original, source.raw_path).read_bytes()
            output_prefix = Path(source.raw_path).parent.parent.as_posix()
            if source.source_type == "prf_occurrence_csv":
                if source.nominal_year is None:
                    raise PipelineError("Package PRF year is missing.")
                approved = ingest_prf(
                    payload,
                    rebuilt,
                    year=source.nominal_year,
                    source_url=source.source_url,
                    profile=source.profile,
                    output=output_prefix,
                    max_buffer_bytes=int(source.metadata.get("max_buffer_bytes", 268435456)),
                )
                (
                    history if source.snapshot_id in expected_version.history_snapshot_ids else prf
                ).append(approved)
            else:
                osm = ingest_osm(
                    payload,
                    rebuilt,
                    effective_at=configuration.spatial.osm_effective_at.isoformat(),
                    source_url=source.source_url,
                    profile=source.profile,
                    output=output_prefix,
                    max_buffer_bytes=int(source.metadata.get("max_buffer_bytes", 268435456)),
                )
        if osm is None:
            raise PipelineError("Package OSM source is missing.")
        coverage = CoverageLedger.from_documents(
            replay["coverage"]["sources"], profile=package.profile, root=rebuilt
        )
        reconstructed = build_dataset(rebuilt, configuration, prf, osm, coverage, history)
        validate_dataset(rebuilt, reconstructed.dataset_version_id)
        if reconstructed.development_digest != expected_version.development_digest:
            raise PipelineError(
                "Reconstructed development digest differs.", 7, "reconstruction_mismatch"
            )
        for split in expected_version.partition_manifests:
            for kind in ("features", "labels"):
                expected = read_parquet(
                    safe_path(original, expected_version.partition_manifests[split][kind])
                )
                actual = read_parquet(
                    safe_path(rebuilt, reconstructed.partition_manifests[split][kind])
                )
                try:
                    pd.testing.assert_frame_equal(
                        expected, actual, check_dtype=False, check_exact=True
                    )
                except AssertionError:
                    raise PipelineError(
                        "Reconstructed features or labels differ from the recorded dataset.",
                        7,
                        "reconstruction_mismatch",
                    ) from None
        repeated = [
            evaluate_bundle(rebuilt, package.bundle_id, reconstructed.dataset_version_id, split)
            for split in ("test_2025", "test_2026")
        ]
        maximum = max(
            compare_results(load_evaluation(original, identifier), actual, tolerance_pp)
            for identifier, actual in zip(package.evaluation_ids, repeated, strict=True)
        )
        report = generate_report(
            rebuilt,
            repeated[0].evaluation_id,
            repeated[1].evaluation_id,
            "artifacts/reports/reproduced",
        )
        shutil.copytree(safe_path(rebuilt, Path(report["manifest"]).parent), output / "report")
        write_json(
            output / "reproduction.json",
            {
                "status": "succeeded",
                "mode": "full",
                "package_id": package.package_id,
                "source_manifest_sha256": hash_file(manifest),
                "max_difference_pp": maximum,
                "undefined_reasons_preserved": True,
                "reconstructed_dataset": True,
                "reconstructed_dataset_version_id": reconstructed.dataset_version_id,
                "inference_replayed": True,
                "training_calls": 0,
                "network_access": "blocked",
                "code_hash": replay["code_hash"],
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-pp", type=float, default=1.0)
    arguments = parser.parse_args()
    try:
        replay_worker(arguments.worker, arguments.output, arguments.tolerance_pp)
    except PipelineError as failure:
        write_json(
            arguments.output / "failure.json", {"message": str(failure), "code": failure.code}
        )
        raise SystemExit(7) from None
