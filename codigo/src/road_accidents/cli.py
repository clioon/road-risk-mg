import json
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from filelock import FileLock
from typer._click.exceptions import ClickException

from road_accidents import CONTRACT_VERSION
from road_accidents.config import PipelineError, load_config


class OutputFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


@dataclass(frozen=True)
class CommandContext:
    work_dir: Path
    dry_run: bool = False
    log_format: OutputFormat = OutputFormat.TEXT


@dataclass
class OperationResult:
    command: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
sources_app = typer.Typer(add_completion=False)
dataset_app = typer.Typer(add_completion=False)
experiment_app = typer.Typer(add_completion=False)
pipeline_app = typer.Typer(add_completion=False)
app.add_typer(sources_app, name="sources")
app.add_typer(dataset_app, name="dataset")
app.add_typer(experiment_app, name="experiment")
app.add_typer(pipeline_app, name="pipeline")


@app.callback(invoke_without_command=True)
def configure(
    context: typer.Context,
    work_dir: Annotated[Path, typer.Option("--work-dir")] = Path("."),
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    log_format: Annotated[OutputFormat, typer.Option("--log-format")] = OutputFormat.TEXT,
    result_format: Annotated[OutputFormat, typer.Option("--result-format")] = OutputFormat.TEXT,
) -> None:
    context.obj = CommandContext(work_dir.resolve(), dry_run, log_format)
    if context.invoked_subcommand is None:
        typer.echo(context.get_help())


def input_path(settings: CommandContext, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (settings.work_dir / path).resolve()


def source_result(command: str, settings: CommandContext, snapshots: list[Any]) -> OperationResult:
    from road_accidents.data.storage import read_json, safe_path, write_json
    from road_accidents.provenance import canonical_hash

    if settings.dry_run:
        return OperationResult(
            command,
            outputs={
                "dry_run": True,
                "approved_snapshot_ids": [source.snapshot_id for source in snapshots],
            },
        )
    catalog = {
        "schema_version": CONTRACT_VERSION,
        "sources": [source.model_dump(mode="json") for source in snapshots],
    }
    identifier = canonical_hash([source.source_manifest_id for source in snapshots])
    relative = f"data/manifests/catalogs/{identifier[7:]}.json"
    destination = safe_path(settings.work_dir, relative)
    locks = settings.work_dir / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with FileLock(str(locks / "source-catalog.lock")):
        if destination.exists():
            if read_json(destination) != catalog:
                raise PipelineError("Selected source catalog failed integrity verification.")
        else:
            write_json(destination, catalog)
    if (settings.work_dir / ".dvc").is_dir():
        with redirect_stdout(sys.stderr):
            from dvc.repo import Repo

            paths = {relative}
            for snapshot in snapshots:
                paths.update((snapshot.raw_path, snapshot.manifest_path))
            with Repo(str(settings.work_dir)) as repository:
                repository.add([str(safe_path(settings.work_dir, path)) for path in sorted(paths)])
    return OperationResult(
        command,
        outputs={
            "source_manifest": relative,
            "snapshot_ids": [source.snapshot_id for source in snapshots],
        },
    )


@sources_app.command("import-prf")
def import_prf_command(
    context: typer.Context,
    file: Annotated[Path, typer.Option("--file")],
    year: Annotated[int, typer.Option("--year", min=2019, max=2026)],
    source_url: Annotated[str, typer.Option("--source-url")],
    output: Annotated[str, typer.Option("--output")] = "data/raw/prf",
    max_buffer_bytes: Annotated[int, typer.Option("--max-buffer-bytes", min=1)] = 268435456,
) -> OperationResult:
    from road_accidents.data.prf import import_prf

    settings: CommandContext = context.obj
    snapshot = import_prf(
        input_path(settings, file),
        settings.work_dir,
        year=year,
        source_url=source_url,
        output=output,
        max_buffer_bytes=max_buffer_bytes,
        dry_run=settings.dry_run,
    )
    return source_result("sources import-prf", settings, [snapshot])


@sources_app.command("fetch-prf")
def fetch_prf_command(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config")],
    years: Annotated[str, typer.Option("--years")] = "2019:2026",
    output: Annotated[str, typer.Option("--output")] = "data/raw/prf",
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
) -> OperationResult:
    from road_accidents.data.prf import fetch_prf_source, import_prf
    from road_accidents.data.schemas import SourcesConfig, SourceSnapshot
    from road_accidents.data.storage import read_json, safe_path

    settings: CommandContext = context.obj
    configuration = load_config(input_path(settings, config), SourcesConfig)
    if years != "2019:2026":
        raise PipelineError("The study requires years 2019:2026.", 2, "usage_error")
    safe_path(settings.work_dir, output)
    snapshots = []
    for year, entry in sorted(configuration.years.items()):
        limit = configuration.ingestion.max_buffer_bytes
        if entry.local_file:
            snapshot = import_prf(
                input_path(settings, Path(entry.local_file)),
                settings.work_dir,
                source_url=entry.url,
                year=year,
                output=output,
                max_buffer_bytes=limit,
                profile=configuration.profile,
                dry_run=settings.dry_run,
            )
        elif entry.selected_manifest and not refresh:
            previous = SourceSnapshot.model_validate(
                read_json(input_path(settings, Path(entry.selected_manifest)))
            )
            if previous.profile != configuration.profile:
                raise PipelineError("Selected source profile cannot be changed.")
            snapshot = import_prf(
                safe_path(settings.work_dir, previous.raw_path),
                settings.work_dir,
                source_url=previous.source_url,
                year=year,
                output=output,
                max_buffer_bytes=limit,
                profile=configuration.profile,
                dry_run=settings.dry_run,
            )
        elif settings.dry_run:
            continue
        else:
            snapshot = fetch_prf_source(
                entry.url,
                settings.work_dir,
                catalog=entry.catalog,
                year=year,
                output=output,
                max_buffer_bytes=limit,
                profile=configuration.profile,
                dry_run=settings.dry_run,
            )
        snapshots.append(snapshot)
    result = source_result("sources fetch-prf", settings, snapshots)
    if settings.dry_run:
        result.outputs["planned_years"] = list(configuration.years)
    return result


@sources_app.command("fetch-osm")
def fetch_osm_command(
    context: typer.Context,
    state: Annotated[str, typer.Option("--state")] = "MG",
    effective_at: Annotated[str, typer.Option("--effective-at")] = "2018-12-31T23:59:59Z",
    output: Annotated[str, typer.Option("--output")] = "data/raw/osm",
    max_buffer_bytes: Annotated[int, typer.Option("--max-buffer-bytes", min=1)] = 268435456,
    endpoint: Annotated[
        str, typer.Option("--endpoint")
    ] = "https://overpass-api.de/api/interpreter",
    tiles: Annotated[bool, typer.Option("--tiles")] = False,
) -> OperationResult:
    from road_accidents.data.osm import fetch_osm
    from road_accidents.data.schemas import OSM_EFFECTIVE_AT
    from road_accidents.data.storage import safe_path

    settings: CommandContext = context.obj
    if (
        state != "MG"
        or datetime.fromisoformat(effective_at.replace("Z", "+00:00")) != OSM_EFFECTIVE_AT
    ):
        raise PipelineError("MG and the frozen historical snapshot are required.", 2, "usage_error")
    safe_path(settings.work_dir, output)
    if settings.dry_run:
        return OperationResult(
            "sources fetch-osm",
            outputs={"dry_run": True, "state": state, "effective_at": effective_at},
        )
    from road_accidents.data.osm_acquisition import fetch_osm_tiled

    acquire = fetch_osm_tiled if tiles else fetch_osm
    snapshot = acquire(
        settings.work_dir,
        effective_at=effective_at,
        output=output,
        max_buffer_bytes=max_buffer_bytes,
        endpoint=endpoint,
    )
    return source_result("sources fetch-osm", settings, [snapshot])


@dataset_app.command("build")
def build_dataset_command(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config")],
    prf_manifest: Annotated[Path, typer.Option("--prf-manifest")],
    osm_manifest: Annotated[Path, typer.Option("--osm-manifest")],
    coverage_manifest: Annotated[Path, typer.Option("--coverage-manifest")],
) -> OperationResult:
    from road_accidents.data.coverage import CoverageLedger
    from road_accidents.data.pipeline import build_dataset
    from road_accidents.data.schemas import (
        CoverageConfig,
        DatasetConfig,
        SourceCatalog,
        SourceSnapshot,
    )
    from road_accidents.data.storage import read_json, safe_path
    from road_accidents.provenance import hash_file

    settings: CommandContext = context.obj
    configuration = load_config(input_path(settings, config), DatasetConfig)
    coverage_config = load_config(input_path(settings, coverage_manifest), CoverageConfig)
    prf = SourceCatalog.model_validate(read_json(input_path(settings, prf_manifest)))
    osm = SourceCatalog.model_validate(read_json(input_path(settings, osm_manifest)))
    if len(osm.sources) != 1 or len(prf.sources) != 8:
        raise PipelineError("One OSM and eight PRF sources are required.")
    coverage = CoverageLedger.from_documents(
        [entry.model_dump(mode="json") for entry in coverage_config.sources],
        profile=coverage_config.profile,
        root=settings.work_dir,
    )
    history = [
        SourceSnapshot.model_validate(read_json(input_path(settings, Path(path))))
        for path in coverage_config.history_sources
    ]
    if settings.dry_run:
        coverage.select([source.snapshot_id for source in prf.sources]).require_study_coverage()
        for source in prf.sources + osm.sources + history:
            if hash_file(safe_path(settings.work_dir, source.raw_path)) != source.sha256:
                raise PipelineError("Source checksum failed.")
        return OperationResult(
            "dataset build", outputs={"dry_run": True, "profile": configuration.profile}
        )
    version = build_dataset(
        settings.work_dir, configuration, prf.sources, osm.sources[0], coverage, history
    )
    return OperationResult(
        "dataset build",
        outputs={
            "dataset_version_id": version.dataset_version_id,
            "development_digest": version.development_digest,
            "validation_report": version.validation_report_uri,
            "graph_manifest": version.graph_manifest,
        },
    )


@dataset_app.command("validate")
def validate_dataset_command(
    context: typer.Context,
    dataset_version: Annotated[str, typer.Option("--dataset-version")],
) -> OperationResult:
    from road_accidents.data.pipeline import validate_dataset

    settings: CommandContext = context.obj
    version = validate_dataset(
        settings.work_dir, dataset_version, publish_seal=not settings.dry_run
    )
    return OperationResult(
        "dataset validate",
        inputs={"dataset_version_id": dataset_version},
        outputs={
            "dataset_version_id": version.dataset_version_id,
            "development_digest": version.development_digest,
            "validation_report": version.validation_report_uri,
            "dataset_status": version.status,
            "dry_run": settings.dry_run,
        },
    )


@app.command("train")
def train_command(
    context: typer.Context,
    dataset_version: Annotated[str, typer.Option("--dataset-version")],
    model: Annotated[str, typer.Option("--model")],
    model_config: Annotated[Path, typer.Option("--model-config")],
    feature_set: Annotated[str, typer.Option("--feature-set")] = "full",
    seed: Annotated[int, typer.Option("--seed")] = 101,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> OperationResult:
    from road_accidents.experiments.schemas import VARIANTS, ModelConfig
    from road_accidents.experiments.training import (
        assert_training_allowed,
        load_development,
        train_run,
    )

    settings: CommandContext = context.obj
    configuration = load_config(input_path(settings, model_config), ModelConfig)
    if configuration.model_family != model or feature_set not in VARIANTS:
        raise PipelineError(
            "Model or feature set does not match its configuration.", 2, "usage_error"
        )
    if settings.dry_run:
        view = load_development(settings.work_dir, dataset_version)
        assert_training_allowed(settings.work_dir, view["development_digest"])
        return OperationResult("train", outputs={"dry_run": True, "model": model, "seed": seed})
    run = train_run(
        settings.work_dir,
        dataset_version,
        configuration,
        seed=seed,
        feature_set=feature_set,
        force=force,
    )
    return OperationResult(
        "train",
        inputs={"dataset_version_id": dataset_version},
        outputs={"run_id": run.run_id, "validation_predictions": run.validation_predictions},
    )


@experiment_app.command("run")
def experiment_run_command(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config")],
    dataset_version: Annotated[str | None, typer.Option("--dataset-version")] = None,
) -> OperationResult:
    from road_accidents.experiments.schemas import ExperimentConfig, ModelConfig
    from road_accidents.experiments.training import (
        assert_training_allowed,
        load_development,
        run_experiment,
    )

    settings: CommandContext = context.obj
    configuration = load_config(input_path(settings, config), ExperimentConfig)
    version = dataset_version or configuration.dataset_version_id
    if version is None or (
        dataset_version
        and configuration.dataset_version_id
        and dataset_version != configuration.dataset_version_id
    ):
        raise PipelineError("One consistent dataset version is required.", 2, "usage_error")
    if settings.dry_run:
        view = load_development(settings.work_dir, version)
        assert_training_allowed(settings.work_dir, view["development_digest"])
        for path in configuration.models.values():
            load_config(input_path(settings, Path(path)), ModelConfig)
        return OperationResult(
            "experiment run", outputs={"dry_run": True, "profile": configuration.profile}
        )
    group = run_experiment(settings.work_dir, version, configuration)
    return OperationResult(
        "experiment run",
        outputs={"parent_run_id": group["parent_run_id"], "run_ids": group["selected_run_ids"]},
    )


@experiment_app.command("freeze")
def experiment_freeze_command(
    context: typer.Context,
    parent_run: Annotated[str, typer.Option("--parent-run")],
) -> OperationResult:
    from road_accidents.data.storage import read_json
    from road_accidents.experiments.freeze import freeze_experiment, validate_selection
    from road_accidents.experiments.training import assert_training_allowed, group_path, load_run

    settings: CommandContext = context.obj
    if settings.dry_run:
        group = read_json(group_path(settings.work_dir, parent_run) / "manifest.json")
        assert_training_allowed(settings.work_dir, group["development_digest"])
        runs = [
            load_run(settings.work_dir, identifier).model_dump(mode="json")
            for identifier in group["selected_run_ids"]
        ]
        validate_selection(runs, group["profile"], group["seeds"])
        return OperationResult("experiment freeze", outputs={"dry_run": True})
    bundle = freeze_experiment(settings.work_dir, parent_run)
    return OperationResult(
        "experiment freeze",
        outputs={
            "bundle_id": bundle.bundle_id,
            "protocol_hash": bundle.protocol_hash,
            "manifest": bundle.manifest_path,
        },
    )


@app.command("evaluate")
def evaluate_command(
    context: typer.Context,
    bundle: Annotated[str, typer.Option("--bundle")],
    dataset_version: Annotated[str, typer.Option("--dataset-version")],
    split: Annotated[str, typer.Option("--split")],
) -> OperationResult:
    from road_accidents.evaluation.runner import evaluate_bundle
    from road_accidents.experiments.freeze import load_bundle

    settings: CommandContext = context.obj
    if split not in {"test_2025", "test_2026"}:
        raise PipelineError("Evaluation requires test_2025 or test_2026.", 2, "usage_error")
    if settings.dry_run:
        load_bundle(settings.work_dir, bundle)
        return OperationResult(
            "evaluate",
            inputs={"dataset_version_id": dataset_version},
            outputs={"dry_run": True, "split": split},
        )
    evaluation = evaluate_bundle(settings.work_dir, bundle, dataset_version, split)
    directory = str(Path(evaluation.manifest_path).parent).replace("\\", "/")
    return OperationResult(
        "evaluate",
        outputs={
            "evaluation_id": evaluation.evaluation_id,
            "bundle_id": bundle,
            "dataset_version_id": dataset_version,
            "split": split,
            "population_manifest": directory + "/population.parquet",
            "metrics": directory + "/metrics.json",
        },
    )


@app.command("report")
def report_command(
    context: typer.Context,
    main_evaluation: Annotated[str, typer.Option("--main-evaluation")],
    temporal_evaluation: Annotated[str, typer.Option("--temporal-evaluation")],
    output: Annotated[str, typer.Option("--output")],
    package_output: Annotated[str | None, typer.Option("--package-output")] = None,
) -> OperationResult:
    from road_accidents.data.storage import read_json, safe_path
    from road_accidents.evaluation.runner import load_evaluation
    from road_accidents.reporting.packaging import export_package
    from road_accidents.reporting.tables import generate_report

    settings: CommandContext = context.obj
    if settings.dry_run:
        first, second = (
            load_evaluation(settings.work_dir, main_evaluation),
            load_evaluation(settings.work_dir, temporal_evaluation),
        )
        if (
            first.split != "test_2025"
            or second.split != "test_2026"
            or (first.bundle_id, first.dataset_version_id)
            != (second.bundle_id, second.dataset_version_id)
        ):
            raise PipelineError(
                "Report evaluations are incompatible.", 7, "incompatible_evaluations"
            )
        safe_path(settings.work_dir, output)
        if package_output:
            safe_path(settings.work_dir, package_output)
        return OperationResult("report", outputs={"dry_run": True})
    report = generate_report(settings.work_dir, main_evaluation, temporal_evaluation, output)
    result = OperationResult(
        "report",
        outputs={
            "manifest": report["manifest"],
            "tables": report["tables"],
            "figures": report["figures"],
        },
    )
    if package_output:
        manifest = export_package(settings.work_dir, report["manifest"], package_output)
        result.outputs.update(
            package_id=read_json(manifest)["package_id"],
            package_manifest=manifest.relative_to(settings.work_dir).as_posix(),
        )
    return result


@app.command("reproduce")
def reproduce_command(
    context: typer.Context,
    manifest: Annotated[Path, typer.Option("--manifest")],
    mode: Annotated[str, typer.Option("--mode")] = "full",
    tolerance_pp: Annotated[float, typer.Option("--tolerance-pp", min=0, max=1)] = 1.0,
) -> OperationResult:
    from road_accidents.data.storage import read_json, safe_path
    from road_accidents.provenance import hash_file
    from road_accidents.reporting.packaging import verify_package
    from road_accidents.reporting.reproduce import reproduce

    settings: CommandContext = context.obj
    source = input_path(settings, manifest)
    if mode not in {"full", "reports"}:
        raise PipelineError("Reproduction mode must be full or reports.", 2, "usage_error")
    if settings.dry_run:
        if mode == "full":
            if "package_id" not in read_json(source):
                raise PipelineError("Full mode requires a package manifest.", 2, "usage_error")
            verify_package(source)
        return OperationResult("reproduce", outputs={"dry_run": True, "mode": mode})
    relative = f"artifacts/reproductions/{hash_file(source)}-{mode}"
    result = reproduce(
        source, mode=mode, tolerance_pp=tolerance_pp, output=safe_path(settings.work_dir, relative)
    )
    return OperationResult(
        "reproduce", outputs={"reproduction_report": relative + "/reproduction.json", **result}
    )


@pipeline_app.command("smoke")
def smoke_command(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config")],
) -> OperationResult:
    from road_accidents.pipeline import SmokeConfig, run_smoke

    settings: CommandContext = context.obj
    configuration = load_config(input_path(settings, config), SmokeConfig)
    if settings.dry_run:
        return OperationResult("pipeline smoke", outputs={"dry_run": True, "profile": "smoke_only"})
    return OperationResult("pipeline smoke", outputs=run_smoke(settings.work_dir, configuration))


def main(args: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if args is None else args
    json_output = any(
        argument == "--result-format=json"
        or (argument == "--result-format" and arguments[index + 1 : index + 2] == ["json"])
        for index, argument in enumerate(arguments)
    )
    started = datetime.now(UTC).isoformat()
    result = OperationResult("")
    failure: PipelineError | None = None
    try:
        returned = app(args=arguments, standalone_mode=False)
        if not isinstance(returned, OperationResult):
            return 0
        result = returned
    except typer.Exit as error:
        return error.exit_code
    except ClickException:
        failure = PipelineError("Invalid command or configuration.", 2, "usage_error")
    except PipelineError as error:
        failure = error
    except (OSError, ValueError, RuntimeError):
        failure = PipelineError(
            "Operation failed; no successful output was published.", 6, "operation_failed"
        )
    envelope = {
        "schema_version": CONTRACT_VERSION,
        "command": result.command,
        "status": "failed" if failure else "succeeded",
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "inputs": result.inputs,
        "outputs": {} if failure else result.outputs,
        "warnings": result.warnings,
        "error": {"code": failure.code, "message": str(failure)} if failure else None,
    }
    if json_output:
        typer.echo(json.dumps(envelope, allow_nan=False))
    elif failure:
        typer.echo(str(failure), err=True)
    else:
        typer.echo(json.dumps(result.outputs, allow_nan=False, indent=2))
    return failure.exit_code if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
