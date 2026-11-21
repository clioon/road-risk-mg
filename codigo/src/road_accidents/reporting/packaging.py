import shutil
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from road_accidents.config import PipelineError
from road_accidents.data.pipeline import dataset_directory, load_dataset
from road_accidents.data.schemas import DatasetConfig, SourceSnapshot
from road_accidents.data.storage import atomic_directory, read_json, safe_path, write_json
from road_accidents.evaluation.runner import load_evaluation
from road_accidents.experiments.freeze import load_bundle
from road_accidents.experiments.schemas import PackageComponent, ReproductionPackage
from road_accidents.experiments.training import group_path, load_run
from road_accidents.provenance import RunTracker, canonical_hash, hash_file
from road_accidents.reporting.tables import verify_report

PACKAGE_ID_EXCLUSIONS = frozenset({"package_id", "created_at", "status"})


def package_failure(
    message: str = "Reproduction package is incomplete or corrupted.",
) -> PipelineError:
    return PipelineError(message, 7, "invalid_package")


def verify_package(manifest: Path) -> ReproductionPackage:
    try:
        package = ReproductionPackage.model_validate(read_json(manifest))
        if (
            package.status != "sealed"
            or canonical_hash(package.model_dump(mode="json"), exclude_keys=PACKAGE_ID_EXCLUSIONS)
            != package.package_id
        ):
            raise package_failure()
        directory = manifest.parent.resolve()
        known: dict[str, PackageComponent] = {}
        for component in package.inventory:
            portable = PurePosixPath(component.path)
            if (
                portable.is_absolute()
                or PureWindowsPath(component.path).drive
                or ".." in portable.parts
                or "\\" in component.path
            ):
                raise package_failure("Package paths must be contained, portable relative paths.")
            if component.path in known:
                raise package_failure("Package inventory contains duplicate paths.")
            known[component.path] = component
            path = safe_path(directory, component.path)
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != component.byte_count
                or hash_file(path) != component.sha256
            ):
                raise package_failure()
        required = {
            "source",
            "evidence",
            "dataset",
            "code",
            "environment",
            "model",
            "preprocessor",
            "bundle",
            "evaluation",
            "report",
            "run_metadata",
            "replay",
        }
        if not required <= {component.component_type for component in package.inventory}:
            raise package_failure("Package inventory is missing a required component category.")
        if (
            package.report_manifest not in known
            or known[package.report_manifest].sha256 != package.report_manifest_hash
        ):
            raise package_failure("Package report identity is invalid.")
        replay = read_json(safe_path(directory, "replay.json"))
        report = read_json(safe_path(directory, package.report_manifest))
        if (
            report["dataset_version_id"] != package.dataset_version_id
            or report["bundle_id"] != package.bundle_id
            or report["evaluation_ids"] != package.evaluation_ids
        ):
            raise package_failure("Package references do not match the completed report.")
        required_payload = set(report["artifact_hashes"])
        bundle_path = f"payload/artifacts/bundles/{package.bundle_id[7:]}/manifest.json"
        if bundle_path not in known:
            raise package_failure()
        bundle = read_json(safe_path(directory, bundle_path))
        required_payload.update(bundle["artifact_hashes"])
        data_path = f"payload/{report['dataset_manifest']}"
        if data_path not in known:
            raise package_failure()
        dataset = read_json(safe_path(directory, data_path))
        required_payload.update(dataset["inventory"])
        for identifier in package.evaluation_ids:
            evaluation_path = f"payload/artifacts/evaluations/{identifier[7:]}/manifest.json"
            if evaluation_path not in known:
                raise package_failure()
            required_payload.update(
                read_json(safe_path(directory, evaluation_path))["artifact_hashes"]
            )
        for path in required_payload:
            if "payload/" + path not in known:
                raise package_failure("Package inventory is not closed over artifact references.")
        for source in replay["sources"]:
            if (
                "payload/" + source["raw_path"] not in known
                or "payload/" + source["manifest_path"] not in known
            ):
                raise package_failure("Source snapshots are missing from the package.")
        for entry in bundle["code_inventory"]:
            if (
                "code/" + entry["path"] not in known
                or known["code/" + entry["path"]].sha256 != entry["sha256"]
            ):
                raise package_failure("Exact frozen code is missing from the package.")
        if "payload/" + replay["configuration"]["boundary_path"] not in known:
            raise package_failure("Reconstruction boundary is missing.")
        return package
    except PipelineError as error:
        if error.exit_code == 7:
            raise
        raise package_failure() from None
    except (OSError, ValueError, KeyError, TypeError):
        raise package_failure() from None


def export_package(root: Path, report_manifest: str, output: str) -> Path:
    root = root.resolve()
    report = verify_report(root, report_manifest)
    bundle = load_bundle(root, report["bundle_id"])
    version = load_dataset(root, report["dataset_version_id"])
    evaluations = [load_evaluation(root, identifier) for identifier in report["evaluation_ids"]]
    destination = safe_path(root, output)
    manifest = destination / "package-manifest.json"
    if destination.exists():
        package = verify_package(manifest)
        if package.report_manifest_hash != hash_file(safe_path(root, report_manifest)):
            raise package_failure("Package destination already belongs to a different report.")
        return manifest
    dataset_dir = dataset_directory(root, version.dataset_version_id)
    sources = [
        SourceSnapshot.model_validate(source)
        for source in read_json(dataset_dir / "sources.json")["sources"]
    ]
    config = DatasetConfig.model_validate(read_json(dataset_dir / "configuration.json"))
    coverage = read_json(dataset_dir / "coverage.json")
    files: dict[str, tuple[Path, str, str]] = {}

    def include(relative: str, kind: str, logical_id: str) -> None:
        files["payload/" + relative] = (safe_path(root, relative), kind, logical_id)

    for source in sources:
        include(source.raw_path, "source", source.snapshot_id)
        include(source.manifest_path, "source", source.source_manifest_id)
    for entry in coverage["sources"]:
        evidence = entry.get("evidence")
        if evidence and evidence.get("document_path"):
            include(evidence["document_path"], "evidence", entry["snapshot_id"])
    include(config.boundary_path, "source", bundle.node_order_hash)
    for relative in version.inventory:
        include(
            relative,
            "evidence" if relative.endswith("coverage.json") else "dataset",
            version.dataset_version_id,
        )
    include(version.manifest_path, "dataset", version.dataset_version_id)
    include(
        (dataset_dir / "sealed.json").relative_to(root).as_posix(),
        "dataset",
        version.dataset_version_id,
    )
    for relative in bundle.artifact_hashes:
        kind = (
            "model"
            if Path(relative).name.startswith("model.")
            else "preprocessor"
            if Path(relative).name == "preprocessor.pkl"
            else "environment"
            if relative.endswith("environment.json")
            else "bundle"
        )
        include(relative, kind, bundle.bundle_id)
    include(bundle.manifest_path, "bundle", bundle.bundle_id)
    for evaluation in evaluations:
        include(evaluation.manifest_path, "evaluation", evaluation.evaluation_id)
        for relative in evaluation.artifact_hashes:
            include(relative, "evaluation", evaluation.evaluation_id)
    include(report_manifest, "report", report["report_id"])
    for relative in report["artifact_hashes"]:
        include(relative, "report", report["report_id"])
    code_archive = root / f"artifacts/code/{bundle.code_hash[7:]}"
    for entry in bundle.code_inventory:
        files["code/" + entry["path"]] = (
            safe_path(code_archive / "files", entry["path"]),
            "code",
            entry["sha256"],
        )
    files["code-snapshot.json"] = (code_archive / "manifest.json", "code", bundle.code_hash)
    files["runtime-environment.json"] = (
        code_archive / "environment.json",
        "environment",
        bundle.environment_lock_hash,
    )
    runs = [load_run(root, identifier) for identifier in bundle.selected_run_ids]
    group = read_json(group_path(root, bundle.parent_run_id) / "manifest.json")
    tracking_ids = (
        [run.mlflow_run_id for run in runs]
        + [evaluation.mlflow_run_id for evaluation in evaluations]
        + [group["mlflow_run_id"]]
    )
    tracker = RunTracker(root)
    metadata = []
    with tracker.lock:
        for tracking_id in sorted(set(tracking_ids)):
            run = tracker.get_run(tracking_id)
            if run.info.status != "FINISHED":
                raise package_failure("Package cannot export incomplete tracking runs.")
            metadata.append(
                {
                    "run_id": tracking_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "parameters": run.data.params,
                    "metrics": run.data.metrics,
                }
            )
    with atomic_directory(root, destination.relative_to(root)) as staging:
        for relative, (source_path, _kind, _logical_id) in sorted(files.items()):
            if not source_path.is_file() or source_path.is_symlink():
                raise package_failure()
            target = safe_path(staging, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            if hash_file(source_path) != hash_file(target):
                raise package_failure()
        write_json(staging / "run-metadata.json", {"runs": metadata})
        replay = {
            "sources": [source.model_dump(mode="json") for source in sources],
            "configuration": config.model_dump(mode="json"),
            "coverage": coverage,
            "original_dataset_version_id": version.dataset_version_id,
            "report_manifest": "payload/" + report_manifest,
            "bundle_id": bundle.bundle_id,
            "evaluation_ids": report["evaluation_ids"],
            "code_hash": bundle.code_hash,
        }
        write_json(staging / "replay.json", replay)
        files["run-metadata.json"] = (
            staging / "run-metadata.json",
            "run_metadata",
            bundle.bundle_id,
        )
        files["replay.json"] = (staging / "replay.json", "replay", version.dataset_version_id)
        inventory = [
            PackageComponent(
                component_type=kind,
                logical_id=logical_id,
                path=relative,
                byte_count=safe_path(staging, relative).stat().st_size,
                sha256=hash_file(safe_path(staging, relative)),
            )
            for relative, (_source, kind, logical_id) in sorted(files.items())
        ]
        payload = {
            "schema_version": "3.0.0",
            "profile": bundle.profile,
            "dataset_version_id": version.dataset_version_id,
            "bundle_id": bundle.bundle_id,
            "evaluation_ids": report["evaluation_ids"],
            "report_manifest_hash": hash_file(safe_path(root, report_manifest)),
            "report_manifest": "payload/" + report_manifest,
            "inventory": [component.model_dump() for component in inventory],
        }
        package = ReproductionPackage.model_validate(
            payload
            | {
                "package_id": canonical_hash(payload),
                "created_at": datetime.now(UTC),
                "status": "sealed",
            }
        )
        write_json(staging / "package-manifest.json", package.model_dump(mode="json"))
        verify_package(staging / "package-manifest.json")
    return manifest
