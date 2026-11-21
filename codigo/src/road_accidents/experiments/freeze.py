from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from road_accidents.config import PipelineError
from road_accidents.data.storage import atomic_directory, read_json, safe_path, write_json
from road_accidents.experiments.schemas import FAMILIES, SEEDS, VARIANTS, FrozenExperiment
from road_accidents.experiments.training import (
    assert_training_allowed,
    group_path,
    load_run,
    run_path,
)
from road_accidents.provenance import canonical_hash, hash_file

VOLATILE_BUNDLE_KEYS = frozenset(
    {"bundle_id", "protocol_hash", "frozen_at", "manifest_path", "status"}
)


def validate_selection(runs: list[dict[str, Any]], profile: str, seeds: list[int]) -> None:
    if (
        profile not in {"confirmatory", "smoke_only"}
        or (profile == "confirmatory" and seeds != list(SEEDS))
        or (profile == "smoke_only" and len(seeds) != 1)
    ):
        raise PipelineError(
            "Frozen selection has an invalid profile or seed set.", 5, "incomplete_protocol"
        )
    expected = {
        (family, variant, seed)
        for family in FAMILIES
        for variant in (
            VARIANTS if profile == "confirmatory" and family in {"graphsage", "gatv2"} else ["full"]
        )
        for seed in seeds
    }
    actual = [(run["model_family"], run["feature_set"], run["seed"]) for run in runs]
    if set(actual) != expected or len(actual) != len(expected):
        raise PipelineError(
            "Frozen selection does not contain the required complete matrix.",
            5,
            "incomplete_protocol",
        )


def bundle_path(root: Path, bundle_id: str) -> Path:
    run_path(root, bundle_id)
    return safe_path(root, f"artifacts/bundles/{bundle_id[7:]}/manifest.json")


def verify_code(bundle: FrozenExperiment) -> None:
    code_root = Path(__file__).resolve().parents[3]
    if canonical_hash(bundle.code_inventory) != bundle.code_hash:
        raise PipelineError("Frozen code inventory is invalid.", 5, "code_changed")
    for entry in bundle.code_inventory:
        if entry["path"] == "specs/001-predicao-acidentes-rodovias/tasks.md":
            continue
        path = safe_path(code_root, entry["path"])
        if not path.is_file() or path.is_symlink() or hash_file(path) != entry["sha256"]:
            raise PipelineError(
                "Source code changed since experiment selection.", 5, "code_changed"
            )
    if hash_file(code_root / "uv.lock") != bundle.environment_lock_hash:
        raise PipelineError(
            "Environment lock differs from the frozen experiment.", 5, "environment_changed"
        )


def load_bundle(root: Path, bundle_id: str, *, check_code: bool = True) -> FrozenExperiment:
    try:
        bundle = FrozenExperiment.model_validate(read_json(bundle_path(root, bundle_id)))
    except (OSError, ValueError):
        raise PipelineError("A valid frozen experiment is required.", 5, "invalid_bundle") from None
    expected = canonical_hash(bundle.model_dump(mode="json"), exclude_keys=VOLATILE_BUNDLE_KEYS)
    if (
        bundle.status != "frozen"
        or bundle.bundle_id != bundle_id
        or bundle.protocol_hash != expected
        or bundle_id != expected
    ):
        raise PipelineError("Frozen experiment identity failed verification.", 5, "bundle_changed")
    for relative, digest in bundle.artifact_hashes.items():
        path = safe_path(root, relative)
        if not path.is_file() or path.is_symlink() or hash_file(path) != digest:
            raise PipelineError(
                "Frozen experiment artifact changed or is missing.", 5, "model_changed"
            )
    if check_code:
        verify_code(bundle)
    return bundle


def freeze_experiment(root: Path, parent_run_id: str) -> FrozenExperiment:
    root = root.resolve()
    group = read_json(group_path(root, parent_run_id) / "manifest.json")
    locks = root / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with FileLock(str(locks / "protocol.lock")):
        runs = [load_run(root, identifier) for identifier in group["selected_run_ids"]]
        validate_selection(
            [run.model_dump(mode="json") for run in runs], group["profile"], group["seeds"]
        )
        first = runs[0]
        for run in runs:
            if (
                run.development_digest,
                run.code_hash,
                run.environment_lock_hash,
                run.node_order_hash,
                run.profile,
            ) != (
                group["development_digest"],
                group["code_hash"],
                group["environment_lock_hash"],
                first.node_order_hash,
                group["profile"],
            ) or run.validation_metrics["auroc"]["value"] is None:
                raise PipelineError(
                    "Selected runs do not share the frozen development protocol.",
                    5,
                    "invalid_selection",
                )
        full_columns = [run.feature_columns for run in runs if run.feature_set == "full"]
        if any(columns != full_columns[0] for columns in full_columns):
            raise PipelineError(
                "Full models do not share the same non-topological features.", 5, "feature_mismatch"
            )
        hashes = {}
        for run in runs:
            hashes.update(run.artifact_hashes)
            manifest = run_path(root, run.run_id) / "manifest.json"
            hashes[manifest.relative_to(root).as_posix()] = hash_file(manifest)
        group_manifest = group_path(root, parent_run_id) / "manifest.json"
        hashes[group_manifest.relative_to(root).as_posix()] = hash_file(group_manifest)
        environment = root / f"artifacts/code/{first.code_hash[7:]}/environment.json"
        hashes[environment.relative_to(root).as_posix()] = hash_file(environment)
        payload = {
            "schema_version": "3.0.0",
            "profile": group["profile"],
            "parent_run_id": parent_run_id,
            "development_dataset_version_id": group["dataset_version_id"],
            "development_digest": group["development_digest"],
            "selected_run_ids": group["selected_run_ids"],
            "artifact_hashes": hashes,
            "baseline_reference": group["baseline_reference"],
            "seeds": group["seeds"],
            "ablation_test_2026": group["ablation_test_2026"],
            "seasonal_rule": group["seasonal_rule"],
            "code_hash": first.code_hash,
            "code_inventory": first.code_inventory,
            "environment_lock_hash": first.environment_lock_hash,
            "node_order_hash": first.node_order_hash,
        }
        identifier = canonical_hash(payload)
        path = bundle_path(root, identifier)
        if path.exists():
            return load_bundle(root, identifier)
        assert_training_allowed(root, first.development_digest)
        bundle = FrozenExperiment.model_validate(
            payload
            | {
                "bundle_id": identifier,
                "protocol_hash": identifier,
                "frozen_at": datetime.now(UTC),
                "status": "frozen",
                "manifest_path": path.relative_to(root).as_posix(),
            }
        )
        verify_code(bundle)
        with atomic_directory(root, path.parent.relative_to(root)) as staging:
            write_json(staging / "manifest.json", bundle.model_dump(mode="json"))
        return bundle
