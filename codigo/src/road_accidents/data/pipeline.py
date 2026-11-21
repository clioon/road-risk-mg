import hashlib
import io
import json
import re
import sys
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from road_accidents.config import PipelineError
from road_accidents.data.coverage import CoverageLedger, apply_coverage
from road_accidents.data.osm import load_osm_graph
from road_accidents.data.prf import curate_prf
from road_accidents.data.schemas import (
    SPLITS,
    DatasetConfig,
    DatasetVersion,
    RoadNetworkVersion,
    SourceSnapshot,
    validate_table,
)
from road_accidents.data.storage import (
    atomic_directory,
    read_json,
    read_parquet,
    safe_path,
    write_geoparquet,
    write_json,
    write_parquet,
)
from road_accidents.features.temporal import (
    FEATURE_COLUMNS,
    features_for_partition,
    validate_feature_columns,
)
from road_accidents.features.weekly import annotate_targets, build_calendar, labels_for_partition
from road_accidents.graph.adjacency import build_adjacency, node_order
from road_accidents.graph.matching import apply_matches, match_accidents
from road_accidents.graph.segments import build_segments
from road_accidents.provenance import canonical_bytes, canonical_hash, capture_code_state, hash_file


def dataset_directory(root: Path, dataset_version_id: str) -> Path:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", dataset_version_id):
        raise PipelineError(
            "Dataset version must be an exact content identifier.", 2, "usage_error"
        )
    return safe_path(root, f"data/processed/datasets/{dataset_version_id[7:]}")


def table_digest(frame: pd.DataFrame) -> str:
    content = pd.DataFrame(frame.copy() if "geometry" in frame else frame, copy=False)
    if "geometry" in content:
        content["geometry"] = content.geometry.map(
            lambda geometry: geometry.normalize().wkb_hex if geometry is not None else None
        )
    digest = hashlib.sha256()
    digest.update(b'{"columns":' + canonical_bytes(list(content.columns)) + b',"rows":[')
    first = True
    for start in range(0, len(content), 5000):
        rows = json.loads(
            content.iloc[start : start + 5000].to_json(
                orient="records", date_format="iso", double_precision=15
            )
        )
        encoded = canonical_bytes(rows)[1:-1]
        if not first:
            digest.update(b",")
        digest.update(encoded)
        first = False
    digest.update(b"]}")
    return "sha256:" + digest.hexdigest()


def load_dataset(root: Path, dataset_version_id: str, *, verify: bool = True) -> DatasetVersion:
    directory = dataset_directory(root, dataset_version_id)
    manifest = directory / "manifest.json"
    try:
        version = DatasetVersion.model_validate(read_json(manifest))
    except (OSError, ValueError):
        raise PipelineError("Dataset manifest is missing or invalid.") from None
    if version.dataset_version_id != dataset_version_id:
        raise PipelineError("Dataset identity does not match its directory.")
    if verify:
        for relative, digest in version.inventory.items():
            path = safe_path(root, relative)
            if not path.is_file() or path.is_symlink() or hash_file(path) != digest:
                raise PipelineError("Dataset inventory failed integrity verification.")
    seal_path = directory / "sealed.json"
    if seal_path.exists():
        seal = read_json(seal_path)
        if seal.get("dataset_version_id") != dataset_version_id or seal.get(
            "manifest_sha256"
        ) != hash_file(manifest):
            raise PipelineError("Dataset seal does not match its manifest.")
        version = version.model_copy(update={"status": "sealed"})
    return version


def build_dataset(
    root: Path,
    config: DatasetConfig,
    prf_snapshots: list[SourceSnapshot],
    osm_snapshot: SourceSnapshot,
    coverage: CoverageLedger,
    history_snapshots: list[SourceSnapshot] | None = None,
) -> DatasetVersion:
    root = root.resolve()
    snapshots = sorted(prf_snapshots, key=lambda source: source.nominal_year or 0)
    history_sources = history_snapshots or []
    if len(snapshots) != 8 or {source.nominal_year for source in snapshots} != set(
        range(2019, 2027)
    ):
        raise PipelineError("Exactly one approved PRF snapshot per study year is required.")
    loaded_ids = {source.snapshot_id for source in snapshots + history_sources}
    if {entry.snapshot_id for entry in coverage.entries} != loaded_ids:
        raise PipelineError("Coverage must identify exactly the loaded source snapshots.")
    if coverage.profile != config.profile or any(
        source.profile != config.profile or source.status != "validated"
        for source in snapshots + history_sources + [osm_snapshot]
    ):
        raise PipelineError(
            "Source, coverage and dataset profiles must match; synthetic data cannot be promoted."
        )
    selected_coverage = coverage.select([source.snapshot_id for source in snapshots])
    selected_coverage.require_study_coverage()
    for source in history_sources:
        coverage.for_snapshot(source.snapshot_id)
    boundary_file = safe_path(root, config.boundary_path)
    if not boundary_file.is_file():
        raise PipelineError("Versioned MG boundary is unavailable.", 3, "source_access")
    boundary = gpd.read_file(boundary_file)
    if config.profile == "confirmatory" and ("synthetic" in boundary and boundary.synthetic.any()):
        raise PipelineError("Synthetic geography cannot support a confirmatory dataset.")
    print("Dataset: loading the verified historical network.", file=sys.stderr, flush=True)
    graph = load_osm_graph(osm_snapshot, root)
    segments = build_segments(graph, osm_snapshot.snapshot_id, boundary)
    del graph
    adjacency = build_adjacency(segments)
    order = node_order(segments)
    calendar = build_calendar(selected_coverage)
    approved = [
        apply_coverage(root, source, coverage.for_snapshot(source.snapshot_id))
        for source in snapshots
    ]
    print(
        "Dataset: curating approved PRF records and matching segments.", file=sys.stderr, flush=True
    )
    curated = gpd.GeoDataFrame(
        pd.concat([curate_prf(source, root, boundary) for source in approved], ignore_index=True),
        geometry="geometry",
        crs=4326,
    )
    duplicate = (
        curated.accident_id.notna()
        & curated.accident_id.duplicated(keep="first")
        & (curated.quality_status == "valid")
    )
    curated.loc[duplicate, "quality_status"] = "invalid"
    curated.loc[duplicate, "exclusion_reason"] = "duplicate"
    matches = match_accidents(curated, segments)
    accidents = annotate_targets(apply_matches(curated, matches), calendar)
    del curated
    history_records = [accidents]
    for source in history_sources:
        past = curate_prf(source, root, boundary)
        history_records.append(apply_matches(past, match_accidents(past, segments)))
    historical = pd.concat(history_records, ignore_index=True)
    transformed: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    logical_partitions = {}
    for split in SPLITS:
        print(f"Dataset: preparing {split} segment-weeks.", file=sys.stderr, flush=True)
        labels = labels_for_partition(segments, accidents, calendar, split)
        features = features_for_partition(segments, historical, calendar, coverage, split)
        if not labels[["segment_id", "week_start"]].equals(features[["segment_id", "week_start"]]):
            raise PipelineError(
                "Feature and target populations are not aligned.", 7, "population_mismatch"
            )
        transformed[split] = features, labels
        logical_partitions[split] = {
            "features": table_digest(features),
            "labels": table_digest(labels),
        }
    configuration_hash = canonical_hash(config.model_dump(mode="json"))
    graph_hash = canonical_hash(
        {
            "segments": table_digest(segments),
            "adjacency": table_digest(adjacency),
            "order": table_digest(order),
        }
    )
    development_digest = canonical_hash(
        {
            "graph": graph_hash,
            "configuration": configuration_hash,
            "source_manifests": [
                source.source_manifest_id
                for source in approved
                if source.nominal_year is not None and source.nominal_year <= 2024
            ],
            "partitions": {split: logical_partitions[split] for split in ("train", "validation")},
        }
    )
    code_root = Path(__file__).resolve().parents[3]
    code = capture_code_state(code_root)
    lock_hash = hash_file(code_root / "uv.lock")
    identity = canonical_hash(
        {
            "schema_version": "3.0.0",
            "configuration": configuration_hash,
            "sources": [
                source.source_manifest_id for source in approved + history_sources + [osm_snapshot]
            ],
            "coverage": coverage.digest,
            "boundary": hash_file(boundary_file),
            "graph": graph_hash,
            "partitions": logical_partitions,
            "code": code["code_hash"],
            "lock": lock_hash,
        }
    )
    destination = dataset_directory(root, identity)
    if destination.exists():
        return load_dataset(root, identity)
    prefix = destination.relative_to(root).as_posix()
    partitions: dict[str, dict[str, Any]] = {}
    row_counts = {
        "approved_source_rows": len(accidents),
        "valid_records": int((accidents.quality_status == "valid").sum()),
        "invalid_records": int((accidents.quality_status == "invalid").sum()),
        "target_eligible": int(accidents.target_eligible.sum()),
        "target_excluded": int((~accidents.target_eligible).sum()),
        "intrinsic_reasons": accidents.exclusion_reason.value_counts().to_dict(),
        "target_reasons": accidents.target_exclusion_reason.value_counts().to_dict(),
        "history_source_rows": sum(len(frame) for frame in history_records[1:]),
    }
    network_manifest = RoadNetworkVersion(
        network_version_id=str(segments.network_version_id.iloc[0]),
        source_snapshot_id=osm_snapshot.snapshot_id,
        effective_at=config.spatial.osm_effective_at,
        configuration_hash=configuration_hash,
        junction_count=len(set(segments.from_junction_id) | set(segments.to_junction_id)),
        segment_count=len(segments),
        adjacency_count=len(adjacency),
        status="sealed",
    )
    with atomic_directory(root, destination.relative_to(root)) as staging:
        write_geoparquet(staging / "network/segments.parquet", segments)
        write_geoparquet(
            staging / "accidents/records.parquet",
            gpd.GeoDataFrame(accidents, geometry="geometry", crs=4326),
        )
        write_parquet(staging / "matches/matches.parquet", matches)
        write_parquet(staging / "calendar/weeks.parquet", calendar)
        write_parquet(staging / "graph/adjacency.parquet", adjacency)
        write_parquet(staging / "graph/nodes.parquet", order)
        write_json(
            staging / "network/manifest.json",
            network_manifest.model_dump(mode="json")
            | {
                "boundary_policy": segments.attrs["boundary_policy"],
                "boundary_sha256": hash_file(boundary_file),
            },
        )
        write_json(
            staging / "graph/manifest.json",
            {
                "schema_version": "3.0.0",
                "network_version_id": network_manifest.network_version_id,
                "nodes": f"{prefix}/graph/nodes.parquet",
                "adjacency": f"{prefix}/graph/adjacency.parquet",
                "nodes_sha256": hash_file(staging / "graph/nodes.parquet"),
                "adjacency_sha256": hash_file(staging / "graph/adjacency.parquet"),
                "node_order_hash": table_digest(order),
                "segment_count": len(order),
            },
        )
        for split, (features, labels) in transformed.items():
            feature_path = f"segment_week/{split}/features.parquet"
            label_path = f"segment_week/{split}/labels.parquet"
            write_parquet(staging / feature_path, features)
            write_parquet(staging / label_path, labels)
            partitions[split] = {
                "features": f"{prefix}/{feature_path}",
                "labels": f"{prefix}/{label_path}",
                "features_sha256": hash_file(staging / feature_path),
                "labels_sha256": hash_file(staging / label_path),
                "population_hash": table_digest(labels[["segment_id", "week_start"]]),
                "row_count": len(labels),
                "week_count": labels.week_start.nunique(),
                "start": str(labels.week_start.min()),
                "end_exclusive": str(
                    calendar[
                        calendar.week_start == labels.week_start.max()
                    ].week_end_exclusive.iloc[0]
                ),
            }
            write_json(
                staging / f"segment_week/{split}/target-audit.json",
                {
                    "positive_count": int(labels.label.sum()),
                    "event_count": int(labels.event_count.sum()),
                    "prevalence": float(labels.label.mean()),
                    "split": split,
                },
            )
        write_json(staging / "configuration.json", config.model_dump(mode="json"))
        write_json(
            staging / "coverage.json",
            {
                "profile": coverage.profile,
                "sources": [entry.model_dump(mode="json") for entry in coverage.entries],
            },
        )
        write_json(
            staging / "sources.json",
            {
                "schema_version": "3.0.0",
                "sources": [
                    source.model_dump(mode="json")
                    for source in approved + history_sources + [osm_snapshot]
                ],
            },
        )
        write_json(
            staging / "validation.json",
            {
                "status": "PASS",
                "dataset_version_id": identity,
                "coverage_basis": "owner_assumption"
                if any(
                    entry.evidence is not None and entry.evidence.kind == "assumption"
                    for entry in coverage.entries
                )
                else "recorded_evidence",
                "row_counts": row_counts,
                "checks": [
                    "privacy_passed",
                    "coverage_recorded_with_evidence_or_authorized_assumption",
                    "mg_only",
                    "complete_targets",
                    "prior_features",
                    "shared_order",
                    "no_fitted_transformers",
                ],
            },
        )
        write_json(
            staging / "development.json",
            {
                "schema_version": "3.0.0",
                "dataset_version_id": identity,
                "development_digest": development_digest,
                "profile": config.profile,
                "partitions": {split: partitions[split] for split in ("train", "validation")},
                "feature_columns": list(FEATURE_COLUMNS),
                "graph_manifest": f"{prefix}/graph/manifest.json",
                "graph_manifest_sha256": hash_file(staging / "graph/manifest.json"),
            },
        )
        inventory = {
            f"{prefix}/{path.relative_to(staging).as_posix()}": hash_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        version = DatasetVersion(
            dataset_version_id=identity,
            profile=config.profile,
            prf_snapshot_ids=[source.snapshot_id for source in approved],
            source_manifest_ids=[
                source.source_manifest_id for source in approved + history_sources + [osm_snapshot]
            ],
            history_snapshot_ids=[source.snapshot_id for source in history_sources],
            development_digest=development_digest,
            network_version_id=network_manifest.network_version_id,
            configuration_hash=configuration_hash,
            git_revision=code["git_revision"],
            code_hash=code["code_hash"],
            environment_lock_hash=lock_hash,
            created_at=datetime.now(UTC),
            row_counts=row_counts,
            partition_manifests=partitions,
            status="validated",
            validation_report_uri=f"{prefix}/validation.json",
            manifest_path=f"{prefix}/manifest.json",
            graph_manifest=f"{prefix}/graph/manifest.json",
            inventory=inventory,
        )
        write_json(staging / "manifest.json", version.model_dump(mode="json"))
    return version


def validate_dataset(
    root: Path, dataset_version_id: str, *, publish_seal: bool = True
) -> DatasetVersion:
    version = load_dataset(root, dataset_version_id)
    directory = dataset_directory(root, dataset_version_id)
    for split, partition in version.partition_manifests.items():
        labels = read_parquet(safe_path(root, partition["labels"]))
        features = read_parquet(safe_path(root, partition["features"]))
        validate_table("SegmentWeekObservation", labels)
        validate_feature_columns(
            column
            for column in features
            if column
            not in {
                "segment_id",
                "week_start",
                "feature_available_at",
                "feature_provenance",
                "history_missing_reasons",
            }
        )
        if features.empty or not features[["segment_id", "week_start"]].equals(
            labels[["segment_id", "week_start"]]
        ):
            raise PipelineError("Dataset populations diverge.", 7, "population_mismatch")
        from road_accidents.features.weekly import TIMEZONE

        starts = pd.to_datetime(features.week_start).dt.tz_localize(TIMEZONE)
        if (pd.to_datetime(features.feature_available_at, utc=True) >= starts).any():
            raise PipelineError(
                "Dataset features violate temporal availability.", 5, "temporal_integrity"
            )
        if split not in SPLITS:
            raise PipelineError("Unexpected dataset partition.")
    counts = version.row_counts
    if counts["approved_source_rows"] != counts["target_eligible"] + counts["target_excluded"]:
        raise PipelineError("Source row audit is incomplete.")
    if version.status != "sealed" and publish_seal:
        seal = {
            "dataset_version_id": dataset_version_id,
            "manifest_sha256": hash_file(directory / "manifest.json"),
            "development_sha256": hash_file(directory / "development.json"),
            "status": "sealed",
        }
        write_json(directory / "sealed.json", seal)
    return version.model_copy(update={"status": "sealed"}) if publish_seal else version


def prepare_dvc(arguments: list[str]) -> int:
    from road_accidents.cli import main

    output = io.StringIO()
    with redirect_stdout(output):
        status = main(["--result-format", "json", "dataset", "build", *arguments])
    if status:
        print(output.getvalue(), end="")
        return status
    result = json.loads(output.getvalue())
    validated = io.StringIO()
    with redirect_stdout(validated):
        status = main(
            [
                "--result-format",
                "json",
                "dataset",
                "validate",
                "--dataset-version",
                result["outputs"]["dataset_version_id"],
            ]
        )
    if status == 0:
        receipt = Path("data/manifests/prepared.json")
        selection = json.loads(validated.getvalue())["outputs"]
        if receipt.exists() and read_json(receipt) != selection:
            raise PipelineError(
                "Prepared selector already identifies different data; let DVC version it."
            )
        if not receipt.exists():
            write_json(receipt, selection)
    print(validated.getvalue(), end="")
    return status


if __name__ == "__main__":
    raise SystemExit(prepare_dvc(sys.argv[1:]))
