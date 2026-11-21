from datetime import UTC, date, datetime
from typing import Any, Literal, Self
from urllib.parse import urlsplit

import pandas as pd
import pandera.pandas as pa
from pydantic import AwareDatetime, Field, field_validator, model_validator

from road_accidents.config import PipelineError, StrictConfig

type Profile = Literal["confirmatory", "smoke_only"]
type Split = Literal["train", "validation", "test_2025", "test_2026"]
SPLITS = {
    "train": (date(2019, 1, 1), date(2024, 1, 1)),
    "validation": (date(2024, 1, 1), date(2025, 1, 1)),
    "test_2025": (date(2025, 1, 1), date(2026, 1, 1)),
    "test_2026": (date(2026, 1, 1), date(2027, 1, 1)),
}
OSM_EFFECTIVE_AT = datetime(2018, 12, 31, 23, 59, 59, tzinfo=UTC)
BUFFER_LIMIT = 268_435_456
PRIVACY_POLICY: Literal["occurrence-osm-allowlist-1"] = "occurrence-osm-allowlist-1"


def public_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Source requires a public HTTPS URL without credentials.")
    if any(
        name in parsed.query.lower() for name in ("token=", "password=", "secret=", "sig=", "key=")
    ):
        raise ValueError("Source URL cannot contain credentials.")
    return value


class CoverageInterval(StrictConfig):
    start: date
    end: date

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError("Coverage interval is reversed.")
        return self


class CoverageEvidence(StrictConfig):
    kind: Literal["official", "attestation", "synthetic", "assumption"]
    statement: str = Field(min_length=1, max_length=65_536)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified_by: str = Field(min_length=1, max_length=120)
    verified_at: AwareDatetime
    document_path: str | None = None
    source_url: str | None = None


class CoverageEntry(StrictConfig):
    snapshot_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    nominal_year: int = Field(ge=2019, le=2026)
    coverage_status: Literal["complete", "partial", "unknown"]
    coverage_intervals: list[CoverageInterval] = Field(default_factory=list)
    evidence: CoverageEvidence | None = None
    available_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def coverage_consistency(self) -> Self:
        if self.evidence is not None and self.evidence.kind == "assumption":
            if self.available_at is not None:
                raise ValueError("Assumed coverage does not prove historical publication.")
            if not self.evidence.document_path:
                raise ValueError("Assumed coverage requires an archived owner authorization.")
        if self.coverage_status == "unknown":
            if self.coverage_intervals or self.available_at:
                raise ValueError("Unknown coverage cannot assert intervals or availability.")
            return self
        if not self.coverage_intervals or self.evidence is None:
            raise ValueError("Known coverage requires intervals and evidence.")
        ordered = sorted(self.coverage_intervals, key=lambda interval: interval.start)
        for index, interval in enumerate(ordered):
            if interval.start.year != self.nominal_year or interval.end.year != self.nominal_year:
                raise ValueError("Coverage belongs to the nominal year.")
            if index and ordered[index - 1].end >= interval.start:
                raise ValueError("Coverage intervals cannot overlap.")
        if self.coverage_status == "complete" and (
            len(ordered) != 1
            or ordered[0].start != date(self.nominal_year, 1, 1)
            or ordered[0].end != date(self.nominal_year, 12, 31)
        ):
            raise ValueError("Complete coverage must span the whole year.")
        return self


class CoverageConfig(StrictConfig):
    profile: Profile = "confirmatory"
    sources: list[CoverageEntry]
    history_sources: list[str] = Field(default_factory=list)


class SourceSnapshot(StrictConfig):
    schema_version: Literal["3.0.0"] = "3.0.0"
    profile: Profile = "confirmatory"
    snapshot_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_manifest_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_type: Literal["prf_occurrence_csv", "osm_network"]
    privacy_status: Literal["passed"] = "passed"
    privacy_policy_version: Literal["occurrence-osm-allowlist-1"] = PRIVACY_POLICY
    source_url: str
    nominal_year: int | None = Field(default=None, ge=2019, le=2026)
    granularity: Literal["occurrence"] | None = None
    retrieved_at: AwareDatetime
    effective_at: AwareDatetime | None = None
    observed_min_date: date | None = None
    observed_max_date: date | None = None
    coverage_status: Literal["unknown", "partial", "complete"] = "unknown"
    coverage_start: date | None = None
    coverage_end: date | None = None
    coverage_intervals: list[CoverageInterval] = Field(default_factory=list)
    coverage_evidence: CoverageEvidence | None = None
    supersedes_snapshot_id: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(gt=0)
    media_type: str
    license: str
    status: Literal["acquired", "validated", "rejected"] = "validated"
    rejection_reason: str | None = None
    raw_path: str
    manifest_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    _public_url = field_validator("source_url")(public_url)

    @model_validator(mode="after")
    def source_consistency(self) -> Self:
        if self.snapshot_id != "sha256:" + self.sha256:
            raise ValueError("Snapshot identity must match the original bytes.")
        if self.source_type == "prf_occurrence_csv" and (
            self.nominal_year is None or self.granularity != "occurrence"
        ):
            raise ValueError("PRF snapshot requires occurrence granularity and year.")
        if self.source_type == "osm_network" and self.effective_at != OSM_EFFECTIVE_AT:
            raise ValueError("Historical OSM snapshot is required.")
        if self.status == "rejected" and not self.rejection_reason:
            raise ValueError("Rejection reason is required.")
        return self


class SourceCatalog(StrictConfig):
    schema_version: Literal["3.0.0"] = "3.0.0"
    sources: list[SourceSnapshot]


class IngestionConfig(StrictConfig):
    max_buffer_bytes: int = Field(default=BUFFER_LIMIT, gt=0)


class PrivacyConfig(StrictConfig):
    ingestion_policy: Literal["reject_before_persist"] = "reject_before_persist"


class SourceEntry(StrictConfig):
    url: str
    catalog: bool = False
    granularity: Literal["occurrence"] = "occurrence"
    selected_manifest: str | None = None
    local_file: str | None = None
    encoding: Literal["cp1252", "utf-8-sig"] = "cp1252"
    delimiter: Literal[";"] = ";"

    _public_url = field_validator("url")(public_url)


class SourcesConfig(StrictConfig):
    profile: Profile = "confirmatory"
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    years: dict[int, SourceEntry]

    @field_validator("years")
    @classmethod
    def all_years(cls, value: dict[int, SourceEntry]) -> dict[int, SourceEntry]:
        if set(value) != set(range(2019, 2027)):
            raise ValueError("Exactly the eight nominal PRF years are required.")
        return value


class StudyConfig(StrictConfig):
    state: Literal["MG"] = "MG"
    timezone: Literal["America/Sao_Paulo"] = "America/Sao_Paulo"
    start: Literal["2019-01-01"] = "2019-01-01"
    end_exclusive: Literal["2027-01-01"] = "2027-01-01"


class SpatialConfig(StrictConfig):
    metric_crs: Literal["EPSG:5880"] = "EPSG:5880"
    max_snap_distance_m: Literal[500] = 500
    osm_effective_at: AwareDatetime = OSM_EFFECTIVE_AT

    @field_validator("osm_effective_at")
    @classmethod
    def historical(cls, value: datetime) -> datetime:
        if value != OSM_EFFECTIVE_AT:
            raise ValueError("The historical OSM instant cannot change.")
        return value


class TemporalConfig(StrictConfig):
    lags_weeks: tuple[int, ...] = (1, 4, 13, 52)
    availability_policy: Literal["documented_before_prediction"] = "documented_before_prediction"

    @field_validator("lags_weeks")
    @classmethod
    def fixed_lags(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != (1, 4, 13, 52):
            raise ValueError("Historical windows are fixed by the protocol.")
        return value


class DatasetConfig(StrictConfig):
    profile: Profile = "confirmatory"
    boundary_path: str = "data/raw/osm/mg-boundary.geojson"
    study: StudyConfig = Field(default_factory=StudyConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    splits: dict[str, str] = Field(
        default_factory=lambda: {name: f"{start}/{end}" for name, (start, end) in SPLITS.items()}
    )
    coverage: dict[str, bool] = Field(default_factory=lambda: {"require_evidence": True})
    weeks: dict[str, str] = Field(default_factory=lambda: {"policy": "complete_within_split"})
    privacy: dict[str, bool] = Field(default_factory=lambda: {"require_passed_snapshot": True})

    @model_validator(mode="after")
    def protocol(self) -> Self:
        if self.splits != {name: f"{start}/{end}" for name, (start, end) in SPLITS.items()}:
            raise ValueError("Temporal partitions cannot change.")
        if (
            self.coverage != {"require_evidence": True}
            or self.privacy != {"require_passed_snapshot": True}
            or self.weeks != {"policy": "complete_within_split"}
        ):
            raise ValueError("Dataset integrity gates cannot be disabled.")
        return self


class RoadNetworkVersion(StrictConfig):
    network_version_id: str
    source_snapshot_id: str
    effective_at: AwareDatetime
    source_crs: Literal["EPSG:4326"] = "EPSG:4326"
    metric_crs: Literal["EPSG:5880"] = "EPSG:5880"
    configuration_hash: str
    junction_count: int = Field(ge=0)
    segment_count: int = Field(gt=0)
    adjacency_count: int = Field(ge=0)
    status: Literal["building", "validated", "sealed", "invalid"]


class DatasetVersion(StrictConfig):
    schema_version: Literal["3.0.0"] = "3.0.0"
    dataset_version_id: str
    profile: Profile
    prf_snapshot_ids: list[str]
    source_manifest_ids: list[str]
    history_snapshot_ids: list[str] = Field(default_factory=list)
    development_digest: str
    supersedes_dataset_version_id: str | None = None
    network_version_id: str
    configuration_hash: str
    git_revision: str | None
    code_hash: str
    environment_lock_hash: str
    created_at: AwareDatetime
    row_counts: dict[str, Any]
    partition_manifests: dict[str, dict[str, Any]]
    status: Literal["building", "validated", "sealed", "invalid"]
    validation_report_uri: str
    manifest_path: str
    graph_manifest: str
    inventory: dict[str, str]


TABLE_SCHEMAS = {
    "AccidentRecord": pa.DataFrameSchema(
        {
            "record_id": pa.Column(str, unique=True),
            "source_snapshot_id": pa.Column(str),
            "quality_status": pa.Column(str, pa.Check.isin(["valid", "invalid"])),
            "target_eligible": pa.Column(bool),
            "exclusion_reason": pa.Column(str, nullable=True),
            "target_exclusion_reason": pa.Column(str, nullable=True),
        }
    ),
    "RoadSegment": pa.DataFrameSchema(
        {
            "segment_id": pa.Column(str, unique=True),
            "network_version_id": pa.Column(str),
            "length_m": pa.Column(float, pa.Check.gt(0)),
            "chord_length_m": pa.Column(float, pa.Check.ge(0)),
            "sinuosity": pa.Column(float, nullable=True),
        }
    ),
    "SegmentAdjacency": pa.DataFrameSchema(
        {
            "network_version_id": pa.Column(str),
            "segment_id_low": pa.Column(str),
            "segment_id_high": pa.Column(str),
            "shared_junction_id": pa.Column(str),
        },
        unique=["network_version_id", "segment_id_low", "segment_id_high", "shared_junction_id"],
    ),
    "SpatialMatch": pa.DataFrameSchema(
        {
            "record_id": pa.Column(str, unique=True),
            "match_status": pa.Column(
                str, pa.Check.isin(["matched", "rejected_distance", "no_candidate"])
            ),
            "distance_m": pa.Column(float, pa.Check.ge(0), nullable=True),
        }
    ),
    "PredictionWeek": pa.DataFrameSchema(
        {
            "week_start": pa.Column(unique=True),
            "week_end_exclusive": pa.Column(),
            "eligibility_status": pa.Column(str, pa.Check.isin(["eligible", "excluded"])),
            "split": pa.Column(str, pa.Check.isin(list(SPLITS)), nullable=True),
        }
    ),
    "SegmentWeekObservation": pa.DataFrameSchema(
        {
            "segment_id": pa.Column(str),
            "week_start": pa.Column(),
            "event_count": pa.Column(int, pa.Check.ge(0)),
            "label": pa.Column(int, pa.Check.isin([0, 1])),
        },
        unique=["segment_id", "week_start"],
    ),
}


def validate_table(entity: str, frame: pd.DataFrame) -> None:
    try:
        TABLE_SCHEMAS[entity].validate(frame, lazy=True)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors):
        raise PipelineError(
            "Dataset table violates its declared schema.", 4, "schema_error"
        ) from None
