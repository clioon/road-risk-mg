from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Self
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from road_accidents.config import PipelineError
from road_accidents.data.schemas import BUFFER_LIMIT, CoverageEntry, Profile, SourceSnapshot
from road_accidents.data.storage import publish_snapshot, read_limited, safe_path
from road_accidents.provenance import canonical_hash, hash_bytes, hash_file


class CoverageLedger:
    def __init__(self, entries: list[CoverageEntry], profile: Profile) -> None:
        self.entries = tuple(entries)
        self.profile = profile
        self.digest = canonical_hash(
            [
                entry.model_dump(mode="json")
                for entry in sorted(entries, key=lambda item: item.snapshot_id)
            ]
        )

    @classmethod
    def from_documents(
        cls,
        documents: list[dict[str, Any]],
        *,
        profile: Profile,
        root: Path | None = None,
    ) -> Self:
        try:
            entries = [CoverageEntry.model_validate(document) for document in documents]
        except ValidationError:
            raise PipelineError("Invalid coverage contract.", 4, "coverage_invalid") from None
        if len({entry.snapshot_id for entry in entries}) != len(entries):
            raise PipelineError("Duplicate coverage entries for the same snapshot.")
        for entry in entries:
            evidence = entry.evidence
            if entry.coverage_status == "unknown":
                continue
            if evidence is None:
                raise PipelineError("Known coverage requires evidence.")
            if evidence.kind == "synthetic" and profile != "smoke_only":
                raise PipelineError("Synthetic coverage cannot support a confirmatory dataset.")
            if evidence.kind == "assumption" and evidence.verified_by != "project_owner":
                raise PipelineError(
                    "Assumed coverage requires explicit project-owner authorization."
                )
            if evidence.document_path:
                if root is None:
                    raise PipelineError("Coverage document must be available for verification.")
                document = safe_path(root, evidence.document_path)
                if not document.is_file() or hash_file(document) != evidence.sha256:
                    raise PipelineError("Coverage document failed integrity verification.")
            elif hash_bytes(evidence.statement.encode("utf-8")) != evidence.sha256:
                raise PipelineError("Coverage statement failed integrity verification.")
            if evidence.kind == "official" and (
                not evidence.source_url or not evidence.document_path
            ):
                raise PipelineError("Official coverage requires an archived source document.")
            if entry.available_at and any(
                interval.end > entry.available_at.astimezone(ZoneInfo("America/Sao_Paulo")).date()
                for interval in entry.coverage_intervals
            ):
                raise PipelineError(
                    "Publication evidence cannot cover future dates.", 5, "temporal_integrity"
                )
        return cls(entries, profile)

    def for_snapshot(self, snapshot_id: str) -> CoverageEntry:
        for entry in self.entries:
            if entry.snapshot_id == snapshot_id:
                return entry
        raise PipelineError("Source snapshot has no coverage evidence.", 4, "coverage_unknown")

    def select(self, snapshot_ids: list[str]) -> Self:
        return type(self)(
            [self.for_snapshot(identifier) for identifier in snapshot_ids], self.profile
        )

    def require_study_coverage(self) -> None:
        if {entry.nominal_year for entry in self.entries} != set(range(2019, 2027)):
            raise PipelineError(
                "Study coverage requires all eight nominal years.", 4, "coverage_unknown"
            )
        if any(entry.coverage_status == "unknown" for entry in self.entries):
            raise PipelineError(
                "Unknown coverage cannot produce a sealed dataset.", 4, "coverage_unknown"
            )

    def covering_ids(
        self,
        start: date,
        end_exclusive: date,
        *,
        available_before: datetime | None = None,
    ) -> list[str]:
        intervals = []
        for entry in self.entries:
            if entry.coverage_status == "unknown":
                continue
            if (
                available_before is not None
                and entry.evidence is not None
                and entry.evidence.kind == "assumption"
            ):
                continue
            if available_before is not None and (
                entry.available_at is None or entry.available_at >= available_before
            ):
                continue
            for interval in entry.coverage_intervals:
                end = interval.end + timedelta(days=1)
                if interval.start < end_exclusive and end > start:
                    intervals.append((interval.start, end, entry.snapshot_id))
        cursor = start
        selected: list[str] = []
        for beginning, end, identifier in sorted(intervals):
            if beginning > cursor:
                return []
            if end > cursor:
                cursor = end
                selected.append(identifier)
            if cursor >= end_exclusive:
                return sorted(set(selected))
        return []

    def overlaps(self, start: date, end_exclusive: date) -> bool:
        return any(
            interval.start < end_exclusive and interval.end >= start
            for entry in self.entries
            for interval in entry.coverage_intervals
        )


def apply_coverage(root: Path, snapshot: SourceSnapshot, entry: CoverageEntry) -> SourceSnapshot:
    if snapshot.snapshot_id != entry.snapshot_id or snapshot.nominal_year != entry.nominal_year:
        raise PipelineError("Coverage evidence does not identify the selected source.")
    intervals = entry.coverage_intervals
    identity = canonical_hash(
        {
            "snapshot_id": snapshot.snapshot_id,
            "source_url": snapshot.source_url,
            "profile": snapshot.profile,
            "privacy_policy_version": snapshot.privacy_policy_version,
            "coverage": entry.model_dump(mode="json"),
        }
    )
    updated = snapshot.model_copy(
        update={
            "source_manifest_id": identity,
            "coverage_status": entry.coverage_status,
            "coverage_start": min((interval.start for interval in intervals), default=None),
            "coverage_end": max((interval.end for interval in intervals), default=None),
            "coverage_intervals": list(intervals),
            "coverage_evidence": entry.evidence,
            "manifest_path": f"data/manifests/sources/{identity[7:]}.json",
        }
    )
    with safe_path(root, snapshot.raw_path).open("rb") as source:
        payload = read_limited(source, int(snapshot.metadata.get("max_buffer_bytes", BUFFER_LIMIT)))
    return publish_snapshot(root, updated, payload)
