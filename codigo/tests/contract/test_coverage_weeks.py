from datetime import date

import pytest

from road_accidents.config import PipelineError
from tests.fixtures.prf_factory import coverage_document


def test_partial_coverage_never_creates_unobserved_negative_weeks():
    from road_accidents.data.coverage import CoverageLedger
    from road_accidents.features.weekly import build_calendar

    entries = [coverage_document(year, "sha256:" + f"{year:064d}") for year in range(2019, 2027)]
    calendar = build_calendar(
        CoverageLedger.from_documents(entries, profile="smoke_only")
    ).set_index("week_start")
    assert calendar.loc[date(2026, 7, 20), "eligibility_status"] == "eligible"
    assert calendar.loc[date(2026, 7, 27), "exclusion_reason"] == "incomplete_week"
    assert calendar.loc[date(2026, 8, 3), "exclusion_reason"] == "unobserved_week"
    assert date(2019, 1, 7) == calendar[calendar.eligibility_status == "eligible"].index.min()


def test_unknown_coverage_is_not_inferred_from_events():
    from road_accidents.data.coverage import CoverageLedger
    from road_accidents.features.weekly import build_calendar

    entries = [coverage_document(year, "sha256:" + f"{year:064d}") for year in range(2019, 2027)]
    entries[-1].update(coverage_status="unknown", coverage_intervals=[], evidence=None)
    with pytest.raises(PipelineError, match="coverage"):
        build_calendar(CoverageLedger.from_documents(entries, profile="smoke_only"))


def test_new_coverage_evidence_changes_identity_not_source_bytes():
    from road_accidents.data.coverage import CoverageLedger

    entries = [coverage_document(2026, "sha256:" + "a" * 64)]
    first = CoverageLedger.from_documents(entries, profile="smoke_only")
    entries[0]["evidence"]["verified_at"] = "2026-09-02T00:00:00Z"
    second = CoverageLedger.from_documents(entries, profile="smoke_only")
    assert first.digest != second.digest
    assert first.entries[0].snapshot_id == second.entries[0].snapshot_id


def test_synthetic_or_tampered_evidence_cannot_approve_real_coverage():
    from road_accidents.data.coverage import CoverageLedger

    entries = [coverage_document(2026, "sha256:" + "a" * 64)]
    with pytest.raises(PipelineError, match="Synthetic"):
        CoverageLedger.from_documents(entries, profile="confirmatory")
    entries[0]["evidence"]["statement"] = "altered statement"
    with pytest.raises(PipelineError, match="integrity"):
        CoverageLedger.from_documents(entries, profile="smoke_only")


def test_publication_is_not_inferred_from_acquisition():
    from datetime import datetime

    from road_accidents.data.coverage import CoverageLedger

    ledger = CoverageLedger.from_documents(
        [coverage_document(2026, "sha256:" + "a" * 64)], profile="smoke_only"
    )
    assert ledger.covering_ids(date(2026, 1, 5), date(2026, 1, 12))
    assert not ledger.covering_ids(
        date(2026, 1, 5),
        date(2026, 1, 12),
        available_before=datetime.fromisoformat("2026-01-12T00:00:00-03:00"),
    )


def test_owner_coverage_assumption_enables_targets_not_historical_publication(tmp_path):
    from datetime import datetime

    from road_accidents.data.coverage import CoverageLedger
    from road_accidents.provenance import hash_bytes

    authorization = b"Owner authorizes assumed coverage, not proof of historical publication."
    (tmp_path / "authorization.txt").write_bytes(authorization)
    entry = coverage_document(2026, "sha256:" + "a" * 64)
    entry["evidence"].update(
        kind="assumption",
        document_path="authorization.txt",
        verified_by="project_owner",
        sha256=hash_bytes(authorization),
        statement=authorization.decode(),
    )
    entry["available_at"] = None
    ledger = CoverageLedger.from_documents([entry], profile="confirmatory", root=tmp_path)
    assert ledger.covering_ids(date(2026, 1, 5), date(2026, 1, 12))
    assert not ledger.covering_ids(
        date(2026, 1, 5),
        date(2026, 1, 12),
        available_before=datetime.fromisoformat("2026-02-01T00:00:00-03:00"),
    )
    entry["available_at"] = "2026-01-01T00:00:00Z"
    with pytest.raises(PipelineError):
        CoverageLedger.from_documents([entry], profile="confirmatory", root=tmp_path)
