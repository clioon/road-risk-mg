from datetime import date


def test_cross_partition_weeks_have_no_targets():
    from road_accidents.features.weekly import partition_for_week

    assert partition_for_week(date(2024, 12, 30)) is None
    assert partition_for_week(date(2025, 12, 29)) is None
    assert partition_for_week(date(2019, 12, 30)) == "train"


def test_civil_week_spans_historical_daylight_saving():
    from road_accidents.features.weekly import week_bounds

    start, end = week_bounds(date(2019, 2, 11))
    assert (end.date() - start.date()).days == 7
    assert end.timestamp() - start.timestamp() == 169 * 3600


def test_target_exclusion_does_not_invalidate_historical_occurrence():
    import pandas as pd

    from road_accidents.data.coverage import CoverageLedger
    from road_accidents.features.weekly import annotate_targets, build_calendar
    from tests.fixtures.prf_factory import coverage_document

    entries = [coverage_document(year, "sha256:" + f"{year:064d}") for year in range(2019, 2027)]
    calendar = build_calendar(CoverageLedger.from_documents(entries, profile="smoke_only"))
    records = pd.DataFrame(
        {
            "record_id": ["one"],
            "source_snapshot_id": [entries[6]["snapshot_id"]],
            "occurred_at": pd.to_datetime(["2025-12-30T12:00:00Z"], utc=True),
            "quality_status": ["valid"],
            "exclusion_reason": [None],
        }
    )
    result = annotate_targets(records, calendar).iloc[0]
    assert result.quality_status == "valid"
    assert not result.target_eligible
    assert result.target_exclusion_reason == "cross_split_week"
