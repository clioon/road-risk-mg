from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from road_accidents.config import PipelineError
from road_accidents.data.coverage import CoverageLedger
from road_accidents.data.schemas import SPLITS, validate_table

TIMEZONE = ZoneInfo("America/Sao_Paulo")


def week_bounds(week_start: date) -> tuple[datetime, datetime]:
    if week_start.weekday() != 0:
        raise PipelineError("A prediction week must begin on Monday.")
    return (
        datetime.combine(week_start, time.min, TIMEZONE),
        datetime.combine(week_start + timedelta(days=7), time.min, TIMEZONE),
    )


def partition_for_week(week_start: date) -> str | None:
    _start, end = week_bounds(week_start)
    for split, (beginning, ending) in SPLITS.items():
        if beginning <= week_start and end.date() <= ending:
            return split
    return None


def build_calendar(coverage: CoverageLedger) -> pd.DataFrame:
    coverage.require_study_coverage()
    first_day = date(2019, 1, 1)
    current = first_day - timedelta(days=first_day.weekday())
    weeks: list[dict[str, Any]] = []
    while current < date(2027, 1, 1):
        ending = current + timedelta(days=7)
        split = partition_for_week(current)
        snapshots = coverage.covering_ids(current, ending)
        reason = None
        if split is None:
            reason = "cross_split_week"
        elif not snapshots:
            reason = "incomplete_week" if coverage.overlaps(current, ending) else "unobserved_week"
        weeks.append(
            {
                "week_start": current,
                "week_end_exclusive": ending,
                "year": current.year,
                "iso_week": current.isocalendar().week,
                "week_index": len(weeks),
                "split": split,
                "eligibility_status": "excluded" if reason else "eligible",
                "exclusion_reason": reason,
                "coverage_snapshot_ids": snapshots,
            }
        )
        current = ending
    frame = pd.DataFrame(weeks)
    validate_table("PredictionWeek", frame)
    return frame


def annotate_targets(records: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    result = records.copy()
    local_dates = pd.to_datetime(result.occurred_at, utc=True).dt.tz_convert(TIMEZONE).dt.date
    result["week_start"] = [
        value - timedelta(days=value.weekday()) if pd.notna(value) else None
        for value in local_dates
    ]
    lookup = calendar.set_index("week_start")
    statuses = result.week_start.map(lookup.eligibility_status)
    result["target_eligible"] = (
        ((result.quality_status == "valid") & statuses.eq("eligible")).fillna(False).astype(bool)
    )
    reasons = result.week_start.map(lookup.exclusion_reason)
    result["target_exclusion_reason"] = [
        None
        if accepted
        else (
            intrinsic
            if quality == "invalid"
            else temporal
            if pd.notna(temporal)
            else "unobserved_week"
        )
        for accepted, intrinsic, quality, temporal in zip(
            result.target_eligible,
            result.exclusion_reason,
            result.quality_status,
            reasons,
            strict=True,
        )
    ]
    validate_table("AccidentRecord", result)
    return result


def labels_for_partition(
    segments: pd.DataFrame,
    records: pd.DataFrame,
    calendar: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    if split not in SPLITS:
        raise PipelineError("Unknown temporal partition.")
    weeks = calendar[(calendar.split == split) & (calendar.eligibility_status == "eligible")][
        ["week_start"]
    ]
    if weeks.empty or segments.empty:
        raise PipelineError("A required temporal population is empty.")
    nodes = pd.DataFrame({"segment_id": sorted(segments.segment_id.tolist())})
    events = records[records.target_eligible][["segment_id", "week_start"]]
    with duckdb.connect(":memory:") as connection:
        connection.register("nodes", nodes)
        connection.register("weeks", weeks)
        connection.register("events", events)
        result = connection.execute("""
            WITH counts AS (
                SELECT segment_id, week_start, count(*) AS event_count
                FROM events GROUP BY segment_id, week_start
            )
            SELECT nodes.segment_id, weeks.week_start,
                   coalesce(counts.event_count, 0)::BIGINT AS event_count,
                   (coalesce(counts.event_count, 0) > 0)::BIGINT AS label
            FROM nodes CROSS JOIN weeks
            LEFT JOIN counts USING (segment_id, week_start)
            ORDER BY week_start, segment_id
        """).fetchdf()
    result["week_start"] = pd.to_datetime(result.week_start).dt.date
    validate_table("SegmentWeekObservation", result)
    return result
