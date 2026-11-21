import json
import math
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from road_accidents.config import PipelineError
from road_accidents.data.coverage import CoverageLedger
from road_accidents.data.schemas import OSM_EFFECTIVE_AT
from road_accidents.features.weekly import TIMEZONE, week_bounds

ROAD_FEATURE_COLUMNS = (
    "length_m",
    "road_type",
    "lanes",
    "maxspeed_kph",
    "sinuosity",
    "lanes_missing",
    "maxspeed_kph_missing",
    "sinuosity_missing",
)
TEMPORAL_FEATURE_COLUMNS = (
    "calendar_year",
    "calendar_month",
    "calendar_iso_week",
    "week_sin",
    "week_cos",
    "accidents_lag_1w",
    "accidents_lag_4w",
    "accidents_lag_13w",
    "accidents_lag_52w",
    "history_available_weeks",
)
FEATURE_COLUMNS = ROAD_FEATURE_COLUMNS + TEMPORAL_FEATURE_COLUMNS


def validate_feature_columns(columns: Iterable[str]) -> None:
    if not set(columns) <= set(FEATURE_COLUMNS):
        raise PipelineError(
            "Predictors contain undeclared, topological or post-event fields.",
            5,
            "temporal_integrity",
        )


def historical_features(
    records: pd.DataFrame,
    segment_ids: list[str],
    week_start: date,
    coverage: CoverageLedger,
) -> pd.DataFrame:
    forecast, _ending = week_bounds(week_start)
    frame = pd.DataFrame({"segment_id": sorted(segment_ids)})
    reasons: dict[str, str] = {}
    sources_used: set[str] = set()
    latest: datetime | None = None
    valid = records[
        (records.quality_status == "valid") & (records.match_status == "matched")
    ].copy()
    valid["occurred_at"] = pd.to_datetime(valid.occurred_at, utc=True)
    supplied = pd.to_datetime(valid.available_at, utc=True)
    availability = {entry.snapshot_id: entry.available_at for entry in coverage.entries}
    source_availability = pd.to_datetime(valid.source_snapshot_id.map(availability), utc=True)
    valid["effective_available_at"] = supplied.where(
        supplied.notna() & (supplied > source_availability), source_availability
    )
    for window in (1, 4, 13, 52):
        beginning = week_start - timedelta(weeks=window)
        ids = coverage.covering_ids(beginning, week_start, available_before=forecast)
        field = f"accidents_lag_{window}w"
        if not ids:
            frame[field] = float("nan")
            reasons[str(window)] = "coverage_or_publication_not_proven"
            continue
        start = datetime.combine(beginning, datetime.min.time(), TIMEZONE)
        candidates = valid[
            (valid.occurred_at >= start)
            & (valid.occurred_at < forecast)
            & valid.source_snapshot_id.isin(ids)
        ]
        if (
            candidates.effective_available_at.isna().any()
            or (candidates.effective_available_at >= forecast).any()
        ):
            frame[field] = float("nan")
            reasons[str(window)] = "record_not_available_before_prediction"
            continue
        identity = "accident_id" if "accident_id" in candidates else "record_id"
        candidates = candidates.sort_values(
            ["effective_available_at", "record_id"]
        ).drop_duplicates(identity, keep="last")
        counts = candidates.groupby("segment_id").size()
        frame[field] = frame.segment_id.map(counts).fillna(0).astype(float)
        sources_used.update(ids)
        times = [coverage.for_snapshot(identifier).available_at for identifier in ids]
        times.extend(value.to_pydatetime() for value in candidates.effective_available_at)
        for available in times:
            if available is not None and (latest is None or available > latest):
                latest = available
    available_weeks = 0
    for window in range(1, 53):
        if not coverage.covering_ids(
            week_start - timedelta(weeks=window), week_start, available_before=forecast
        ):
            break
        available_weeks = window
    frame["history_available_weeks"] = available_weeks
    frame["history_missing_reasons"] = [dict(reasons) for _identifier in frame.segment_id]
    frame["history_snapshot_ids"] = [sorted(sources_used) for _identifier in frame.segment_id]
    frame["history_available_at"] = latest
    return frame


def features_for_partition(
    segments: pd.DataFrame,
    records: pd.DataFrame,
    calendar: pd.DataFrame,
    coverage: CoverageLedger,
    split: str,
) -> pd.DataFrame:
    columns = ["segment_id", "length_m", "road_type", "lanes", "maxspeed_kph", "sinuosity"]
    roads = pd.DataFrame(segments[columns]).sort_values("segment_id").reset_index(drop=True)
    for field in ("lanes", "maxspeed_kph", "sinuosity"):
        roads[field + "_missing"] = roads[field].isna().astype(int)
    network_version = str(segments.network_version_id.iloc[0])
    weeks = calendar[(calendar.split == split) & (calendar.eligibility_status == "eligible")]
    batches = []
    for week in weeks.itertuples():
        history = historical_features(records, roads.segment_id.tolist(), week.week_start, coverage)
        batch = roads.merge(history, on="segment_id", validate="one_to_one")
        batch["week_start"] = week.week_start
        batch["calendar_year"] = week.week_start.year
        batch["calendar_month"] = week.week_start.month
        batch["calendar_iso_week"] = week.iso_week
        batch["week_sin"] = math.sin(2 * math.pi * week.iso_week / 53)
        batch["week_cos"] = math.cos(2 * math.pi * week.iso_week / 53)
        available: Any = history.history_available_at.iloc[0]
        available_at = max(OSM_EFFECTIVE_AT, available) if pd.notna(available) else OSM_EFFECTIVE_AT
        batch["feature_available_at"] = pd.Timestamp(available_at).tz_convert(UTC)
        if (batch.feature_available_at >= week_bounds(week.week_start)[0].astimezone(UTC)).any():
            raise PipelineError(
                "A feature is not available before prediction.", 5, "temporal_integrity"
            )
        batch["feature_provenance"] = json.dumps(
            {
                "network_version_id": network_version,
                "history_snapshot_ids": history.history_snapshot_ids.iloc[0],
                "calendar": "known_in_advance",
                "availability_policy": "documented_before_prediction",
            },
            sort_keys=True,
        )
        batch["history_missing_reasons"] = json.dumps(
            history.history_missing_reasons.iloc[0], sort_keys=True
        )
        batch = batch.drop(columns=["history_snapshot_ids", "history_available_at"])
        batches.append(batch)
    if not batches:
        raise PipelineError("A required feature population is empty.")
    validate_feature_columns(FEATURE_COLUMNS)
    return (
        pd.concat(batches, ignore_index=True)
        .sort_values(["week_start", "segment_id"])
        .reset_index(drop=True)
    )
