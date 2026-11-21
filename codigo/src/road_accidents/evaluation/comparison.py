from typing import Any

import pandas as pd

from road_accidents.config import PipelineError
from road_accidents.data.pipeline import table_digest

KEYS = ["week_start", "segment_id"]


def population_keys(frame: pd.DataFrame, *, allow_empty: bool = False) -> pd.DataFrame:
    if not set(KEYS) <= set(frame) or (frame.empty and not allow_empty):
        raise PipelineError("Evaluation population is missing or empty.", 7, "population_mismatch")
    keys = frame[KEYS].copy()
    keys["week_start"] = pd.to_datetime(keys.week_start).dt.date
    if keys.isna().any().any() or keys.duplicated(KEYS).any():
        raise PipelineError(
            "Evaluation population contains missing or duplicate keys.", 7, "population_mismatch"
        )
    return keys.sort_values(KEYS).reset_index(drop=True)


def require_same_population(expected: pd.DataFrame, observed: pd.DataFrame) -> None:
    if not population_keys(expected).equals(population_keys(observed)):
        raise PipelineError(
            "Model observation keys differ within the evaluation period.", 7, "population_mismatch"
        )


def seasonal_keys(first: pd.DataFrame, second: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left, right = population_keys(first), population_keys(second)
    left_weeks = pd.to_datetime(left.week_start).dt.isocalendar().week
    right_weeks = pd.to_datetime(right.week_start).dt.isocalendar().week
    common_weeks = set(left_weeks) & set(right_weeks)
    common_segments = set(left.segment_id) & set(right.segment_id)
    return (
        left[left_weeks.isin(common_weeks) & left.segment_id.isin(common_segments)].reset_index(
            drop=True
        ),
        right[right_weeks.isin(common_weeks) & right.segment_id.isin(common_segments)].reset_index(
            drop=True
        ),
    )


def cohort_manifest(keys: pd.DataFrame, cohort_id: str) -> dict[str, Any]:
    ordered = population_keys(keys, allow_empty=True)
    return {
        "cohort_id": cohort_id,
        "population_hash": table_digest(ordered),
        "observation_count": len(ordered),
        "segment_count": ordered.segment_id.nunique(),
        "week_count": ordered.week_start.nunique(),
        "iso_weeks": sorted(
            int(week) for week in pd.to_datetime(ordered.week_start).dt.isocalendar().week.unique()
        ),
        "status": "available" if not ordered.empty else "no_common_seasonal_population",
    }
