import pandas as pd
import pytest

from road_accidents.config import PipelineError
from tests.fixtures.run_factory import model_frame


def test_population_mismatch_is_rejected_before_metrics():
    from road_accidents.evaluation.comparison import require_same_population

    first, _ = model_frame(2025)
    with pytest.raises(PipelineError):
        require_same_population(first, first.iloc[:-1])
    with pytest.raises(PipelineError):
        require_same_population(first, pd.concat([first, first.iloc[:1]]))


def test_seasonal_cohort_uses_iso_weeks_and_segments_only():
    from road_accidents.evaluation.comparison import seasonal_keys

    first, _ = model_frame(2025, weeks=4)
    second, _ = model_frame(2026, weeks=2)
    left, right = seasonal_keys(first, second)
    assert set(pd.to_datetime(left.week_start).dt.isocalendar().week) == set(
        pd.to_datetime(right.week_start).dt.isocalendar().week
    )
    assert len(left) == len(right) > 0
    assert set(left.segment_id) == set(right.segment_id)
