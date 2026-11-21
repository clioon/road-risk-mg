import hashlib
from datetime import date

import pandas as pd
import pytest

from road_accidents.config import PipelineError
from tests.fixtures.prf_factory import coverage_document


@pytest.mark.parametrize("missing", [None, "coverage", "availability"])
def test_excluded_target_week_can_supply_proven_history(missing):
    from road_accidents.data.coverage import CoverageLedger
    from road_accidents.features.temporal import historical_features

    documents = [
        coverage_document(2025, "sha256:" + "a" * 64, available_at="2026-01-01T00:00:00-03:00"),
        coverage_document(2026, "sha256:" + "b" * 64, available_at="2026-01-04T23:59:59-03:00"),
    ]
    documents[-1]["coverage_intervals"][0]["end"] = "2026-01-04"
    documents[-1]["evidence"]["statement"] = "Synthetic January coverage through 2026-01-04"
    documents[-1]["evidence"]["sha256"] = hashlib.sha256(
        documents[-1]["evidence"]["statement"].encode()
    ).hexdigest()
    if missing == "coverage":
        documents[-1]["coverage_intervals"][0]["end"] = "2026-01-03"
    if missing == "availability":
        documents[-1]["available_at"] = None
    records = pd.DataFrame(
        {
            "record_id": ["first", "second", "duplicate", "future"],
            "source_snapshot_id": [
                "sha256:" + "a" * 64,
                "sha256:" + "b" * 64,
                "sha256:" + "b" * 64,
                "sha256:" + "b" * 64,
            ],
            "segment_id": ["segment"] * 4,
            "occurred_at": pd.to_datetime(
                [
                    "2025-12-30T12:00:00Z",
                    "2026-01-03T12:00:00Z",
                    "2026-01-03T12:00:00Z",
                    "2026-01-05T12:00:00Z",
                ],
                utc=True,
            ),
            "available_at": pd.to_datetime(["2026-01-04T23:00:00Z"] * 4, utc=True),
            "quality_status": ["valid", "valid", "invalid", "valid"],
            "target_eligible": [False] * 4,
            "match_status": ["matched"] * 4,
        }
    )
    ledger = CoverageLedger.from_documents(documents, profile="smoke_only")
    features = historical_features(records, ["segment"], date(2026, 1, 5), ledger)
    if missing:
        assert pd.isna(features.iloc[0].accidents_lag_1w)
        assert features.iloc[0].history_missing_reasons
    else:
        assert features.iloc[0].accidents_lag_1w == 2


def test_future_or_label_columns_are_not_predictors():
    from road_accidents.features.temporal import validate_feature_columns

    with pytest.raises(PipelineError) as error:
        validate_feature_columns(["length_m", "causa_acidente", "label"])
    assert error.value.exit_code == 5
