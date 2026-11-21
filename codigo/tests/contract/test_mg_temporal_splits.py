def test_schema_rejects_old_validity_semantics():
    import pandas as pd
    import pytest

    from road_accidents.config import PipelineError
    from road_accidents.data.schemas import validate_table

    frame = pd.DataFrame(
        {
            "record_id": ["1"],
            "source_snapshot_id": ["snapshot"],
            "quality_status": ["accepted"],
            "target_eligible": [True],
            "exclusion_reason": [None],
            "target_exclusion_reason": [None],
        }
    )
    with pytest.raises(PipelineError):
        validate_table("AccidentRecord", frame)


def test_dataset_config_cannot_disable_integrity():
    import pytest
    from pydantic import ValidationError

    from road_accidents.data.schemas import DatasetConfig

    with pytest.raises(ValidationError):
        DatasetConfig(privacy={"require_passed_snapshot": False})
    with pytest.raises(ValidationError):
        DatasetConfig(splits={"train": "2019-01-01/2026-01-01"})


def test_curated_rows_retain_exclusions_and_only_mg_is_valid(tmp_path, project_root):
    import geopandas as gpd

    from road_accidents.data.prf import curate_prf, ingest_prf
    from tests.fixtures.prf_factory import prf_bytes

    snapshot = ingest_prf(
        prf_bytes(2026),
        tmp_path,
        year=2026,
        source_url="https://example.org/2026.csv",
        profile="smoke_only",
    )
    boundary = gpd.read_file(project_root / "tests/fixtures/mg_boundary.geojson")
    rows = curate_prf(snapshot, tmp_path, boundary)
    assert len(rows) == 8
    valid = rows[rows.quality_status == "valid"]
    assert set(valid.state_code) == {"MG"}
    reasons = set(rows.exclusion_reason.dropna())
    assert {"outside_scope", "duplicate", "zero_coordinates", "missing_datetime"} <= reasons
    assert rows.record_id.is_unique


def test_four_partitions_are_fixed():
    from datetime import date

    from road_accidents.features.weekly import partition_for_week

    assert partition_for_week(date(2019, 1, 7)) == "train"
    assert partition_for_week(date(2023, 12, 25)) == "train"
    assert partition_for_week(date(2024, 1, 1)) == "validation"
    assert partition_for_week(date(2025, 1, 6)) == "test_2025"
    assert partition_for_week(date(2026, 1, 5)) == "test_2026"
