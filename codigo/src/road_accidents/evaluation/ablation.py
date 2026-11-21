from road_accidents.config import PipelineError
from road_accidents.features.temporal import (
    FEATURE_COLUMNS,
    ROAD_FEATURE_COLUMNS,
    TEMPORAL_FEATURE_COLUMNS,
)


def feature_columns(feature_set: str) -> list[str]:
    if feature_set in {"full", "without_topology"}:
        return list(FEATURE_COLUMNS)
    if feature_set == "without_road":
        return [column for column in FEATURE_COLUMNS if column not in ROAD_FEATURE_COLUMNS]
    if feature_set == "without_temporal":
        return [column for column in FEATURE_COLUMNS if column not in TEMPORAL_FEATURE_COLUMNS]
    raise PipelineError("Unknown ablation feature set.", 2, "usage_error")
