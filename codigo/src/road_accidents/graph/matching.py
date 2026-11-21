import math
from typing import Any

import geopandas as gpd
import pandas as pd

from road_accidents.config import PipelineError
from road_accidents.data.schemas import validate_table


def match_accidents(records: gpd.GeoDataFrame, segments: gpd.GeoDataFrame) -> pd.DataFrame:
    if records.crs is None or segments.crs is None:
        raise PipelineError("Snapping requires an explicit CRS for points and roads.")
    eligible = records[(records.quality_status == "valid") & records.geometry.notna()].reset_index(
        drop=True
    )
    points = eligible.to_crs(5880)
    roads = segments.to_crs(5880).reset_index(drop=True)
    candidates: dict[int, list[tuple[float, str, str]]] = {}
    if not points.empty and not roads.empty:
        indexes, distances = roads.sindex.nearest(
            points.geometry, return_all=True, return_distance=True
        )
        for point_index, road_index, distance in zip(
            indexes[0], indexes[1], distances, strict=True
        ):
            road = roads.iloc[int(road_index)]
            candidates.setdefault(int(point_index), []).append(
                (float(distance), str(road.segment_id), str(road.network_version_id))
            )
    results: list[dict[str, Any]] = []
    for index, record in enumerate(eligible.itertuples()):
        nearby = sorted(candidates.get(index, []))
        distance, identifier, version = nearby[0] if nearby else (math.nan, None, None)
        status = (
            "no_candidate"
            if not nearby
            else "matched"
            if distance <= 500.0
            else "rejected_distance"
        )
        results.append(
            {
                "record_id": record.record_id,
                "accident_id": record.accident_id,
                "network_version_id": version,
                "candidate_segment_id": identifier,
                "segment_id": identifier if status == "matched" else None,
                "distance_m": distance,
                "tie_count": len(nearby),
                "tie_break_rule": "minimum_canonical_segment_id" if len(nearby) > 1 else None,
                "candidate_ids": [item[1] for item in nearby],
                "match_status": status,
            }
        )
    columns = [
        "record_id",
        "accident_id",
        "network_version_id",
        "candidate_segment_id",
        "segment_id",
        "distance_m",
        "tie_count",
        "tie_break_rule",
        "candidate_ids",
        "match_status",
    ]
    frame = pd.DataFrame(results, columns=columns).astype({"distance_m": float})
    validate_table("SpatialMatch", frame)
    return frame


def apply_matches(records: gpd.GeoDataFrame, matches: pd.DataFrame) -> gpd.GeoDataFrame:
    columns = ["record_id", "segment_id", "distance_m", "match_status"]
    combined = records.merge(matches[columns], on="record_id", how="left", validate="one_to_one")
    rejected = (combined.quality_status == "valid") & (combined.match_status != "matched")
    for index in combined.index[rejected]:
        reason = (
            "distance_over_500m"
            if combined.at[index, "match_status"] == "rejected_distance"
            else "no_eligible_segment"
        )
        combined.at[index, "quality_status"] = "invalid"
        combined.at[index, "exclusion_reason"] = reason
        combined.at[index, "target_exclusion_reason"] = reason
        combined.at[index, "exclusion_reasons"] = [reason]
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=records.crs)
