import geopandas as gpd
from shapely.geometry import LineString, Point


def test_snapping_is_inclusive_and_ties_are_canonical():
    from road_accidents.graph.matching import match_accidents

    segments = gpd.GeoDataFrame(
        {"segment_id": ["b", "a"], "network_version_id": ["network", "network"]},
        geometry=[LineString([(0, 0), (1000, 0)]), LineString([(0, 0), (1000, 0)])],
        crs=5880,
    )
    records = gpd.GeoDataFrame(
        {
            "record_id": ["on", "outside"],
            "accident_id": ["1", "2"],
            "quality_status": ["valid", "valid"],
        },
        geometry=[Point(100, 500), Point(100, 500.000001)],
        crs=5880,
    )
    matched = match_accidents(records, segments).set_index("record_id")
    assert matched.loc["on", "match_status"] == "matched"
    assert matched.loc["on", "distance_m"] == 500.0
    assert matched.loc["on", "candidate_segment_id"] == "a"
    assert matched.loc["on", "tie_count"] == 2
    assert matched.loc["outside", "match_status"] == "rejected_distance"
