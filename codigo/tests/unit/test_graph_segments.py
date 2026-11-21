import json

import pandas as pd


def test_physical_segments_preserve_parallel_isolated_and_closed_roads(
    synthetic_sources, project_root
):
    from road_accidents.data.osm import load_osm_graph
    from road_accidents.graph.segments import build_segments

    inputs = synthetic_sources
    graph = load_osm_graph(inputs["osm"], inputs["root"])
    segments = build_segments(graph, inputs["osm"].snapshot_id, inputs["boundary"])
    expected = json.loads((project_root / "tests/fixtures/network_expected.json").read_text())
    assert len(segments) == len(expected["federal_way_ids"])
    assert segments.segment_id.is_unique
    assert segments.length_m.gt(0).all()
    assert segments.crs.to_epsg() == 4326
    way_ids = {way_id for values in segments.osm_way_ids for way_id in values}
    assert way_ids == set(expected["federal_way_ids"])
    closed = segments[segments.osm_way_ids.map(lambda values: 107 in values)].iloc[0]
    assert closed.chord_length_m == 0
    assert pd.isna(closed.sinuosity)
    assert "sinuosity" in closed.missing_attributes
    second = build_segments(graph, inputs["osm"].snapshot_id, inputs["boundary"])
    assert list(second.segment_id) == list(segments.segment_id)


def test_missing_road_attributes_remain_missing(synthetic_sources):
    from road_accidents.data.osm import load_osm_graph
    from road_accidents.graph.segments import build_segments

    inputs = synthetic_sources
    segments = build_segments(
        load_osm_graph(inputs["osm"], inputs["root"]), inputs["osm"].snapshot_id, inputs["boundary"]
    )
    missing = segments[segments.osm_way_ids.map(lambda values: 103 in values)].iloc[0]
    assert pd.isna(missing.lanes) and pd.isna(missing.maxspeed_kph)
