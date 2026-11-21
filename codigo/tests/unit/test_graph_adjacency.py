import json


def test_adjacency_uses_junctions_not_geometric_crossings(synthetic_sources, project_root):
    from road_accidents.data.osm import load_osm_graph
    from road_accidents.graph.adjacency import build_adjacency, node_order
    from road_accidents.graph.segments import build_segments

    inputs = synthetic_sources
    segments = build_segments(
        load_osm_graph(inputs["osm"], inputs["root"]), inputs["osm"].snapshot_id, inputs["boundary"]
    )
    links = build_adjacency(segments)
    way_for_segment = {row.segment_id: row.osm_way_ids[0] for row in segments.itertuples()}
    observed = {
        tuple(sorted((way_for_segment[row.segment_id_low], way_for_segment[row.segment_id_high])))
        for row in links.itertuples()
    }
    expected = json.loads((project_root / "tests/fixtures/network_expected.json").read_text())
    assert observed == {tuple(pair) for pair in expected["adjacent_way_pairs"]}
    assert not (links.segment_id_low == links.segment_id_high).any()
    order = node_order(segments)
    assert order.node_index.tolist() == list(range(len(segments)))
    assert order.segment_id.tolist() == sorted(segments.segment_id)
