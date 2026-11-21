import numpy as np
import pytest

from tests.fixtures.run_factory import graph_structure, model_frame


def test_feature_ablations_remove_only_the_declared_category():
    from road_accidents.evaluation.ablation import feature_columns
    from road_accidents.features.temporal import (
        FEATURE_COLUMNS,
        ROAD_FEATURE_COLUMNS,
        TEMPORAL_FEATURE_COLUMNS,
    )

    assert feature_columns("full") == list(FEATURE_COLUMNS)
    assert not set(feature_columns("without_road")) & set(ROAD_FEATURE_COLUMNS)
    assert not set(feature_columns("without_temporal")) & set(TEMPORAL_FEATURE_COLUMNS)
    assert feature_columns("without_topology") == list(FEATURE_COLUMNS)


@pytest.mark.parametrize("family", ["graphsage", "gatv2"])
def test_without_topology_predictions_ignore_other_nodes(family):
    import torch

    from road_accidents.graph.dataset import create_graph_model, score_graph_model

    frame, _ = model_frame(2024)
    nodes, adjacency = graph_structure()
    torch.manual_seed(101)
    matrix = np.ones((len(frame), 3), dtype=np.float32)
    model = create_graph_model(family, 3, {"hidden_channels": 8, "dropout": 0.0})
    original = score_graph_model(model, frame, matrix, nodes, adjacency, without_topology=True)
    matrix[frame.segment_id != "segment-0"] = 100
    perturbed = score_graph_model(model, frame, matrix, nodes, adjacency, without_topology=True)
    np.testing.assert_allclose(
        original[frame.segment_id == "segment-0"],
        perturbed[frame.segment_id == "segment-0"],
        atol=1e-7,
    )
