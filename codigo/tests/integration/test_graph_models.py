import numpy as np
import pytest

from tests.fixtures.run_factory import graph_structure, model_frame


@pytest.mark.parametrize("family", ["graphsage", "gatv2"])
def test_forward_preserves_isolated_nodes_and_self_information(family):
    import torch

    from road_accidents.models.gatv2 import GATv2
    from road_accidents.models.graphsage import GraphSAGE

    model = (GraphSAGE if family == "graphsage" else GATv2)(3, hidden_channels=8, dropout=0.0)
    features = torch.ones((4, 3), requires_grad=True)
    result = model(features, torch.empty((2, 0), dtype=torch.long))
    assert result.shape == (4,) and torch.isfinite(result).all()
    result.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


@pytest.mark.parametrize("family", ["graphsage", "gatv2"])
def test_graph_model_train_inference_and_reload_preserve_rows(tmp_path, family):
    from road_accidents.features.preprocessing import SharedPreprocessor
    from road_accidents.graph.dataset import fit_graph_model, score_graph_model
    from road_accidents.models.baselines import load_model, save_model

    features, labels = model_frame()
    validation, _ = model_frame(2024)
    nodes, adjacency = graph_structure()
    processor = SharedPreprocessor().fit(features)
    parameters = {"epochs": 2, "hidden_channels": 8, "batch_size": 2, "dropout": 0.0}
    model = fit_graph_model(
        family, features, processor.transform(features), labels, nodes, adjacency, parameters, 101
    )
    scores = score_graph_model(model, validation, processor.transform(validation), nodes, adjacency)
    assert scores.shape == labels.shape
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    path = tmp_path / "model.pt"
    save_model(model, path, family)
    restored = load_model(path, family)
    np.testing.assert_allclose(
        score_graph_model(restored, validation, processor.transform(validation), nodes, adjacency),
        scores,
        atol=1e-7,
    )
