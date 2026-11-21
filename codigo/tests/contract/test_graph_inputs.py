import numpy as np
import pytest
from torch_geometric.data import Batch

from road_accidents.config import PipelineError
from tests.fixtures.run_factory import graph_structure, model_frame


def test_weekly_graphs_use_common_matrix_preserve_isolated_nodes_and_batch_boundaries():
    from road_accidents.features.preprocessing import SharedPreprocessor
    from road_accidents.graph.dataset import weekly_graphs

    frame, labels = model_frame()
    nodes, adjacency = graph_structure()
    matrix = SharedPreprocessor().fit(frame).transform(frame)
    graphs = weekly_graphs(frame, matrix, nodes, adjacency, labels=labels)
    assert len(graphs) == 4
    for graph in graphs:
        assert graph.num_nodes == 4 and graph.x.shape[1] == matrix.shape[1]
        np.testing.assert_allclose(graph.x.numpy(), matrix[graph.row_positions.numpy()])
        assert 3 not in graph.edge_index.numpy()
    batch = Batch.from_data_list(graphs)
    assert (batch.batch[batch.edge_index[0]] == batch.batch[batch.edge_index[1]]).all()


def test_permuted_node_table_without_matching_index_is_rejected():
    from road_accidents.features.preprocessing import SharedPreprocessor
    from road_accidents.graph.dataset import weekly_graphs

    frame, _labels = model_frame()
    matrix = SharedPreprocessor().fit(frame).transform(frame)
    nodes, adjacency = graph_structure()
    nodes["segment_id"] = list(reversed(nodes.segment_id))
    with pytest.raises(PipelineError):
        weekly_graphs(frame, matrix, nodes, adjacency)
