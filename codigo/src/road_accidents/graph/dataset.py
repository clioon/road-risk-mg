from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from road_accidents.config import PipelineError


class WeeklyGraphs(Dataset[Data]):
    def __init__(
        self,
        features: pd.DataFrame,
        matrix: NDArray[np.float32],
        nodes: pd.DataFrame,
        adjacency: pd.DataFrame,
        labels: NDArray[np.int64] | None = None,
        without_topology: bool = False,
    ) -> None:
        self.nodes = nodes.segment_id.tolist()
        if self.nodes != sorted(set(self.nodes)) or nodes.node_index.tolist() != list(
            range(len(nodes))
        ):
            raise PipelineError("Graph node order is not canonical.", 7, "node_order_mismatch")
        if (
            len(matrix) != len(features)
            or matrix.ndim != 2
            or not np.isfinite(matrix).all()
            or features.duplicated(["segment_id", "week_start"]).any()
        ):
            raise PipelineError(
                "Graph features do not match observation keys.", 7, "graph_input_mismatch"
            )
        if labels is not None and (
            labels.shape != (len(features),) or not np.isin(labels, [0, 1]).all()
        ):
            raise PipelineError("Graph labels are not aligned binary targets.")
        self.matrix, self.labels = matrix, labels
        positions = features[["segment_id", "week_start"]].reset_index(drop=True)
        positions["row_position"] = np.arange(len(features))
        self.weeks: list[tuple[str, NDArray[np.int64]]] = []
        for week, frame in positions.groupby("week_start", sort=True):
            if set(frame.segment_id) != set(self.nodes) or len(frame) != len(self.nodes):
                raise PipelineError(
                    "Every graph week must contain every canonical segment.",
                    7,
                    "population_mismatch",
                )
            ordered = (
                frame.set_index("segment_id")
                .loc[self.nodes, "row_position"]
                .to_numpy(dtype=np.int64)
            )
            self.weeks.append((str(week), ordered))
        lookup = dict(zip(self.nodes, range(len(self.nodes)), strict=True))
        pairs: set[tuple[int, int]] = set()
        for row in adjacency.itertuples():
            if (
                row.segment_id_low not in lookup
                or row.segment_id_high not in lookup
                or row.segment_id_low == row.segment_id_high
            ):
                raise PipelineError("Graph adjacency references invalid segments.")
            first, second = lookup[row.segment_id_low], lookup[row.segment_id_high]
            pairs.update(((first, second), (second, first)))
        self.edge_index = (
            torch.tensor(sorted(pairs), dtype=torch.long).T.contiguous()
            if pairs and not without_topology
            else torch.empty((2, 0), dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.weeks)

    def __getitem__(self, index: int) -> Data:
        week, positions = self.weeks[index]
        graph = Data(
            x=torch.from_numpy(np.ascontiguousarray(self.matrix[positions])),
            edge_index=self.edge_index,
            num_nodes=len(self.nodes),
            row_positions=torch.from_numpy(positions.copy()),
            week_start=week,
        )
        if self.labels is not None:
            graph.y = torch.from_numpy(self.labels[positions].astype(np.float32))
        return graph


def weekly_graphs(
    features: pd.DataFrame,
    matrix: NDArray[np.float32],
    nodes: pd.DataFrame,
    adjacency: pd.DataFrame,
    *,
    labels: NDArray[np.int64] | None = None,
    without_topology: bool = False,
) -> WeeklyGraphs:
    return WeeklyGraphs(features, matrix, nodes, adjacency, labels, without_topology)


def create_graph_model(family: str, input_channels: int, parameters: dict[str, Any]) -> Any:
    from road_accidents.models.gatv2 import GATv2
    from road_accidents.models.graphsage import GraphSAGE

    options = {
        "input_channels": input_channels,
        "hidden_channels": parameters.get("hidden_channels", 64),
        "dropout": parameters.get("dropout", 0.1),
    }
    if family == "graphsage":
        return GraphSAGE(**options)
    if family == "gatv2":
        return GATv2(**options, heads=parameters.get("heads", 2))
    raise PipelineError("Unknown graph model family.", 2, "usage_error")


def fit_graph_model(
    family: str,
    features: pd.DataFrame,
    matrix: NDArray[np.float32],
    labels: NDArray[np.int64],
    nodes: pd.DataFrame,
    adjacency: pd.DataFrame,
    parameters: dict[str, Any],
    seed: int,
    *,
    without_topology: bool = False,
) -> Any:
    if set(np.unique(labels)) != {0, 1}:
        raise PipelineError("Graph training requires both classes.", 6, "single_class_training")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    graphs = weekly_graphs(
        features, matrix, nodes, adjacency, labels=labels, without_topology=without_topology
    )
    batches = DataLoader(
        graphs,
        batch_size=parameters.get("batch_size", 4),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = create_graph_model(family, matrix.shape[1], parameters)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=parameters.get("learning_rate", 0.001),
        weight_decay=parameters.get("weight_decay", 0.0001),
    )
    objective = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float((labels == 0).sum() / (labels == 1).sum()))
    )
    model.train()
    for _epoch in range(parameters.get("epochs", 30)):
        for batch in batches:
            optimizer.zero_grad()
            loss = objective(model(batch.x, batch.edge_index), batch.y)
            if not torch.isfinite(loss):
                raise PipelineError("Graph training loss is nonfinite.", 6, "invalid_training")
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def score_graph_model(
    model: Any,
    features: pd.DataFrame,
    matrix: NDArray[np.float32],
    nodes: pd.DataFrame,
    adjacency: pd.DataFrame,
    *,
    without_topology: bool = False,
) -> NDArray[np.float64]:
    graphs = weekly_graphs(features, matrix, nodes, adjacency, without_topology=without_topology)
    scores = np.full(len(features), np.nan, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(graphs, batch_size=4, shuffle=False):
            predicted = torch.sigmoid(model(batch.x, batch.edge_index)).cpu().numpy()
            scores[batch.row_positions.cpu().numpy()] = predicted
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise PipelineError(
            "Graph inference did not cover every observation.", 7, "population_mismatch"
        )
    return scores
