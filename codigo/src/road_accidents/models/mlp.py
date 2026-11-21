from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from road_accidents.config import PipelineError


class MLP(nn.Module):
    family = "mlp"

    def __init__(
        self, input_channels: int, hidden_channels: int = 64, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.architecture = {
            "input_channels": input_channels,
            "hidden_channels": hidden_channels,
            "dropout": dropout,
        }
        self.layers = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, features: Tensor, edge_index: Tensor | None = None) -> Tensor:
        return cast(Tensor, self.layers(features)).squeeze(-1)


def fit_mlp(
    matrix: NDArray[np.float32], target: NDArray[np.int64], parameters: dict[str, Any], seed: int
) -> MLP:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = MLP(
        matrix.shape[1], parameters.get("hidden_channels", 64), parameters.get("dropout", 0.1)
    )
    features = torch.from_numpy(matrix)
    labels = torch.from_numpy(target.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    batches = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels),
        batch_size=parameters.get("batch_size", 1024),
        shuffle=True,
        generator=generator,
    )
    weight = torch.tensor(float((target == 0).sum() / (target == 1).sum()))
    objective = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=parameters.get("learning_rate", 0.001),
        weight_decay=parameters.get("weight_decay", 0.0001),
    )
    model.train()
    for _epoch in range(parameters.get("epochs", 30)):
        for inputs, actual in batches:
            optimizer.zero_grad()
            loss = objective(model(inputs), actual)
            if not torch.isfinite(loss):
                raise PipelineError(
                    "Neural training produced a nonfinite loss.", 6, "invalid_training"
                )
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def score_mlp(model: MLP, matrix: NDArray[np.float32]) -> NDArray[np.float64]:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(matrix), 4096):
            chunks.append(
                torch.sigmoid(model(torch.from_numpy(matrix[start : start + 4096]))).cpu().numpy()
            )
    return np.asarray(np.concatenate(chunks), dtype=np.float64)


def save_neural(model: Any, path: Path) -> None:
    with path.open("xb") as output:
        torch.save(
            {
                "family": model.family,
                "architecture": model.architecture,
                "state_dict": model.state_dict(),
            },
            output,
        )


def load_neural(path: Path, family: str) -> Any:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["family"] != family or family not in {"mlp", "graphsage", "gatv2"}:
        raise PipelineError("Neural model family does not match the manifest.")
    if family == "mlp":
        model: Any = MLP(**checkpoint["architecture"])
    else:
        from road_accidents.graph.dataset import create_graph_model

        architecture = dict(checkpoint["architecture"])
        input_channels = architecture.pop("input_channels")
        model = create_graph_model(family, input_channels, architecture)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model
