from typing import cast

from torch import Tensor, nn
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    family = "graphsage"

    def __init__(
        self, input_channels: int, hidden_channels: int = 64, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.architecture = {
            "input_channels": input_channels,
            "hidden_channels": hidden_channels,
            "dropout": dropout,
        }
        self.first = SAGEConv(input_channels, hidden_channels, root_weight=True)
        self.second = SAGEConv(hidden_channels, 1, root_weight=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        hidden = self.dropout(self.first(features, edge_index).relu())
        return cast(Tensor, self.second(hidden, edge_index)).squeeze(-1)
