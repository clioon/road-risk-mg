from typing import cast

from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv


class GATv2(nn.Module):
    family = "gatv2"

    def __init__(
        self, input_channels: int, hidden_channels: int = 64, dropout: float = 0.1, heads: int = 2
    ) -> None:
        super().__init__()
        self.architecture = {
            "input_channels": input_channels,
            "hidden_channels": hidden_channels,
            "dropout": dropout,
            "heads": heads,
        }
        self.first = GATv2Conv(
            input_channels,
            hidden_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )
        self.second = GATv2Conv(hidden_channels, 1, heads=1, concat=False, add_self_loops=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        hidden = self.dropout(self.first(features, edge_index).relu())
        return cast(Tensor, self.second(hidden, edge_index)).squeeze(-1)
