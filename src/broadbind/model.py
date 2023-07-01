from torch.nn import Module, Linear, ReLU, Dropout
from torch_geometric.nn import XConv, global_mean_pool


class GNN(Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        spatial_dimension: int,
        k_neighbors: int,
        hidden_channels: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.conv = XConv(
            in_channels=input_dimension,
            out_channels=hidden_dimension,
            dim=spatial_dimension,
            kernel_size=k_neighbors,
            hidden_channels=hidden_channels,
        )
        self.relu = ReLU()
        self.linear = Linear(
            in_features=hidden_dimension, out_features=output_dimension
        )
        self.dropout = Dropout(p=dropout_rate)

    def forward(self, data):
        # TODO try not using sequential
        x = self.conv(x=data.x, pos=data.pos, batch=data.batch)
        x = self.relu(x)
        x = global_mean_pool(x=x, batch=data.batch)
        return self.linear(x)
