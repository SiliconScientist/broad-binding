from torch.nn import Module, Linear, ReLU, Sequential
from torch_geometric.nn import PointTransformerConv, knn_graph


class GNN(Module):
    def __init__(
        self, input_dimension, hidden_dimension, output_dimension, k_neighbors
    ) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.graph_conv = PointTransformerConv()
        self.linear = Linear(in_features=input_dimension, out_features=hidden_dimension)

    def forward(self, data):
        edge_index = knn_graph(
            data.pos, k=self.k_neighbors, batch=data.batch, loop=True
        )
        representation = self.graph_conv(x=data.x, pos=data.pos, edge_index=edge_index)
        return self.linear(representation)
