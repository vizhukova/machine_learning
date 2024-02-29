import torch
from torch_geometric.nn import GCNConv

class PyG_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # we define two layers of graph convolutional network
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x: matrix of node features in the form (num_nodes, in_channels)
        # edge_index: matrix of graph connections in the form (2, num_edges)
        
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# for Cora citation database
num_features = 1433
num_classes = 7
num_hidden = 16
model = PyG_GCN(num_features, num_hidden, num_classes)