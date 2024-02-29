import torch
import torch.nn.functional as F
from dgl.nn import GraphConv

class DGL_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DGL_GCN, self).__init__()
       # define two layers of graph convolutional network
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, out_channels)

    def forward(self, g, in_feat):
        # g: a graph object in DGL containing information about node connections
        # in_feat: matrix of node features in the format (num_nodes, in_channels)
        
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# for Cora citation database
num_features = 1433
num_classes = 7
num_hidden = 16
model = DGL_GCN(num_features, num_hidden, num_classes)
