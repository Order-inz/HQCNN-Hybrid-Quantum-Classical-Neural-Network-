import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GNNLayer(node_dim, edge_dim, hidden_dim))
            node_dim = hidden_dim
    
    def forward(self, x, edge_index, edge_weights):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weights)
        return x

class GNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(GNNLayer, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index, edge_weights):
        src, dst = edge_index
        messages = torch.cat([x[src], x[dst], edge_weights.unsqueeze(-1)], dim=-1)
        messages = self.edge_mlp(messages)

        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)
        x = self.node_mlp(torch.cat([x, aggregated], dim=-1))
        return x
