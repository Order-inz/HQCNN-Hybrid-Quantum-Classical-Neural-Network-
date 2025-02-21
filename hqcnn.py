import torch
import torch.nn as nn
from models.transformer import TransformerEncoder
from models.gnn import GNN

class HQCNN(nn.Module):
    def __init__(self, num_qubits, hidden_dim, num_heads, num_layers):
        super(HQCNN, self).__init__()
        self.num_qubits = num_qubits
        self.hidden_dim = hidden_dim
        
        # Transformer 模块
        self.transformer = TransformerEncoder(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # GNN 模块
        self.gnn = GNN(
            node_dim=hidden_dim,
            edge_dim=1,  # 边特征维度
            hidden_dim=hidden_dim,
            num_layers=3
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, num_qubits * num_qubits)
    
    def forward(self, node_features, edge_index, edge_weights):
        # Transformer 处理节点特征
        node_embeddings = self.transformer(node_features)
        
        # GNN 处理图结构
        graph_embeddings = self.gnn(node_embeddings, edge_index, edge_weights)
        
        # 输出重构的量子态密度矩阵
        output = self.output_layer(graph_embeddings)
        density_matrix = output.view(-1, self.num_qubits, self.num_qubits)
        return density_matrix
