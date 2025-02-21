import torch
import torch.optim as optim
from models.hqcnn import HQCNN
from utils.loss import fidelity_loss, physical_constraint_loss
from data.generate_data import generate_quantum_data

num_qubits = 4
hidden_dim = 256
num_heads = 4
num_layers = 3
batch_size = 32
epochs = 100

model = HQCNN(num_qubits, hidden_dim, num_heads, num_layers)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

train_data = generate_quantum_data(num_samples=1000, num_qubits=num_qubits)

for epoch in range(epochs):
    for batch in train_data:
        node_features, edge_index, edge_weights, target = batch
        pred = model(node_features, edge_index, edge_weights)
        loss = fidelity_loss(pred, target) + physical_constraint_loss(pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
