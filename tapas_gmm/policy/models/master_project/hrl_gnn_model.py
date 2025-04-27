# Load a dataset:
dataset = Planetoid(root, name="Cora")

# Create a mini-batch loader:
loader = NeighborLoader(dataset[0], num_neighbors=[25, 10])


# Create your GNN model:
class GNN(torch.nn.Module):
    def __init__(self):
        # Choose between different GNN building blocks:
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


# Train you GNN model:
for data in loader:
    y_hat = model(data.x, data.edge_index)
    loss = criterion(y_hat, data.y)
