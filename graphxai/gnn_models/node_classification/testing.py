import torch
from torch_geometric.nn import GCNConv, GINConv, BatchNorm

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class GCN_1layer(torch.nn.Module):
    def __init__(self, input_feat, classes):
        super(GCN_1layer, self).__init__()
        self.conv1 = GCNConv(input_feat, classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

class GCN_2layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GCN_2layer, self).__init__()
        self.conv1 = GCNConv(input_feat, hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

class GCN_3layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GCN_3layer, self).__init__()
        self.gcn1 = GCNConv(input_feat, hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.gcn3 = GCNConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.gcn2(x, edge_index)
        x = self.batchnorm2(x)
        x = x.relu()
        x = self.gcn3(x, edge_index)
        return x

class GCN_3layer_basic(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GCN_3layer_basic, self).__init__()
        self.gcn1 = GCNConv(input_feat, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.gcn3 = GCNConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = x.relu()
        x = self.gcn2(x, edge_index)
        x = x.relu()
        x = self.gcn3(x, edge_index)
        return x

class GIN_1layer(torch.nn.Module):
    def __init__(self, input_feat, classes):
        super(GIN_1layer, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, classes)
        self.conv1 = GINConv(self.mlp_gin1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

class GIN_2layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GIN_2layer, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, classes)
        self.gin2 = GINConv(self.mlp_gin2)

    def forward(self, x, edge_index):
        x = self.gin1(x, edge_index)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.gin2(x, edge_index)
        return x

class GIN_3layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GIN_3layer, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin2 = GINConv(self.mlp_gin2)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.mlp_gin3 = torch.nn.Linear(hidden_channels, classes)
        self.gin3 = GINConv(self.mlp_gin3)

    def forward(self, x, edge_index):
        x = self.gin1(x, edge_index)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.gin2(x, edge_index)
        x = self.batchnorm2(x)
        x = x.relu()
        x = self.gin3(x, edge_index)
        return x

def train(model, optimizer,
          criterion, data):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss
    
def test(model, data, num_classes = 2):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.

    acc = accuracy_score(data.y[data.test_mask].tolist(), pred[data.test_mask].tolist())
    if num_classes == 2:
        test_score = f1_score(data.y[data.test_mask].tolist(), pred[data.test_mask].tolist())
        precision = precision_score(data.y[data.test_mask].tolist(), pred[data.test_mask].tolist())
        recall = recall_score(data.y[data.test_mask].tolist(), pred[data.test_mask].tolist())
        return test_score, acc, precision, recall
    
    return acc