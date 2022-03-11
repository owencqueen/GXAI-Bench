import ipdb, os
import random
import torch
import argparse
from gxai_eval.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic, test, train, val
from gxai_eval.gnn_models.node_classification.testing import GIN_1layer, GIN_2layer

# Load BAHouses dataset
bah = torch.load(open('data/BAH.pth', 'rb'))

data = bah.get_graph(use_fixed_split=True)
inhouse = (data.y == 1).nonzero(as_tuple=True)[0]

model = GIN_2layer(16, input_feat = 1, classes = 2)

# Train the model:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

best_f1=0
for epoch in range(1, 1001):
    loss = train(model, optimizer, criterion, data)
    f1, acc, precision, recall, auprc, auroc = val(model, data, get_auc=True)
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join('model', f'model_GIN.pth'))

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {f1:.4f}, Val AUROC: {auroc:.4f}')

# Testing performance
f1, acc, precision, recall, auprc, auroc = test(model, data, get_auc=True)
print(f'Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')
