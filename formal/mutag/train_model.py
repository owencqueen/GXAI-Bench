import ipdb, os
import random
import torch
import argparse
from gxai_eval.gnn_models.graph_classification import train, test
from gxai_eval.gnn_models.graph_classification.gcn import GCN_2layer, GCN_3layer
from gxai_eval.gnn_models.graph_classification.gin import GIN_2layer, GIN_3layer

dataset = torch.load('data/mutag.pth')
train_loader, _ = dataset.get_train_loader(batch_size = 32)
test_loader, _ = dataset.get_test_loader()
val_loader, _ = dataset.get_val_loader()

model = GIN_3layer(7, 32, 2)

# Train the model:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

best_f1=0
for epoch in range(1, 101):
    train(model, optimizer, criterion, train_loader)
    #f1, prec, rec, auprc, auroc = test(model, test_loader)
    f1, prec, rec, auprc, auroc = test(model, val_loader)

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join('model', 'GIN.pth'))

    print(f'Epoch: {epoch:03d}, Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')

f1, precision, recall, auprc, auroc = test(model, test_loader)
print(f'Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')
