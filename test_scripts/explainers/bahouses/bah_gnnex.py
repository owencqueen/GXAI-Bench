import random
import torch
import matplotlib.pyplot as plt

from gxai_eval.explainers import GNNExplainer
from gxai_eval.gnn_models.node_classification.testing import GIN_2layer, test, train 
from gxai_eval.datasets import BAHouses

# Load dataset:
# Smaller graph is shown to work well with model accuracy, graph properties
bah = BAHouses(
    n = 300 + 30 * 5,
    m = 1,
    model_layers= 2,
    num_houses = 30
)
data = bah.get_graph(use_fixed_split=True)
inhouse = (data.y == 1).nonzero(as_tuple=True)[0]

# Test on 3-layer basic GCN, 16 hidden dim:
model = GIN_2layer(16, input_feat = 3, classes = 2)

# Train the model:
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 101):
    loss = train(model, optimizer, criterion, data)
    f1, acc, precision, recall, auroc, auprc = test(model, data, get_auc = True)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')

# Get prediction of a node in the 2-house class:
model.eval()
random.seed(1345)
node_idx = random.choice(inhouse).item()
pred = model(data.x, data.edge_index)[node_idx, :].reshape(-1, 1)
pred_class = pred.argmax(dim=0).item()

print('GROUND TRUTH LABEL: \t {}'.format(data.y[node_idx].item()))
print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=0).item()))

# Set up plotting:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 6))

# Ground truth plot:
gt_exp = bah.explanations[node_idx]
gt_exp.context_draw(num_hops = bah.num_hops, graph_data = data, additional_hops = 1, heat_by_exp = True, ax = ax1)
ax1.set_title('Ground Truth')

# Run Explainer ----------------------------------------------------------
gnnex = GNNExplainer(model)
exp = gnnex.get_explanation_node(
                    x = data.x, 
                    y = data.y, 
                    node_idx = node_idx, 
                    edge_index = data.edge_index)
# ------------------------------------------------------------------------

# Grad-CAM plot:
exp.context_draw(num_hops = bah.num_hops, graph_data = data, additional_hops = 1, heat_by_exp = True, ax = ax2)
ax2.set_title('GNNExplainer')

# More plotting details:
ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ax1.text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(data.y[node_idx].item()))
ax1.text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(pred.argmax(dim=0).item()))

plt.show()