import os, sys
import argparse, random
import numpy as np
import torch
from tqdm import tqdm

from utils import get_exp_method

from gxai_eval.explainers import PGExplainer
from gxai_eval.utils.performance.load_exp import exp_exists_graph
from gxai_eval.metrics.metrics_graph import graph_exp_acc_graph, graph_exp_faith_graph
from gxai_eval.gnn_models.graph_classification.gin import GIN_3layer

my_base_graphxai = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GXAI-Bench'

parser = argparse.ArgumentParser()
parser.add_argument('--exp_method', required=True, help='name of the explanation method')
parser.add_argument('--faithfulness', action='store_true')
parser.add_argument('--accuracy', action='store_true')
#parser.add_argument('--save_dir', default='./fairness_results/', help='folder for saving results')
args = parser.parse_args()

seed_value=912
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get dataset:
dataset = torch.load('data/ac.pth')
test_inds = dataset.test_index # Index of testing objects

# Set up model:
model = GIN_3layer(14, 32, 2)

# Load model:
# Construct path to model:
mpath = os.path.join('model', 'GIN.pth')
model.load_state_dict(torch.load(mpath))

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)

gea_node = []
gea_edge = []

gef_node = []
gef_edge = []

GEA_PATH = os.path.join('accuracy', 'results')
GEF_PATH = os.path.join('faith', 'results')

add_args = {
    'batch': torch.zeros((1,)).long().to(device)
}

if args.exp_method.lower()=='pgex':
    # Train the PGExplainer
    masked_dataset = [dataset[i][0] for i in dataset.train_index] # Mask to list of only train data
    # fixed for 3-layer conv:
    explainer = PGExplainer(model, explain_graph = True, emb_layer_name='conv3', max_epochs=10, lr=0.1)
    explainer.train_explanation_model(masked_dataset, forward_kwargs=add_args) # Train model on training data


for idx in tqdm(test_inds):

    # Gets graph, ground truth explanation:
    data, gt_exp = dataset[idx]

    data = data.to(device)

    pred_class = model(data.x.to(device), data.edge_index.to(device), **add_args).reshape(-1, 1).argmax(dim=0)

    if pred_class != data.y.item():
        continue

        
    if (args.exp_method.lower() == 'pgex'):
        # Explainer is set before loop
        forward_kwargs = {
            'x': data.x.to(device),
            'edge_index': data.edge_index.to(device),
            'label': pred_class,
            'forward_kwargs': add_args
        }

    else:
        explainer, forward_kwargs = get_exp_method(args.exp_method, model, criterion, pred_class, data, device)

    exp = explainer.get_explanation_graph(**forward_kwargs)

    if args.accuracy:
        # Accuracy:
        _, node_acc, edge_acc = graph_exp_acc_graph(gt_exp, exp, node_thresh_factor = 0.5)

        gea_node.append(node_acc)
        gea_edge.append(edge_acc)

    if args.faithfulness:
        # Faithfulness:
        _, node_faith, edge_faith = graph_exp_faith_graph(exp, data, model, forward_kwargs = add_args)

        gef_node.append(node_faith)
        gef_edge.append(edge_faith)

# Save:
if args.accuracy:
    np.save(open(os.path.join(GEA_PATH, f'{args.exp_method}_GEA_node.npy'), 'wb'), gea_node)
    np.save(open(os.path.join(GEA_PATH, f'{args.exp_method}_GEA_edge.npy'), 'wb'), gea_edge)

if args.faithfulness:
    np.save(open(os.path.join(GEF_PATH, f'{args.exp_method}_GEF_node.npy'), 'wb'), gef_node)
    np.save(open(os.path.join(GEF_PATH, f'{args.exp_method}_GEF_edge.npy'), 'wb'), gef_edge)
