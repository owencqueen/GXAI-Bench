import tqdm
import ipdb
import argparse, sys; sys.path.append('../..')
import random as rand
import torch
from metrics import *
from gxai_eval.explainers import *
from gxai_eval.utils.performance.load_exp import exp_exists
from gxai_eval.gnn_models.node_classification.testing import GIN_2layer, GIN_3layer_basic, GCN_3layer_basic, GSAGE_3layer

#my_base_graphxai = '/home/owq978/GraphXAI'
my_base_graphxai = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GXAI-Bench'

def get_exp_method(method, model, criterion, bah, node_idx, pred_class):
    method = method.lower()
    if method=='gnnex':
        exp_method = GNNExplainer(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='grad':
        exp_method = GradExplainer(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='cam':
        exp_method = CAM(model, activation = lambda x: torch.argmax(x, dim=1))
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='gcam':
        exp_method = GradCAM(model, criterion = criterion)
        forward_kwargs={'x':data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device),
                        'average_variant': [True]}
    elif method=='gbp':
        exp_method = GuidedBP(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='glime':
        exp_method = GraphLIME(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='ig':
        exp_method = IntegratedGradExplainer(model, criterion = criterion)
        forward_kwargs = {'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'node_idx': int(node_idx),
                        'label': pred_class}
    elif method=='glrp':
        exp_method = GNN_LRP(model)
        forward_kwargs={'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'node_idx': node_idx,
                        'label': pred_class,
                        'edge_aggregator':torch.sum}
    elif method=='pgmex':
        exp_method=PGMExplainer(model, explain_graph=False, p_threshold=0.1)
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'top_k_nodes': 10}
    elif method=='pgex':
        exp_method=PGEX
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'label': pred_class}
    elif method=='rand':
        exp_method = RandomExplainer(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='subx':
        exp_method = SubgraphX(model, reward_method = 'gnn_score', num_hops = bah.num_hops, rollout=5)
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'label': pred_class,
                        'max_nodes': 10}
    else:
        OSError('Invalid argument!!')
    return exp_method, forward_kwargs

def get_model():
    model = GIN_2layer(16, input_feat=1, classes=2)
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--exp_method', required=True, help='name of the explanation method')
parser.add_argument('--save_dir', default='./results/', help='folder for saving results')
parser.add_argument('--ignore_training', action='store_true', help='Ignores model training for PGEX')
args = parser.parse_args()

seed_value=912
rand.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Smaller graph is shown to work well with model accuracy, graph properties
bah = torch.load(open(os.path.join(my_base_graphxai, 'formal/bahouses/data/BAH.pth'), 'rb'))
data = bah.get_graph(use_fixed_split=True)
test_set = data.test_mask.nonzero(as_tuple=True)[0]

# Test on 2-layer basic GIN, 16 hidden dim:
model = get_model().to(device)

# Get prediction of a node in the 2-house class:
mpath = os.path.join(my_base_graphxai, 'formal/bahouses/model/model_GIN.pth')
model.load_state_dict(torch.load(mpath))

# Pre-train PGEX before running:
if args.exp_method.lower() == 'pgex':
    PGEX=PGExplainer(model, emb_layer_name = 'gin3', max_epochs=10, lr=0.1)
    PGEX.train_explanation_model(data.to(device))

gea_feat = []
gea_node = []
gea_edge = []

# Get predictions
pred = model(data.x.to(device), data.edge_index.to(device))
criterion = torch.nn.CrossEntropyLoss().to(device)

# Cached graphs:
G = to_networkx_conv(data, to_undirected=True)

for node_idx in tqdm.tqdm(test_set):

    node_idx = node_idx.item()

    # Get predictions
    pred_class = pred[node_idx, :].reshape(-1, 1).argmax(dim=0)

    if pred_class != data.y[node_idx]:
        # Don't evaluate if the prediction is incorrect
        continue

    # Get explanation method
    explainer, forward_kwargs = get_exp_method(args.exp_method, model, criterion, bah, node_idx, pred_class)

    exp = explainer.get_explanation_node(**forward_kwargs)

    # Calculate metrics
    feat, node, edge = graph_exp_acc(gt_exp = bah.explanations[node_idx], generated_exp=exp)
    gea_feat.append(feat)
    gea_node.append(node)
    gea_edge.append(edge)


############################
# Saving the metric values
# save_dir='./results_homophily/'
np.save(os.path.join(args.save_dir, f'{args.exp_method}_GEA_feat.npy'), gea_feat)
np.save(os.path.join(args.save_dir, f'{args.exp_method}_GEA_node.npy'), gea_node)
np.save(os.path.join(args.save_dir, f'{args.exp_method}_GEA_edge.npy'), gea_edge)
