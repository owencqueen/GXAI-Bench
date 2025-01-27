import tqdm
import ipdb
import argparse, sys
import random as rand
import torch
from metrics import *
from gxai_eval.explainers import *
from gxai_eval.datasets  import load_ShapeGraph
from gxai_eval.datasets.shape_graph import ShapeGraph
from gxai_eval.gnn_models.node_classification.testing import GIN_3layer_basic


def get_exp_method(method, model, criterion, bah, node_idx, pred_class):
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
    elif method == 'cam':
        act = lambda x: torch.argmax(x, dim=1)
        exp_method = CAM(model, activation=act)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'label': pred_class,
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
    elif method=='rand':
        exp_method = RandomExplainer(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='subx':
        exp_method = SubgraphX(model, reward_method = 'gnn_score', num_hops = bah.model_layers)
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'label': pred_class,
                        'max_nodes': 15}
    else:
        OSError('Invalid argument!!')
    return exp_method, forward_kwargs


parser = argparse.ArgumentParser()
parser.add_argument('--exp_method', required=True, help='name of the explanation method')
parser.add_argument('--expt_name', help='folder for saving results')
parser.add_argument('--evaluate', type=bool, help='flag for evaluating metric')
args = parser.parse_args()


# Folder to collect epoch snapshots
save_dir = os.path.join(os.getcwd(), f'results_{args.expt_name}')

# Folder to store results
if not os.path.exists(save_dir):
    os.makedirs(name=save_dir)

seed_value=912
rand.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ShapeGraph dataset
if args.expt_name == 'small':
    bah = torch.load(open('/home/cha567/GraphXAI/data/ShapeGraph/SG_triangles.pickle', 'rb'))
elif args.expt_name == 'large':
    bah = torch.load(open('/home/cha567/GraphXAI/data/ShapeGraph/SG_homophily.pickle', 'rb'))
else:
    OSError('Invalid argument!!')

data = bah.get_graph(use_fixed_split=True)

inhouse = (data.test_mask == True).nonzero(as_tuple=True)[0]
# inhouse = (data.y[data.test_mask] == 1).nonzero(as_tuple=True)[0]
# np.random.shuffle(inhouse.numpy())
# print(inhouse)

# Test on 3-layer basic GCN, 16 hidden dim:
model = GIN_3layer_basic(16, input_feat = 11, classes = 2).to(device)

# Load model
if args.expt_name == 'small':
    model.load_state_dict(torch.load('./model_weights/model_triangle.pth'))
elif args.expt_name == 'large':
    model.load_state_dict(torch.load('./model_weights/model_homophily.pth'))
else:
    OSError('Invalid argument!!')

gea_feat = []
gea_node = []
gea_edge = []

# Get predictions
pred = model(data.x.to(device), data.edge_index.to(device))

criterion = torch.nn.CrossEntropyLoss().to(device)

if args.exp_method=='pgex' and not args.evaluate:
    explainer = PGExplainer(model, emb_layer_name = 'gin3' if isinstance(model, GIN_3layer_basic) else 'gcn3', max_epochs=1, lr=0.1)
    explainer.train_explanation_model(data.to(device))

for node_idx in tqdm.tqdm(inhouse):

    node_idx = node_idx.item()

    # Get predictions
    pred_class = pred[node_idx, :].reshape(-1, 1).argmax(dim=0)
    
    if pred_class == data.y[node_idx]:

        if not args.evaluate:

            # Get explanation method
            if args.exp_method != 'pgex':
                explainer, forward_kwargs = get_exp_method(args.exp_method, model, criterion, bah, node_idx, pred_class)
            else:
                forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'label': pred_class}

            # Get explanations
            gt_exp = bah.explanations[node_idx]
            exp = explainer.get_explanation_node(**forward_kwargs)

            # Save exp
            np.save(f'{save_dir}/{args.exp_method}_{node_idx}.pickle', exp)
            np.save(f'{save_dir}/gt_{node_idx}.pickle', gt_exp)

        else:
            exp = np.load(f'{save_dir}/{args.exp_method}_{node_idx}.pickle.npy', allow_pickle=True).ravel()[0]
            # gt_exp = np.load(f'{save_dir}/gt_{node_idx}.pickle.npy', allow_pickle=True)

            # Calculate metrics
            # feat, node, edge = graph_exp_acc(gt_exp, exp)
            feat, node, edge = graph_exp_faith(exp, bah, model, sens_idx=[bah.sensitive_feature])

            gea_feat.append(feat)
            gea_node.append(node)
            gea_edge.append(edge)

############################
if args.evaluate:
    # Saving the metric values
    np.save(f'{save_dir}/{args.exp_method}_gef_feat.npy', gea_feat)
    np.save(f'{save_dir}/{args.exp_method}_gef_node.npy', gea_node)
    np.save(f'{save_dir}/{args.exp_method}_gef_edge.npy', gea_edge)
