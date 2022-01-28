import torch
from typing import List

from .explanation import Explanation

def aggregate_explanations(
        exp_list: List[Explanation], 
        reference_exp = None,
        feature_aggregator = torch.max,
        node_aggregator = torch.max,
        edge_aggregator = torch.max,
        node_level = True):
    '''
    Assumes all explanations are generated by a method that has
    '''

    feature_imp = None
    node_imp = None
    edge_imp = None

    ref_exp = reference_exp if reference_exp is not None else exp_list[0]

    # Check for None's in the first explanation:
    if ref_exp.feature_imp is not None:
        feature_imp = feature_aggregator(torch.stack([exp.feature_imp for exp in exp_list]), dim = 0)[0]

    if ref_exp.node_imp is not None:
        node_imp = node_aggregator(torch.stack([exp.node_imp for exp in exp_list]), dim = 0)[0]

    if ref_exp.edge_imp is not None:
        edge_imp = edge_aggregator(torch.stack([exp.edge_imp for exp in exp_list]), dim = 0)[0]

    exp = Explanation(
        feature_imp = feature_imp,
        node_imp = node_imp,
        edge_imp = edge_imp,
        node_idx = ref_exp.node_idx
    )

    if node_level:
        exp.set_enclosing_subgraph(ref_exp.enc_subgraph)
    else:
        exp.set_whole_graph(ref_exp.graph)

    return exp