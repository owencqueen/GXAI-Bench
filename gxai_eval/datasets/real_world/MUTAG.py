import torch
import itertools
import numpy as np

from sklearn.model_selection import train_test_split

from torch_geometric.datasets import TUDataset

from gxai_eval.datasets.dataset import GraphDataset
from gxai_eval.utils import Explanation, match_edge_presence
from gxai_eval.datasets.utils.substruct_chem_match import match_NH2, match_substruct, MUTAG_NO2, make_NO2
from gxai_eval.utils import aggregate_explanations

# 0	C
# 1	O
# 2	Cl
# 3	H
# 4	N
# 5	F
# 6	Br
# 7	S
# 8	P
# 9	I
# 10	Na
# 11	K
# 12	Li
# 13	Ca

def make_iter_combinations(length):
    '''
    Builds increasing level of combinations, including all comb's at r = 1, ..., length - 1
    Used for building combinations of explanations
    '''

    if length == 1:
        return [[0]]

    inds = np.arange(length)

    exps = [[i] for i in inds]
    
    for l in range(1, length - 1):
        exps += list(itertools.combinations(inds, l + 1))

    exps.append(list(inds)) # All explanations

    return exps


class MUTAG(GraphDataset):
    '''
    GraphXAI implementation MUTAG dataset
        - Contains MUTAG with ground-truth 

    Args:
        root (str): Root directory in which to store the dataset
            locally.
        generate (bool, optional): (:default: :obj:`False`) 
    '''

    def __init__(self,
        root: str,
        use_fixed_split: bool = True, 
        generate: bool = True,
        split_sizes = (0.7, 0.2, 0.1),
        seed = None
        ):

        self.graphs = TUDataset(root=root, name='MUTAG')
        # self.graphs retains all qualitative and quantitative attributes from PyG

        self.__make_explanations()

        super().__init__(name = 'MUTAG', seed = seed, split_sizes = split_sizes)


    def __make_explanations(self):
        '''
        Makes explanations for MUTAG dataset
        '''

        self.explanations = []

        # Need to do substructure matching
        for i in range(len(self.graphs)):

            molG = self.get_graph_as_networkx(i)

            node_imp = torch.zeros(molG.number_of_nodes())

            nh2_matches = []

            # Screen for NH2:
            for n in molG.nodes():
                # Screen all nodes through match_NH2
                # match_NH2 is very quick
                m = match_NH2(molG, n, google = False)
                if m:
                    nh2_matches.append(m)

            # Screen for NO2:
            no2_matches = match_substruct(molG, make_NO2(google = False), google=False)

            eidx = self.graphs[i].edge_index
            explanations_i = []

            all_matches = no2_matches + nh2_matches

            for m in all_matches:
                node_imp = torch.zeros((molG.number_of_nodes(),))

                node_imp[m] = 1
                edge_imp = match_edge_presence(eidx, m)

                exp = Explanation(
                    node_imp = node_imp.float(),
                    edge_imp = edge_imp.float()
                )

                exp.set_whole_graph(self.graphs[i])

                exp.has_match = True

                explanations_i.append(exp)

            if len(explanations_i) == 0:
                # Set a null explanation:
                exp = Explanation(
                    node_imp = torch.zeros((molG.number_of_nodes(),), dtype = torch.float),
                    edge_imp = torch.zeros((eidx.shape[1],), dtype = torch.float)
                )

                exp.set_whole_graph(self.graphs[i])

                exp.has_match = False

                explanations_i = [exp]

                self.explanations.append(explanations_i)

            else:
                # Combinatorial combination of matches:
                exp_matches_inds = make_iter_combinations(len(all_matches))

                comb_explanations = []

                # Run combinatorial build of all explanations
                for eid in exp_matches_inds:
                    # Get list of explanations:
                    L = [explanations_i[j] for j in eid]
                    tmp_exp = aggregate_explanations(L, node_level = False)
                    tmp_exp.has_match = True
                    comb_explanations.append(tmp_exp) # No reference provided
                    

                self.explanations.append(comb_explanations)
            
            # for m in no2_matches:
            #     node_imp[m] = 1 # Mask-in those values

            #     # Update edge mask:
                
            #     cumulative_edge_mask = cumulative_edge_mask.bool() | (match_edge_presence(eidx, m))

            # for m in nh2_matches:
            #     node_imp[m] = 1

            #     # Update edge_mask:
            
            #     cumulative_edge_mask = cumulative_edge_mask.bool() | (match_edge_presence(eidx, m))

            # TODO: mask-in edge importance

            # exp = Explanation(
            #     node_imp = node_imp,
            #     edge_imp = cumulative_edge_mask.float(),
            # )

            # exp.set_whole_graph(self.graphs[i])

            # self.explanations.append(exp)
