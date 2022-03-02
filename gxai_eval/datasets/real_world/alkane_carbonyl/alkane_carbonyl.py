import os
import torch

from gxai_eval.datasets.real_world.extract_google_datasets import load_graphs 
from gxai_eval.datasets import GraphDataset


ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*'
]

fc_data_dir = os.path.join(os.path.dirname(__file__), 'ac_data')
fc_smiles_df = 'AC_smiles.csv'

class AlkaneCarbonyl(GraphDataset):

    def __init__(
            self,
            split_sizes = (0.7, 0.2, 0.1),
            seed = None,
            data_path: str = fc_data_dir,
            device = None,
        ):
        '''
        Args:
            split_sizes (tuple): 
            seed (int, optional):
            data_path (str, optional):
        '''
        
        self.device = device

        self.graphs, self.explanations, self.zinc_ids = \
            load_graphs(data_path, os.path.join(data_path, fc_smiles_df))

        super().__init__(name = 'AklaneCarbonyl', seed = seed, split_sizes = split_sizes, device = device)