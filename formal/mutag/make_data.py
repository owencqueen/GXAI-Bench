import os
import torch
import numpy as np
from gxai_eval.datasets.real_world.MUTAG import MUTAG

my_base = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GXAI-Bench'

dataset = MUTAG(
    split_sizes = (0.6, 0.3, 0.1), 
    seed = 1239,
    root = os.path.join(my_base, 'formal/mutag/data')
    )  

print('Class imbalance:')
print('Label==0:', np.sum([dataset.graphs[i].y.item() == 0 for i in range(len(dataset))]))
print('Label==1:', np.sum([dataset.graphs[i].y.item() == 1 for i in range(len(dataset))]))

torch.save(dataset, open('data/mutag.pth', 'wb'))