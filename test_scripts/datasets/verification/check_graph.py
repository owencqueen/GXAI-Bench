import sys, os
import torch
import networkx as nx
from gxai_eval.datasets.load_synthetic import Owen_root
from gxai_eval.datasets.utils.shapes import house
from gxai_eval.datasets.utils.verify import verify_motifs

assert len(sys.argv) == 2, 'usage: python3 check_graph.py <filename>'

file = sys.argv[1]

bah = torch.load(os.path.join(Owen_root, file))

if verify_motifs(bah.G, house):
    print('Verified')
else:
    print('Not Verified')