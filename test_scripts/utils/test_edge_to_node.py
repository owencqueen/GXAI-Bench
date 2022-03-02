import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import from_networkx

from gxai_eval.datasets.utils.bound_graph_pref_att import ba_around_shape
from gxai_eval.utils import node_mask_from_edge_mask


house = nx.house_graph()

for n in house.nodes:
    house.nodes[n]['shape'] = 1

G = ba_around_shape(house, add_size = 7, show_subgraphs=False)

node_c = [G.nodes[n]['shape'] for n in G.nodes]

nx.draw(G, node_color = node_c)
plt.show()