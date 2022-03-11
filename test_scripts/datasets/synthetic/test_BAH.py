import torch
from gxai_eval.datasets import BAHouses

dataset = BAHouses(
    n = 300 + 30 * 5,
    m = 1,
    num_hops = 2,
    num_houses = 30,
    seed = None
)

data = dataset.get_graph()

dataset.explanations[-20].context_draw(
    num_hops = 2,
    graph_data = data,
    show = True
)

