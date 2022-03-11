import torch, pickle
from gxai_eval.datasets import BAHouses

dataset = BAHouses(
    n = 300 + 50 * 4,
    m = 1,
    model_layers = 2,
    num_houses = 50,
    seed = 1234
)

torch.save(dataset, open('data/BAH.pth', 'wb'))