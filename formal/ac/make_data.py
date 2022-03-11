import torch
import numpy as np
from gxai_eval.datasets import AlkaneCarbonyl

dataset = AlkaneCarbonyl(split_sizes = (0.7, 0.2, 0.1), seed = 1234)

torch.save(dataset, open('data/ac.pth', 'wb'))