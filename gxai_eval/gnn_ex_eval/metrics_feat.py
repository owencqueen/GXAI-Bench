import math
import time
import os
import ipdb

import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.utils import convert
from networkx.algorithms.swap import double_edge_swap as swap
from networkx.linalg.graphmatrix import adjacency_matrix as adj_mat

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve


class FeatMetrics():
	def __init__(self, model, labels, edge_index, mapping, device):
		self.model = model
		self.labels = labels
		self.device = device
		self.edge_index = edge_index
		self.mapping = mapping

	def fair_metric(self, pred, org, gt, sens):
		idx_s0 = sens == 0
		idx_s1 = sens == 1
		parity_1 = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
		parity_2 = abs(sum(org[idx_s0]) / sum(idx_s0) - sum(org[idx_s1]) / sum(idx_s1))
		bound = abs(np.mean((gt[idx_s0] - pred[idx_s0] + org[idx_s0]))) + abs(np.mean((gt[idx_s1] - pred[idx_s1] + org[idx_s1])))
		return abs(parity_1-parity_2), bound/2

	def rewire_edges(self, x, edge_index, degree):
		# Convert to networkx graph for rewiring edges
		data = Data(x=x, edge_index=edge_index)
		G = convert.to_networkx(data, to_undirected=True)
		rewired_G = swap(G, nswap=degree, max_tries=degree * 25, seed=912)
		rewired_adj_mat = adj_mat(rewired_G)
		rewired_edge_indexes = convert.from_scipy_sparse_matrix(rewired_adj_mat)[0]
		return rewired_edge_indexes

	def local_group_faithfulness(self, per_x, node_exp, num_samples, top_k_features, ind, degree):

		# Calculate the bound
		explanation_mask = torch.where(node_exp.data >= node_exp.topk(int(top_k_features))[0][-1].item(),
									   torch.ones_like(node_exp.data), torch.zeros_like(node_exp.data))
		x_norm = torch.norm((torch.ones_like(explanation_mask) - explanation_mask).mul(per_x[0][ind, :])).item()
		weight_norm = 1
		for name, param in self.model.named_parameters():
			if 'lin_l.weight' in name or 'fc.weight' in name:
				weight_norm = weight_norm * torch.norm(param).item()
		gamma = weight_norm * x_norm

		# Calculate the number of faithful nodes
		correct: int = 0
		output_diff_norm = 0
		norm_output_diff_norm = 0
		self.model.to(self.device)
		for i in range(num_samples):
			# predictions for perturbed node features
			self.model.zero_grad()

			if i == 0:
				output_per_x = self.model(per_x[i].to(self.device), self.edge_index.to(self.device))
				preds_per_x = torch.argmax(output_per_x, dim=-1)

				# predictions for masked node features using explanations
				xx = per_x[i].clone()
				xx[ind, :] = xx[ind, :].mul(explanation_mask)
				output_exp_per_x = self.model(xx.to(self.device), self.edge_index.to(self.device))
				preds_exp_per_x = torch.argmax(output_exp_per_x, dim=-1)
			else:
				try:
					rewired_edge_index = self.rewire_edges(x=per_x[i].to(self.device),
														   edge_index=self.edge_index.to(self.device),
														   degree=degree)
				except:
					continue

				output_per_x = self.model(per_x[i].to(self.device), rewired_edge_index.to(self.device))
				preds_per_x = torch.argmax(output_per_x, dim=-1)

				# predictions for masked node features using explanations
				xx = per_x[i].clone()
				xx[ind, :] = xx[ind, :].mul(explanation_mask)
				output_exp_per_x = self.model(xx.to(self.device), rewired_edge_index.to(self.device))
				preds_exp_per_x = torch.argmax(output_exp_per_x, dim=-1)

			# Calculate local_group_faithfulness
			correct += 1 * preds_per_x[self.mapping].eq(preds_exp_per_x[self.mapping]).item()
			output_diff_norm += torch.norm(output_exp_per_x[self.mapping] - output_per_x[self.mapping]).item()
			norm_output_diff_norm += torch.norm(
				output_exp_per_x[self.mapping] - output_per_x[self.mapping]).item() / torch.norm(
				torch.FloatTensor([[0.0, 1.0]]) - torch.FloatTensor([[1.0, 0.0]])).item()
		return correct / num_samples, gamma, output_diff_norm, norm_output_diff_norm/num_samples

	def dist_explanation(self, exp, pert_exp, top_k_features, ind):
		exp_mask = torch.where(exp.data >= exp.topk(int(top_k_features))[0][-1].item(), torch.ones_like(exp.data),
							   torch.zeros_like(exp.data))
		pert_exp_mask = torch.where(pert_exp.data >= pert_exp.topk(int(top_k_features))[0][-1].item(),
									torch.ones_like(pert_exp.data), torch.zeros_like(pert_exp.data))
		return F.pairwise_distance(exp_mask.unsqueeze(dim=0), pert_exp_mask.unsqueeze(dim=0), 1) / exp_mask.shape[0]

	def local_group_fairness(self, per_x, subset, node_exp, num_samples, top_k_features, ind, sens_idx, degree):
		explanation_mask = torch.where(node_exp.data >= node_exp.topk(int(top_k_features))[0][-1].item(),
									   torch.ones_like(node_exp.data), torch.zeros_like(node_exp.data))

		# Generate the predictions
		org_pred = []
		exp_pred = []
		gt = []
		sens = []
		for i in range(num_samples):
			self.model.zero_grad()
			if i == 0:
				sens.append(per_x[i][ind, sens_idx].item())

				# predictions for perturbed node features
				output_per_x = self.model(per_x[i].to(self.device), self.edge_index.to(self.device))
				preds_per_x = torch.argmax(output_per_x, dim=-1)

				# predictions for masked node features using explanations
				xx = per_x[i].clone()
				xx[ind, :] = xx[ind, :].mul(explanation_mask)
				output_exp_per_x = self.model(xx.to(self.device), self.edge_index.to(self.device))
				preds_exp_per_x = torch.argmax(output_exp_per_x, dim=-1)
			else:
				try:
					rewired_edge_index = self.rewire_edges(x=per_x[i].to(self.device), edge_index=self.edge_index,
														   degree=degree)
				except:
					continue

				sens.append(per_x[i][ind, sens_idx].item())

				# predictions for perturbed node features
				output_per_x = self.model(per_x[i].to(self.device), rewired_edge_index.to(self.device))
				preds_per_x = torch.argmax(output_per_x, dim=-1)

				# predictions for masked node features using explanations
				xx = per_x[i].clone()
				xx[ind, :] = xx[ind, :].mul(explanation_mask)
				output_exp_per_x = self.model(xx.to(self.device), rewired_edge_index.to(self.device))
				preds_exp_per_x = torch.argmax(output_exp_per_x, dim=-1)
			gt.append(self.labels[subset][self.mapping].item())
			org_pred.append(preds_per_x.type_as(self.labels)[self.mapping].item())
			exp_pred.append(preds_exp_per_x.type_as(self.labels)[self.mapping].item())
		if np.unique(np.array(sens)).shape[0] == 1:
			return 0, 0
		else:
			return self.fair_metric(np.array(exp_pred), np.array(org_pred), np.array(gt), np.array(sens))
