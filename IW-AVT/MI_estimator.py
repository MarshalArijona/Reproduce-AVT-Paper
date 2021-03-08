import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NWJ(nn.Module):
	def __init__(self, x_dim, y_dim, hidden_size):
		super(NWJ, self).__init__()
		self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
					  nn.ReLU(),
					  nn.Linear(hidden_size, hidden_size),
					  nn.ReLU(),
					  nn.Linear(hidden_size, 1)
					)
	
	def mi_est(self, x_samples, y_samples):
		#shuffle and concatenate
		sample_size = y_samples.shape[0]
		random_index = torch.randint(sample_size, (sample_size,)).long()

		x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
		y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

		T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1))
		T1 = self.T_func(torch.cat([x_tile, y_tile], dim=-1))-1.

		lower_bound = T0.mean() - (T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean()

		return lower_bound


