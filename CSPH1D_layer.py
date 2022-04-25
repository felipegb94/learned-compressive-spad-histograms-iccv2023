#### Standard Library Imports
import os

#### Library imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
import torchvision
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports


def zero_norm(v, dim=-1):
	zero_mean_v = (v - torch.mean(v, dim=dim, keepdim=True)) 
	zn_v = zero_mean_v / (torch.linalg.norm(zero_mean_v, ord=2, dim=dim, keepdim=True) + 1e-6)
	return zn_v

class CSPH1DLayer(nn.Module):
	def __init__(self, k=2, num_bins=1024):
		# Init parent class
		super(CSPH1DLayer, self).__init__()
		
		self.Cmat = torch.nn.Parameter(torch.randn(num_bins, k).type(torch.float32))


	def forward(self, inputs):
		## Move time dim to last dimension
		inputs_reshaped = torch.transpose(inputs, -3, -1)

		## Compute compressive histogram
		B = torch.matmul(inputs_reshaped, self.Cmat)

		## Compute ZNCC Scores Table
		zn_Cmat = zero_norm(self.Cmat, dim=-1)
		zn_B = zero_norm(B, dim=-1)
		zncc = torch.matmul(zn_B, zn_Cmat.t())
		
		## Transpose time dimension again
		zncc = torch.transpose(zncc, -3, -1)

		return zncc, B

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__=='__main__':
	import matplotlib.pyplot as plt

	# Set random input
	batch_size = 1
	(nr, nc, nt) = (32, 32, 1024) 
	inputs = torch.randn((batch_size, 1, nt, nr, nc))

	# Set compression params
	k = 16
	csph1D_layer = CSPH1DLayer(k=k, num_bins=nt)
	print("csph1D_layer Layer N Params: {}".format(count_parameters(csph1D_layer)))

	zncc, B = csph1D_layer(inputs)


	zn_Cmat = zero_norm(csph1D_layer.Cmat, dim=-1)
	zn_B = zero_norm(B, dim=-1)

	lookup = torch.matmul(zn_B, zn_Cmat.t())
