#### Standard Library Imports
import os

#### Library imports
import numpy as np
import scipy.linalg
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
from toflib.coding import TruncatedFourierCoding, HybridGrayBasedFourierCoding, GatedCoding, HybridFourierBasedGrayCoding


def zero_norm(v, dim=-1):
	zero_mean_v = (v - torch.mean(v, dim=dim, keepdim=True)) 
	zn_v = zero_mean_v / (torch.linalg.norm(zero_mean_v, ord=2, dim=dim, keepdim=True) + 1e-6)
	return zn_v

class CSPH1DLayer(nn.Module):
	def __init__(self, k=2, num_bins=1024, init='TruncFourier'):
		# Init parent class
		super(CSPH1DLayer, self).__init__()

		Cmat_init = np.zeros((num_bins, k))
		if(init == 'TruncFourier'):
			coding_obj = TruncatedFourierCoding(num_bins, n_codes=k, include_zeroth_harmonic=False)
			Cmat_init = coding_obj.C
		elif(init == 'HybridGrayFourier'):
			coding_obj = HybridGrayBasedFourierCoding(num_bins, n_codes=k, include_zeroth_harmonic=False)
			Cmat_init = coding_obj.C
		elif(init == 'HybridFourierGray'):
			coding_obj = HybridFourierBasedGrayCoding(num_bins, n_codes=k)
			Cmat_init = coding_obj.C
		elif(init == 'CoarseHist'):
			coding_obj = GatedCoding(num_bins, n_gates=k)
			Cmat_init = coding_obj.C
		elif(init == 'RandBinary'):
			Cmat_init = torch.randn(num_bins, k)
			Cmat_init[Cmat_init >= 0] = 1
			Cmat_init[Cmat_init < 0] = -1
		elif(init == 'Rand'):
			Cmat_init = torch.randn(num_bins, k)
			Cmat_init[Cmat_init >= 1] = 1
			Cmat_init[Cmat_init <= -1] = -1
		else:
			assert(False), "Invalid CSPH1D init ID"
		
		self.Cmat = torch.nn.Parameter(torch.tensor(Cmat_init).type(torch.float32), requires_grad=False)
		self.Cmat_t = self.Cmat.transpose(0,1)

	def forward_matmul(self, inputs):
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
		B = torch.transpose(B, -1, -4).squeeze(-1) # Make B as a 2D image

		return zncc, B

	# def forward_conv(self, inputs):
	# 	'''
	# 		Inputs should have dims (B, 1, D0, D1, D2)
	# 	'''
	# 	Cmat_t_filter = self.Cmat_t.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
	# 	breakpoint()
	# 	B = F.conv3d(inputs, Cmat_t_filter, stride=1, padding=0)

	# 	## Compute ZNCC Scores Table
	# 	zn_Cmat_t_filter = zero_norm(Cmat_t_filter, dim=0)
	# 	zn_B = zero_norm(B, dim=-4)
	# 	# zncc = torch.matmul(zn_B, zn_Cmat.t())

	def forward(self, inputs):
		return self.forward_matmul(inputs)

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__=='__main__':
	import matplotlib.pyplot as plt

	# Set random input
	batch_size = 3
	(nr, nc, nt) = (32, 32, 1024) 
	inputs = torch.randn((batch_size, 1, nt, nr, nc))
	inputs[inputs<2] = 0

	simple_hist_input = torch.zeros((1, 1, nt, 1, 1))
	simple_hist_input[0, 0, 100, 0, 0] = 10

	## Init CSPH Layer Set compression params
	k = 32
	init_id = 'TruncFourier'
	# init_id = 'HybridGrayFourier'
	# init_id = 'HybridFourierGray'
	# init_id = 'CoarseHist'
	# init_id = 'Rand'
	csph1D_layer = CSPH1DLayer(k=k, num_bins=nt, init=init_id)
	print("csph1D_layer Layer N Params: {}".format(count_parameters(csph1D_layer)))

	## Plot codes
	Cmat = csph1D_layer.Cmat.cpu().numpy()
	assert(Cmat.shape[-1] == k), "Incorrect Cmat shape"
	plt.clf()
	plt.subplot(2,1,1)
	c=0; plt.plot(Cmat[:,c], label='Code/Col: {}'.format(c))
	c=k//4; plt.plot(Cmat[:,c], label='Code/Col: {}'.format(c))
	c=k//2; plt.plot(Cmat[:,c], label='Code/Col: {}'.format(c))
	plt.legend()

	## Test with random inputs
	zncc, B = csph1D_layer(inputs)
	zncc2, B2 = csph1D_layer.forward_conv(inputs)
	print("Input Shape: {}".format(inputs.shape))
	print("ZNCC Out Shape: {}".format(zncc.shape))
	print("B Out Shape: {}".format(B.shape))
	
	# Look at outputs
	plt.subplot(2,1,2)
	plt.plot(inputs[0,0,:,10,10], '--', label="Inputs 1")
	plt.plot(zncc[0,0,:,10,10], label='ZNCC Outputs 1')

	## Test with random inputs
	zncc, B = csph1D_layer(simple_hist_input)
	print("Input Shape: {}".format(simple_hist_input.shape))
	print("ZNCC Out Shape: {}".format(zncc.shape))
	print("B Out Shape: {}".format(B.shape))
	
	# Look at outputs
	plt.plot(simple_hist_input[0,0,:,0,0], '--', label="Inputs 2")
	plt.plot(zncc[0,0,:,0,0], label='ZNCC Outputs 2')





