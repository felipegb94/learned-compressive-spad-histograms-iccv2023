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


def pad_h_irf(h_irf, num_bins):
	if(not (h_irf is None)):
		if(len(h_irf) < num_bins):
			h_irf_new = np.zeros((num_bins,)).astype(h_irf.dtype)
			h_irf_new[0:len(h_irf)] = h_irf
		else:
			h_irf_new = h_irf
			assert(len(h_irf_new) == num_bins), "Length of input h_irf needs to be <= num_bins"
		# Center irf if needed
		return np.roll(h_irf_new, shift=-1*np.argmax(h_irf_new, axis=-1))
	else: return h_irf

class CSPH1DGlobalEncodingLayer(nn.Module):
	def __init__(self, k=2, num_bins=1024, init='TruncFourier', h_irf=None, optimize_weights=False):
		# Init parent class
		super(CSPH1DGlobalEncodingLayer, self).__init__()

		self.num_bins = num_bins
		self.k = k
		# Pad IRF with zeros if needed
		self.h_irf = pad_h_irf(h_irf, num_bins=num_bins) # This is used to select the frequencies that will be used in HybridGrayFourier

		if(init == 'TruncFourier'):
			coding_obj = TruncatedFourierCoding(num_bins, n_codes=k, include_zeroth_harmonic=False, h_irf=self.h_irf, account_irf=True)
			Cmat_init = coding_obj.C
		elif(init == 'HybridGrayFourier'):
			coding_obj = HybridGrayBasedFourierCoding(num_bins, n_codes=k, include_zeroth_harmonic=False, h_irf=self.h_irf, account_irf=True)
			Cmat_init = coding_obj.C
		elif(init == 'CoarseHist'):
			coding_obj = GatedCoding(num_bins, n_gates=k, h_irf=self.h_irf, account_irf=True)
			Cmat_init = coding_obj.C
		elif(init == 'Rand'):
			Cmat_init = np.random.randn(num_bins, k)*0.01
			Cmat_init[Cmat_init >= 1] = 1
			Cmat_init[Cmat_init <= -1] = -1
		else:
			assert(False), "Invalid CSPH1D init ID"
		
		# Define a 1x1 convolution that does a dot product on the time axis (i.e., dim=-3)
		self.Cmat1D = torch.nn.Conv2d( in_channels=self.num_bins 
										, out_channels=self.k 
										, kernel_size=1 
										, stride = 1, padding=0, dilation=1, bias=False)
		# Init weights
		self.Cmat_init = Cmat_init.transpose()
		self.Cmat1D.weight.data = torch.from_numpy(self.Cmat_init[..., np.newaxis, np.newaxis].astype(np.float32))
		self.Cmat1D.weight.requires_grad = optimize_weights



	def forward(self, inputs):
		'''
			Expected input dims == (Batch, Nt, Nr, Nc)
		'''
		## Compute compressive histogram
		B = self.Cmat1D(inputs)
		return B


if __name__=='__main__':
	import matplotlib.pyplot as plt

	# Set random input
	k=8
	batch_size = 3
	(nr, nc, nt) = (32, 32, 1024) 
	inputs = torch.randn((batch_size, nt, nr, nc))
	inputs[inputs<2] = 0

	simple_hist_input = torch.zeros((2, nt, 32, 32))
	simple_hist_input[0, 100, 0, 0] = 3
	simple_hist_input[0, 200, 0, 0] = 1
	simple_hist_input[0, 50, 0, 0] = 1
	simple_hist_input[0, 540, 0, 0] = 1

	simple_hist_input[1, 300, 0, 0] = 2
	simple_hist_input[1, 800, 0, 0] = 1
	simple_hist_input[1, 34, 0, 0] = 1
	simple_hist_input[1, 900, 0, 0] = 1


	## 1D CSPH Coding Object
	csph1D_obj = CSPH1DGlobalEncodingLayer(k=k, num_bins=nt, init='TruncFourier', optimize_weights=False)
	# Plot coding matrix
	plt.clf()
	plt.plot(csph1D_obj.Cmat1D.weight.data.cpu().numpy().squeeze().transpose())

	# Test Fourier encoding
	outputs = csph1D_obj(inputs)
	simple_hist_output = csph1D_obj(simple_hist_input)
	print("CSPH1D Global Encoding:")
	print("    inputs1: {}".format(inputs.shape))
	print("    outputs1: {}".format(outputs.shape))
	print("    inputs2: {}".format(simple_hist_input.shape))
	print("    outputs2: {}".format(simple_hist_output.shape))
	# Validate outputs
	fft_inputs = torch.fft.rfft(inputs, dim=-3)
	fft_inputs_real = fft_inputs.real[:, 1:1+(k//2), :]
	fft_inputs_imag = fft_inputs.imag[:, 1:1+(k//2), :]
	print("    outputs1.real - inputs.fft.real: {}".format(torch.mean(torch.abs(fft_inputs_real - outputs[:,0::2,:]))))
	print("    outputs1.imag - inputs.fft.imag: {}".format(torch.mean(torch.abs(fft_inputs_imag - outputs[:,1::2,:]))))

