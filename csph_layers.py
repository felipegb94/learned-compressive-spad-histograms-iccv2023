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

def zero_norm_vec(v, dim=-1):
	zero_mean_v = (v - torch.mean(v, dim=dim, keepdim=True)) 
	zn_v = zero_mean_v / (torch.linalg.norm(zero_mean_v, ord=2, dim=dim, keepdim=True) + 1e-6)
	return zn_v

def norm_vec(v, dim=-1):
	n_v = v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + 1e-6)
	return n_v

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

class CSPH1DGlobalDecodingZNCC(nn.Module):
	def __init__(self, csph1D_encoding_layer: CSPH1DGlobalEncodingLayer):
		# Init parent class
		super(CSPH1DGlobalDecodingZNCC, self).__init__()

		self.csph1D_encoding_layer = csph1D_encoding_layer
		self.h_irf = csph1D_encoding_layer.h_irf

		self.norm_op = zero_norm_vec

	def forward(self, input_csph):
		'''
			Input CSPH should have dims (Batch, K, nr, nc)
		'''
		curr_Cmat = self.csph1D_encoding_layer.Cmat1D.weight

		## Compute ZNCC Scores Table
		zn_csph = self.norm_op(input_csph, dim=-3)
		zn_Cmat = self.norm_op(curr_Cmat, dim=0)

		zncc = torch.matmul(torch.transpose(zn_csph, -3, -1), zn_Cmat.squeeze())

		return torch.transpose(zncc, -3, -1)

class CSPH2DLocalEncodingLayer(nn.Module):
	def __init__(self, k = 2, down_factor = 1):
		# Init parent class
		super(CSPH2DLocalEncodingLayer, self).__init__()

		kernel = down_factor
		stride = down_factor

		# By setting groups == k we make it a separable convolution
		self.Cmat2D = torch.nn.Conv2d(
					in_channels = k, 
					out_channels = k, 
					kernel_size = kernel, 
					stride=stride, 
					groups=k, 
					bias=False)

		

	def forward(self, inputs):
		'''
			Expected input dims == (Batch, Nt, Nr, Nc)
		'''
		## Compute compressive histogram
		B = self.Cmat2D(inputs)
		return B

class CSPH2DLocalDecodingLayer(nn.Module):
	def __init__(self, factor = 1):
		# Init parent class
		super(CSPH2DLocalDecodingLayer, self).__init__()
		self.upsample_layer = nn.UpsamplingBilinear2d(scale_factor=factor)
		
	def forward(self, inputs):
		return self.upsample_layer(inputs)

class CSPH1D2DLayer(nn.Module):
	def __init__(self, k=2, num_bins=1024, init='TruncFourier', h_irf=None, optimize_weights=False, down_factor=1):
		# Init parent class
		super(CSPH1D2DLayer, self).__init__()

		self.csph1D_encoding = CSPH1DGlobalEncodingLayer(k=k, num_bins=num_bins, init=init, h_irf=h_irf, optimize_weights=optimize_weights)
		self.csph2D_encoding = CSPH2DLocalEncodingLayer(k=k, down_factor=down_factor)

		self.csph1D_decoding = CSPH1DGlobalDecodingZNCC(self.csph1D_encoding)
		self.csph2D_decoding = CSPH2DLocalDecodingLayer(factor=down_factor)

	def forward(self, inputs):
		'''
			Expected input dims == (Batch, Nt, Nr, Nc)
		'''
		## Compute compressive histogram
		B1 = self.csph1D_encoding(inputs)
		B2 = self.csph2D_encoding(B1)
		B3 = self.csph2D_decoding(B2)
		B4 = self.csph1D_decoding(B3)
		return B4, B2

if __name__=='__main__':
	import matplotlib.pyplot as plt

	# Set random input
	k=64
	batch_size = 3
	(nr, nc, nt) = (32, 32, 1024) 
	inputs = torch.randn((batch_size, nt, nr, nc))
	inputs[inputs<2] = 0

	simple_hist_input = torch.zeros((2, nt, 32, 32))
	simple_hist_input[0, 100, 0, 0] = 3
	# simple_hist_input[0, 200, 0, 0] = 1
	# simple_hist_input[0, 50, 0, 0] = 1
	# simple_hist_input[0, 540, 0, 0] = 1

	simple_hist_input[1, 300, 0, 0] = 2
	# simple_hist_input[1, 800, 0, 0] = 1
	# simple_hist_input[1, 34, 0, 0] = 1
	# simple_hist_input[1, 900, 0, 0] = 1

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


	## 1D2D CSPH Coding Object
	csph1D2D_obj = CSPH1D2DLayer(k=k, num_bins=nt, init='TruncFourier', optimize_weights=False, down_factor=1)
	(outputs, csph_out) = csph1D2D_obj(inputs)
	(simple_hist_output, simple_hist_csph) = csph1D2D_obj(simple_hist_input)
	print("CSPH1D2D Layer:")
	print("    inputs1: {}".format(inputs.shape))
	print("    outputs1: {}".format(outputs.shape))
	print("    inputs2: {}".format(simple_hist_input.shape))
	print("    outputs2: {}".format(simple_hist_output.shape))

	plt.clf()
	plt.subplot(2,1,1)
	plt.plot(simple_hist_input[0,:,0,0])
	plt.plot(simple_hist_output[0,:,0,0].detach().cpu().numpy())
	plt.plot(simple_hist_output[0,:,1,1].detach().cpu().numpy())

	plt.subplot(2,1,2)
	plt.plot(simple_hist_input[1,:,0,0])
	plt.plot(simple_hist_output[1,:,0,0].detach().cpu().numpy())
	plt.plot(simple_hist_output[1,:,1,1].detach().cpu().numpy())


