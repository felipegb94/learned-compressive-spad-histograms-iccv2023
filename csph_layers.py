#### Standard Library Imports
import os
from pickletools import optimize
from typing import OrderedDict

#### Library imports
import numpy as np
from requests import post
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
from model_unet import UpBlock


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

def get_temporal_cmat_init(k, num_bins, init_id, h_irf=None):
	if(init_id == 'TruncFourier'):
		coding_obj = TruncatedFourierCoding(num_bins, n_codes=k, include_zeroth_harmonic=False, h_irf=h_irf, account_irf=True)
		Cmat_init = coding_obj.C
	elif(init_id == 'HybridGrayFourier'):
		coding_obj = HybridGrayBasedFourierCoding(num_bins, n_codes=k, include_zeroth_harmonic=False, h_irf=h_irf, account_irf=True)
		Cmat_init = coding_obj.C
	elif(init_id == 'HybridFourierGray'):
		coding_obj = HybridFourierBasedGrayCoding(num_bins, n_codes=k, h_irf=h_irf, account_irf=True)
		Cmat_init = coding_obj.C
	elif(init_id == 'CoarseHist'):
		coding_obj = GatedCoding(num_bins, n_gates=k, h_irf=h_irf, account_irf=True)
		Cmat_init = coding_obj.C
	elif(init_id == 'Rand'):
		Cmat_init = np.random.randn(num_bins, k)*0.01
		Cmat_init[Cmat_init >= 1] = 1
		Cmat_init[Cmat_init <= -1] = -1
	else:
		assert(False), "Invalid CSPH1D init ID"
	return Cmat_init

class CSPH1DGlobalEncodingLayer(nn.Module):
	def __init__(self, k=2, num_bins=1024, init='TruncFourier', h_irf=None, optimize_weights=False):
		# Init parent class
		super(CSPH1DGlobalEncodingLayer, self).__init__()

		self.num_bins = num_bins
		self.k = k
		# Pad IRF with zeros if needed
		self.h_irf = pad_h_irf(h_irf, num_bins=num_bins) # This is used to select the frequencies that will be used in HybridGrayFourier

		Cmat_init = get_temporal_cmat_init(k=k, num_bins=num_bins, init_id=init, h_irf=self.h_irf)
		
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

		self.csph2D_decoding = CSPH2DLocalDecodingLayer(factor=down_factor)

		# # 1D ZNCC decoding
		# self.csph1D_decoding = CSPH1DGlobalDecodingZNCC(self.csph1D_encoding)

		## 1D Learned decoding
		self.csph1D_decoding = nn.Sequential(
			nn.Conv2d(k, num_bins, kernel_size=1, stride=1, bias=True),
			nn.ReLU(inplace=True))
		nn.init.kaiming_normal_(self.csph1D_decoding[0].weight, 0, 'fan_in', 'relu'); 
		nn.init.constant_(self.csph1D_decoding[0].bias, 0.0)


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

class CSPH1DGlobal2DLocalLayer4xDown(nn.Module):
	def __init__(self, k=2, num_bins=1024, init='TruncFourier', h_irf=None, optimize_weights=False):
		# Init parent class
		super(CSPH1DGlobal2DLocalLayer4xDown, self).__init__()
		self.down_factor = 4

		self.csph1D_encoding = CSPH1DGlobalEncodingLayer(k=k, num_bins=num_bins, init=init, h_irf=h_irf, optimize_weights=optimize_weights)
		self.csph2D_encoding = CSPH2DLocalEncodingLayer(k=k, down_factor=self.down_factor)

		# self.csph2D_decoding = CSPH2DLocalDecodingLayer(factor=self.down_factor)
		self.csph2D_decoding1 = UpBlock(in_channels = k, out_channels=k, post_conv=False,
							use_dropout=False, norm=None, upsampling_mode='bilinear')
		self.csph2D_decoding2 = UpBlock(in_channels = k, out_channels=k, post_conv=False,
							use_dropout=False, norm=None, upsampling_mode='bilinear')

		# # 1D ZNCC decoding
		# self.csph1D_decoding = CSPH1DGlobalDecodingZNCC(self.csph1D_encoding)

		## 1D Learned decoding
		self.csph1D_decoding = nn.Sequential(
			nn.Conv2d(k, num_bins, kernel_size=1, stride=1, bias=True),
			nn.ReLU(inplace=True))
		nn.init.kaiming_normal_(self.csph1D_decoding[0].weight, 0, 'fan_in', 'relu'); 
		nn.init.constant_(self.csph1D_decoding[0].bias, 0.0)


	def forward(self, inputs):
		'''
			Expected input dims == (Batch, Nt, Nr, Nc)
		'''
		## Compute compressive histogram
		B1 = self.csph1D_encoding(inputs)
		B2 = self.csph2D_encoding(B1)
		B3_1 = self.csph2D_decoding1(B2)
		B3_2 = self.csph2D_decoding2(B3_1)
		B4 = self.csph1D_decoding(B3_2)
		return B4, B2

class CSPHEncodingLayer(nn.Module):
	def __init__(self, k=2, num_bins=1024, 
					tblock_init='TruncFourier', h_irf=None, optimize_tdim_codes=False,
					nt_blocks=1, spatial_block_dims=(1, 1), 
					encoding_type='separable'
				):
		# Init parent class
		super(CSPHEncodingLayer, self).__init__()
		# Validate some input params
		assert((num_bins % nt_blocks) == 0), "Number of bins should be divisible by number of time blocks"
		assert(len(spatial_block_dims) == 2), "Spatial dims should be a tuple of size 2"

		self.num_bins = num_bins
		self.k = k
		self.tblock_init=tblock_init
		self.optimize_tdim_codes=optimize_tdim_codes
		self.nt_blocks=nt_blocks
		self.tblock_len = int(self.num_bins / nt_blocks) 
		self.spatial_block_dims=spatial_block_dims
		# Pad IRF with zeros if needed
		self.h_irf = pad_h_irf(h_irf, num_bins=num_bins) # This is used to select the frequencies that will be used in HybridGrayFourier
		# Get initialization matrix for the temporal dimension block
		self.Cmat_tdim_init = get_temporal_cmat_init(k=k, num_bins=self.tblock_len, init_id=tblock_init, h_irf=self.h_irf)
		
		# Initialize the encoding layer
		self.encoding_type = encoding_type
		self.encoding_kernel_dims = (self.tblock_len,) +  self.spatial_block_dims
		self.encoding_kernel_stride = self.encoding_kernel_dims
		if(self.encoding_type == 'separable'):
			# Separable convolution encoding
			# First, define a tblock_lenx1x1 convolution kernel
			self.Cmat_tdim = torch.nn.Conv3d( in_channels=1
										, out_channels=self.k 
										, kernel_size=(self.tblock_len, 1, 1) 
										, stride=(self.tblock_len, 1, 1)
										, padding=0, dilation = 1, bias=False 
			)
			# init weights for tdim layer
			self.Cmat_tdim.weight.data = torch.from_numpy(self.Cmat_tdim_init.transpose()[:,np.newaxis,:,np.newaxis,np.newaxis]).type(self.Cmat_tdim.weight.data.dtype)

			self.Cmat_xydim = torch.nn.Conv3d( in_channels=self.k
										, out_channels=self.k 
										, kernel_size=(1,) + self.spatial_block_dims
										, stride=(1,) + self.spatial_block_dims
										, padding=0, dilation=1, bias=False 
										, groups=self.k
			)

			self.csph_coding_layer = nn.Sequential( OrderedDict([
														('Cmat_tdim', self.Cmat_tdim)
														, ('Cmat_xydim', self.Cmat_xydim)
			]))
			expected_num_params = (self.k*self.tblock_len) + (self.k*self.spatial_block_dims[0]*self.spatial_block_dims[1])
		elif(self.encoding_type == 'full'):
			self.Cmat_txydim = torch.nn.Conv3d( in_channels=1
										, out_channels=self.k 
										, kernel_size=self.encoding_kernel_dims
										, stride=self.encoding_kernel_stride
										, padding=0, dilation = 1, bias=False 
			)
			self.csph_coding_layer = nn.Sequential( OrderedDict([
														('Cmat_txydim', self.Cmat_txydim)
			]))
			expected_num_params = self.k*self.encoding_kernel_dims[0]*self.encoding_kernel_dims[1]*self.encoding_kernel_dims[2]
		else:
			raise ValueError('Invalid encoding_type ({}) given as input.'.format(self.encoding_type))

		print("Initialization a CSPH Encoding Layer:")
		print("    - Encoding Type: {}".format(self.encoding_type))
		print("    - Encoding Kernel Dims: {}".format(self.encoding_kernel_dims))
		num_params = sum(p.numel() for p in self.csph_coding_layer.parameters())
		print("    - Num CSPH Params: {}".format(num_params))
		assert(expected_num_params == num_params), "Expected number of params does not match the coding layer params"
		self.Cmat_size = num_params


	def forward(self, inputs):
		'''
			Expected input dims == (Batch, Nt, Nr, Nc)
		'''
		## Compute compressive histogram
		B = self.csph_coding_layer(inputs)
		return B

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


	## CSPH1DGlobal2DLocalLayer4xDown CSPH Coding Object
	csph1DGlobal2DLocal4xDown_obj = CSPH1DGlobal2DLocalLayer4xDown(k=k, num_bins=nt, init='TruncFourier', optimize_weights=False)
	(outputs, csph_out) = csph1DGlobal2DLocal4xDown_obj(inputs)
	(simple_hist_output, simple_hist_csph) = csph1DGlobal2DLocal4xDown_obj(simple_hist_input)
	print("CSPH1DGlobal2DLocalLayer4xDown Layer:")
	print("    inputs1: {}".format(inputs.shape))
	print("    outputs1: {}".format(outputs.shape))
	print("    inputs2: {}".format(simple_hist_input.shape))
	print("    outputs2: {}".format(simple_hist_output.shape))

	## CSPHEncodingLayer Coding Object
	init_params = ['HybridGrayFourier','CoarseHist','TruncFourier']
	k = 32
	optimize_tdim_codes = True
	nt_blocks = 4
	spatial_block_dims = [(1,1), (2,2), (4,4)]				
	encoding_types=[ 'full', 'separable']
	for init_param in init_params:
		for spatial_block_dim in spatial_block_dims:
			for encoding_type in encoding_types:
				csph_layer_obj = CSPHEncodingLayer(k=k, num_bins=nt 
												, tblock_init=init_param, optimize_tdim_codes=optimize_tdim_codes
												, spatial_block_dims=spatial_block_dim, encoding_type=encoding_type
												, nt_blocks=nt_blocks)
				csph_out = csph_layer_obj(inputs.unsqueeze(1))

	csph_layer_weights = csph_layer_obj.csph_coding_layer[0].weight.data.cpu().numpy()
	# (simple_hist_output, simple_hist_csph) = csph1DGlobal2DLocal4xDown_obj(simple_hist_input)
	print("CSPHEncodingLayer Layer:")
	print("    inputs1: {}".format(inputs.unsqueeze(1).shape))
	print("    outputs1: {}".format(csph_out.shape))
	# print("    inputs2: {}".format(simple_hist_input.shape))
	# print("    outputs2: {}".format(simple_hist_output.shape))

