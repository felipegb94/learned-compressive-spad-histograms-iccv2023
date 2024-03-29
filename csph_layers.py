#### Standard Library Imports
import os
from pickletools import optimize
from typing import OrderedDict
import math

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
from unfilt_backproj_3D import UnfiltBackproj3DTransposeConv
from layers_parametric1D import IRF1DLayer

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

def compute_csph3d_expected_num_params(encoding_type, tblock_len, xyblock_len, k):
	if(encoding_type == 'full'):
		num_params = tblock_len*xyblock_len*k
	elif(encoding_type == 'separable'):
		num_params = (tblock_len*k) + (xyblock_len*k)
	elif(encoding_type == 'csph1d'):
		num_params = (tblock_len*k)
	else:
		assert(False), "Invalid encoding type for csph3d"
	return num_params

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
	elif(init_id == 'BinaryRand'):
		Cmat_init = np.random.randn(num_bins, k)
		Cmat_init[Cmat_init >= 0] = 1
		Cmat_init[Cmat_init < 0] = -1
	else:
		assert(False), "Invalid CSPH1D init ID"
	return Cmat_init

def get_rand_torch_conv3d(in_ch, k, kernel3d_size, separable=False):
	# Conv kernel for spatial dims
	stride3d_size = kernel3d_size
	if(separable): groups = k
	else: groups = 1
	conv3d_layer = torch.nn.Conv3d(in_channels=in_ch
										, out_channels=k
										, kernel_size=kernel3d_size
										, stride=stride3d_size
										, groups=groups
										, padding=0, dilation=1, bias=False)
	W3d = conv3d_layer.weight.data
	return (conv3d_layer, W3d)

def get_rand_torch_conv3d_W(k, kernel3d_size, separable=False):
	'''
		Random initialization of a torch.nn.Parameter that requires grad by default
	'''
	out_ch = k
	## Set the size of the 3D convolutional kernel  
	# If not separable, we expect in_ch == 1 since this is for the first layer
	# If separable, we expect in_ch == k, so the second dim is (in_ch / groups) where groups = k, so we get the second dim as 1
	# In both cases we get the same shape so:
	conv3d_weight_shape = (out_ch, 1) + kernel3d_size
	conv3d_tensor = torch.empty(conv3d_weight_shape)
	# Although we implement W as a convolutional kernel, it really is applying a linear transform to an image block of size kernel3d_size and projecting it into a K-dimensional space. So we want to initialize things just like a linear layer where the number of input features is actually the size of the kernel 
	in_f = kernel3d_size[0]*kernel3d_size[1]*kernel3d_size[2]
	torch.nn.init.uniform_(conv3d_tensor, a=-1./math.sqrt(in_f), b=1./math.sqrt(in_f))
	conv3d_W = torch.nn.Parameter(conv3d_tensor, requires_grad=True) 
	return conv3d_W

def compute_scale(beta, alpha, beta_q, alpha_q):
	return (float(beta) - float(alpha)) / (float(beta_q) - float(alpha_q))

def compute_zero_point(scale, alpha, alpha_q):
	## Cast everything to float first to make sure that we dont input torch.tensors to zero point
	return round(-1*((float(alpha)/float(scale)) - float(alpha_q)))

def quantize_qint8_manual(X, X_range=None):
	if(X_range is None):
		(X_min, X_max) = (X.min(), X.max())
	else:
		X_min = X_range[0]
		X_max = X_range[1]
		# assert(X_min <= X.detach().cpu().numpy().min()), "minimum should be contained in range"
		# assert(X_max >= X.detach().cpu().numpy().max()), "maximum should be contained in range"
	(min_q, max_q) = (-128, 127)
	scale = compute_scale(X_max, X_min, max_q, min_q)
	zero_point = compute_zero_point(scale, X_min, min_q)
	qX = (X / scale) + zero_point
	qX_int8  = torch.round(qX).type(torch.int8)
	return  (qX_int8, scale, zero_point)

def dequantize_qint8(X_qint8, X_scale, X_zero_point):
	## Cast everything to float 32 first since X_zero_point may have different precision than X_qint8
	return (X_qint8.type(torch.float32) - float(X_zero_point))*float(X_scale)

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

class CSPH3DLayerv1(nn.Module):
	'''
		Same as CSPH3DLayer but it uses an older CSPH3D layer implementation that stores the coding matrix weights inside an nn.Conv3d module. The overall performance should be the same
	'''
	def __init__(self, k=2, num_bins=1024, 
					tblock_init='Rand', h_irf=None, optimize_tdim_codes=False, optimize_codes=True, 
					nt_blocks=1, spatial_down_factor=1, 
					encoding_type='full'
				):
		# Init parent class
		super(CSPH3DLayerv1, self).__init__()
		# Validate some input params
		assert((num_bins % nt_blocks) == 0), "Number of bins should be divisible by number of time blocks"

		self.num_bins = num_bins
		self.k = k
		self.tblock_init=tblock_init
		self.optimize_tdim_codes=optimize_tdim_codes
		self.optimize_codes=optimize_codes
		self.nt_blocks=nt_blocks
		self.tblock_len = int(self.num_bins / nt_blocks) 
		self.spatial_down_factor=spatial_down_factor
		# Pad IRF with zeros if needed
		self.h_irf = pad_h_irf(h_irf, num_bins=self.tblock_len) # This is used to select the frequencies that will be used in HybridGrayFourier
		# Get initialization matrix for the temporal dimension block
		self.Cmat_tdim_init = get_temporal_cmat_init(k=k, num_bins=self.tblock_len, init_id=tblock_init, h_irf=self.h_irf)
		# reshape matrix to have the shape of a convolutional kernel in 3D
		W1d = torch.from_numpy(self.Cmat_tdim_init.transpose()[:,np.newaxis,:,np.newaxis,np.newaxis])
		
		# Initialize the encoding layer
		self.encoding_type = encoding_type
		self.encoding_kernel_dims = (self.tblock_len, self.spatial_down_factor, self.spatial_down_factor)
		self.encoding_kernel_stride = self.encoding_kernel_dims

		print("Initialization a CSPH Encoding Layer:")
		print("    - Encoding Type: {}".format(self.encoding_type))
		print("    - Optimized tdim codes: {}".format(self.optimize_tdim_codes and self.optimize_codes))
		print("    - Optimized spatial codes: {}".format(self.optimize_codes))
		print("    - Optimized codes: {}".format(self.optimize_codes))
		print("    - Encoding Kernel Dims: {}".format(self.encoding_kernel_dims))
		print("    - Number of Filters/Codes: {}".format(self.k))

		self.unfiltered_backproj_layer = UnfiltBackproj3DTransposeConv()

		if(self.encoding_type == 'full'):
			# initalize a random conv3d layer
			(self.Cmat_txydim, _) = get_rand_torch_conv3d(1, k, self.encoding_kernel_dims, separable=False)

			# If tblock init was not rand, initialize with the Cmat_tdim_init
			if(not (self.tblock_init == 'Rand')):
				## Initialize the weights
				# change the type  for W1d
				W1d = W1d.type(self.Cmat_txydim.weight.data.dtype)
				# get a random W2d initialized with pytorch conv layer
				kernel3d_W2d_size = (1, self.encoding_kernel_dims[-2], self.encoding_kernel_dims[-1])
				(_, W2d) = get_rand_torch_conv3d(k, k, kernel3d_W2d_size, separable=True)
				# init weights
				self.Cmat_txydim.weight.data = W1d*W2d
			# optimize codes only if this flag is set
			# sometimes we may want to compare against codes that are not optimized or with random codes
			self.Cmat_txydim.weight.requires_grad = self.optimize_codes

			self.csph_layer = nn.Sequential( OrderedDict([
														('Cmat_txydim', self.Cmat_txydim)
			]))
			## Set the method to obtain the weights for unfiltered backprojection
			self.get_unfilt_backproj_W3d = self.get_unfilt_backproj_W3d_full
			expected_num_params = self.k*self.encoding_kernel_dims[0]*self.encoding_kernel_dims[1]*self.encoding_kernel_dims[2]
		elif(self.encoding_type == 'separable'):
			# Separable convolution encoding
			## Initialize time domain coding layer
			# initalize a random conv3d layer but with last 2 dims as 1 (note first layer is not init as separable)
			(self.Cmat_tdim, _) = get_rand_torch_conv3d(1, k, (self.tblock_len, 1, 1), separable=False)
			# If we use rand init, leave the matrix as is, but store the weights
			if(self.tblock_init == 'Rand'):
				self.Cmat_tdim_init = torch.clone(self.Cmat_tdim.weight.data).cpu().numpy() 
			else:
				# init weights for tdim layer
				self.Cmat_tdim.weight.data = torch.from_numpy(self.Cmat_tdim_init.transpose()[:,np.newaxis,:,np.newaxis,np.newaxis]).type(self.Cmat_tdim.weight.data.dtype)
			# Only optimize the codes if neeeded
			self.Cmat_tdim.weight.requires_grad = self.optimize_tdim_codes and self.optimize_codes

			## Initialize spatial domain encoding layer
			# get a random W2d initialized with pytorch conv layer
			kernel3d_W2d_size = (1, self.encoding_kernel_dims[-2], self.encoding_kernel_dims[-1])
			(self.Cmat_xydim, W2d) = get_rand_torch_conv3d(k, k, kernel3d_W2d_size, separable=True)
			# Only optimize code if needed
			self.Cmat_xydim.weight.requires_grad = self.optimize_codes
			## Set the method to obtain the weights for unfiltered backprojection
			self.get_unfilt_backproj_W3d = self.get_unfilt_backproj_W3d_separable
			## Set the coding layer
			self.csph_layer = nn.Sequential( OrderedDict([
														('Cmat_tdim', self.Cmat_tdim)
														, ('Cmat_xydim', self.Cmat_xydim)
			]))
			expected_num_params = (self.k*self.tblock_len) + (self.k*self.spatial_down_factor*self.spatial_down_factor)
		else:
			raise ValueError('Invalid encoding_type ({}) given as input.'.format(self.encoding_type))

		num_params = sum(p.numel() for p in self.csph_layer.parameters())
		print("    - Num CSPH Params: {}".format(num_params))
		assert(expected_num_params == num_params), "Expected number of params does not match the coding layer params"
		self.Cmat_size = num_params

	def get_unfilt_backproj_W3d_full(self):
		'''
			If we use the full coding matrix, we use these weights for unfiltered backjprojection
		'''
		return self.Cmat_txydim.weight

	def get_unfilt_backproj_W3d_separable(self):
		'''
			If we use a separable coding matrix, we use the outer product of the time and spatial domain for the unfiltered backprojection
		'''
		return self.Cmat_tdim.weight*self.Cmat_xydim.weight

	def forward(self, inputs):
		'''
			Expected input dims == (Batch, Nt, Nr, Nc)
		'''
		## Compute compressive histogram
		B = self.csph_layer(inputs)
		## Upsample using unfiltered backprojection (similar to transposed convolution, but with fixed weights)
		X = self.unfiltered_backproj_layer(y=B, W=self.get_unfilt_backproj_W3d())
		return X
class CSPHEncodingLayer(nn.Module):
	def __init__(self, k=2, num_bins=1024, 
					tblock_init='TruncFourier', h_irf=None, optimize_tdim_codes=False,
					nt_blocks=1, spatial_down_factor=1, 
					encoding_type='separable'
				):
		# Init parent class
		super(CSPHEncodingLayer, self).__init__()
		# Validate some input params
		assert((num_bins % nt_blocks) == 0), "Number of bins should be divisible by number of time blocks"

		self.num_bins = num_bins
		self.k = k
		self.tblock_init=tblock_init
		self.optimize_tdim_codes=optimize_tdim_codes
		self.nt_blocks=nt_blocks
		self.tblock_len = int(self.num_bins / nt_blocks) 
		self.spatial_down_factor=spatial_down_factor
		# Pad IRF with zeros if needed
		self.h_irf = pad_h_irf(h_irf, num_bins=self.tblock_len) # This is used to select the frequencies that will be used in HybridGrayFourier
		# Get initialization matrix for the temporal dimension block
		self.Cmat_tdim_init = get_temporal_cmat_init(k=k, num_bins=self.tblock_len, init_id=tblock_init, h_irf=self.h_irf)
		
		# Initialize the encoding layer
		self.encoding_type = encoding_type
		self.encoding_kernel_dims = (self.tblock_len, self.spatial_down_factor, self.spatial_down_factor)
		self.encoding_kernel_stride = self.encoding_kernel_dims

		print("Initialization a CSPH Encoding Layer:")
		print("    - Encoding Type: {}".format(self.encoding_type))
		print("    - Encoding Kernel Dims: {}".format(self.encoding_kernel_dims))

		if(self.encoding_type == 'separable'):
			# Separable convolution encoding
			# First, define a tblock_lenx1x1 convolution kernel
			self.Cmat_tdim = torch.nn.Conv3d( in_channels=1
										, out_channels=self.k 
										, kernel_size=(self.tblock_len, 1, 1) 
										, stride=(self.tblock_len, 1, 1)
										, padding=0, dilation = 1, bias=False 
			)
			# If we use rand init, leave the matrix as is, but store the weights
			if(self.tblock_init == 'Rand'):
				self.Cmat_tdim_init = torch.clone(self.Cmat_tdim.weight.data).cpu().numpy() 
			else:
				# init weights for tdim layer
				self.Cmat_tdim.weight.data = torch.from_numpy(self.Cmat_tdim_init.transpose()[:,np.newaxis,:,np.newaxis,np.newaxis]).type(self.Cmat_tdim.weight.data.dtype)
			# Only optimize the codes if neeeded
			self.Cmat_tdim.weight.requires_grad = self.optimize_tdim_codes

			self.Cmat_xydim = torch.nn.Conv3d( in_channels=self.k
										, out_channels=self.k 
										, kernel_size=(1, self.spatial_down_factor, self.spatial_down_factor)
										, stride=(1, self.spatial_down_factor, self.spatial_down_factor)
										, padding=0, dilation=1, bias=False 
										, groups=self.k
			)


			self.csph_coding_layer = nn.Sequential( OrderedDict([
														('Cmat_tdim', self.Cmat_tdim)
														, ('Cmat_xydim', self.Cmat_xydim)
			]))
			expected_num_params = (self.k*self.tblock_len) + (self.k*self.spatial_down_factor*self.spatial_down_factor)
		elif(self.encoding_type == 'full'):
			self.Cmat_txydim = torch.nn.Conv3d( in_channels=1
										, out_channels=self.k 
										, kernel_size=self.encoding_kernel_dims
										, stride=self.encoding_kernel_stride
										, padding=0, dilation = 1, bias=False 
			)
			if(self.tblock_init == 'Rand'):
				self.Cmat_tdim_init = torch.clone(self.Cmat_txydim.weight.data).cpu().numpy() 
			else:
				# for each row and col set to the initialization codes
				for i in range(self.spatial_down_factor):
					for j in range(self.spatial_down_factor):
						self.Cmat_txydim.weight.data[..., i, j] = torch.from_numpy(self.Cmat_tdim_init.transpose()[:,np.newaxis,:]).type(self.Cmat_txydim.weight.data.dtype)
				self.Cmat_tdim_init = torch.clone(self.Cmat_txydim.weight.data).cpu().numpy() 
			
			self.csph_coding_layer = nn.Sequential( OrderedDict([
														('Cmat_txydim', self.Cmat_txydim)
			]))
			expected_num_params = self.k*self.encoding_kernel_dims[0]*self.encoding_kernel_dims[1]*self.encoding_kernel_dims[2]
		else:
			raise ValueError('Invalid encoding_type ({}) given as input.'.format(self.encoding_type))

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

class CSPHDecodingLayer(nn.Module):
	def __init__(self, k, num_bins, nt_blocks, up_factor_xydim):
		super(CSPHDecodingLayer, self).__init__()

		self.num_bins = num_bins
		self.k = k
		self.nt_blocks=nt_blocks
		self.tblock_len = int(self.num_bins / nt_blocks) 
		self.up_factor_xydim = up_factor_xydim
		self.Nt = self.num_bins

		## Create temporal decoding block
		self.temporal_decoding_block = nn.Sequential(
											nn.Conv3d(in_channels=self.k, out_channels=self.tblock_len, kernel_size=1, bias=True),
											nn.ReLU(inplace=True)
											)
		 

		## Apply a 1x1x1 conv block for each
		decoding_block_1x1 = nn.Conv2d(in_channels=self.Nt, out_channels=self.Nt*self.up_factor_xydim*self.up_factor_xydim, kernel_size=1, bias=True)
		self.spatial_decoding_block = nn.Sequential(decoding_block_1x1, 
													nn.ReLU(inplace=True),
													nn.PixelShuffle(self.up_factor_xydim))


	def forward(self, inputs):
		'''
			Expected input dims == (Batch, k, Mt, Mr, Mc)
		'''
		## Spatial upsampled output
		(batch, k_codes, Mt, Mr, Mc) = inputs.shape
		(Nr, Nc) = (self.up_factor_xydim*Mr, self.up_factor_xydim*Mc)
		## Dims: Batch, Nt/Mt, Mt, Mr, Mc
		temporally_decoded_out = self.temporal_decoding_block(inputs)

		## Reshape appropriately so that we stack the channel dimensions (a copy needs to be made due to the reshape that is needed)
		# See test in the main function that shows why we want this type of reshape
		temporally_decoded_out_reshaped = torch.permute(temporally_decoded_out, dims=(0, 2, 1, 3, 4)).reshape((batch, self.Nt, Mr, Mc))

		## Reshape to merge the time and channel dimension
		decoded_out = self.spatial_decoding_block(temporally_decoded_out_reshaped) 

		return decoded_out.unsqueeze(1)

class CSPH3DLayer(nn.Module):
	def __init__(self, k=2, num_bins=1024, 
					tblock_init='Rand', h_irf=None, optimize_tdim_codes=False, optimize_codes=True, 
					nt_blocks=1, spatial_down_factor=1, 
					encoding_type='full',
					csph_out_norm='none',
					account_irf=False,
					smooth_tdim_codes=False,
					apply_zncc_norm=False,
					zero_mean_tdim_codes=False
				):
		# Init parent class
		super(CSPH3DLayer, self).__init__()
		# Validate some input params
		assert((num_bins % nt_blocks) == 0), "Number of bins should be divisible by number of time blocks"
		self.emulate_int8_quantization = False
		self.num_bins = num_bins
		self.k = k
		self.tblock_init=tblock_init
		self.optimize_tdim_codes=optimize_tdim_codes
		self.optimize_codes=optimize_codes
		self.nt_blocks=nt_blocks
		self.tblock_len = int(self.num_bins / nt_blocks) 
		# self.tblock_len = 1024 
		self.spatial_down_factor=spatial_down_factor
		# Create IRF layer if we want to account for h irf when doing the unfiltered backprojections
		self.irf_layer = IRF1DLayer(irf=h_irf, conv_dim=0) 
		self.account_irf = account_irf
		# Zero mean codes --> If true make time dimension code zero-mean at every iteration. This can help remove effect of ambient
		self.zero_mean_tdim_codes = zero_mean_tdim_codes
		# Smooth tdim codes --> If true it will smooth the tdim codes before encoding
		self.smooth_tdim_codes = smooth_tdim_codes
		# apply_zncc_norm --> If true it will apply the normalization used in zero-mean normalized cross-corr
		self.apply_zncc_norm = apply_zncc_norm
		if(self.apply_zncc_norm): self.zncc_norm = self.zero_norm
		else: self.zncc_norm = self.no_zero_norm
		# Pad IRF with zeros if needed. This is only used to select frequencies in the tblock initialization.
		self.h_irf = pad_h_irf(h_irf, num_bins=self.tblock_len) 
		# Get initialization matrix for the temporal dimension block
		self.Cmat_tdim_init = get_temporal_cmat_init(k=k, num_bins=self.tblock_len, init_id=tblock_init, h_irf=self.h_irf)
		# reshape matrix to have the shape of a convolutional kernel in 3D
		W1d = torch.from_numpy(self.Cmat_tdim_init.transpose()[:,np.newaxis,:,np.newaxis,np.newaxis])
		
		# Initialize the encoding layer
		self.encoding_type = encoding_type
		self.encoding_kernel3d_dims = (self.tblock_len, self.spatial_down_factor, self.spatial_down_factor)
		self.encoding_kernel3d_txy = self.encoding_kernel3d_dims
		self.encoding_kernel3d_t = (self.encoding_kernel3d_dims[0], 1, 1)
		self.encoding_kernel3d_xy = (1, self.encoding_kernel3d_dims[1], self.encoding_kernel3d_dims[2])

		print("Initialization a CSPH Encoding Layer:")
		print("    - Encoding Type: {}".format(self.encoding_type))
		print("    - csph_out_norm: {}".format(csph_out_norm))
		print("    - zero_mean_tdim_codes: {}".format(zero_mean_tdim_codes))
		print("    - smooth_tdim_codes: {}".format(smooth_tdim_codes))
		print("    - apply_zncc_norm: {}".format(apply_zncc_norm))
		print("    - account_irf: {}".format(account_irf))
		print("    - tblock_init: {}".format(self.tblock_init))
		print("    - Optimized tdim codes: {}".format(self.optimize_tdim_codes and self.optimize_codes))
		print("    - Optimized spatial codes: {}".format(self.optimize_codes))
		print("    - Optimized codes: {}".format(self.optimize_codes))
		print("    - Encoding Kernel Dims: {}".format(self.encoding_kernel3d_dims))
		print("    - Number of Filters/Codes: {}".format(self.k))

		self.unfiltered_backproj_layer = UnfiltBackproj3DTransposeConv()
		if(self.encoding_type == 'full'):
			# initalize a random conv3d layer
			self.Cmat_txydim = get_rand_torch_conv3d_W(k, self.encoding_kernel3d_dims, separable=False)
			# If tblock init was not rand, initialize with the Cmat_tdim_init
			if(not (self.tblock_init == 'Rand')):
				## Initialize the weights
				# change the type  for W1d
				W1d = W1d.type(self.Cmat_txydim.data.dtype)
				# get a random W2d initialized with pytorch conv layer
				W2d = get_rand_torch_conv3d_W(k, self.encoding_kernel3d_xy, separable=True)
				## re-init the parameter tensor
				self.Cmat_txydim = nn.Parameter(W1d*(W2d.data), requires_grad=True)
			## store the initial matrix
			self.Cmat_txydim_init = torch.clone(self.Cmat_txydim.data).cpu().numpy()
			# optimize codes only if this flag is set
			# sometimes we may want to compare against codes that are not optimized or with random codes
			self.Cmat_txydim.requires_grad = self.optimize_codes
			## Set the function to compute the compressive histogram
			self.csph_layer = self.csph_layer_full
			## Select how the coding matrix weights are constrained (zero mean and/or smooth)
			if(self.zero_mean_tdim_codes and self.smooth_tdim_codes):
				self.get_Cmat = self.get_smooth_zero_mean_full_Cmat
			elif(self.smooth_tdim_codes):
				self.get_Cmat = self.get_smooth_full_Cmat
			elif(self.zero_mean_tdim_codes):
				self.get_Cmat = self.get_zero_mean_full_Cmat
			else:
				self.get_Cmat = self.get_full_Cmat
			## Set the method to obtain the weights for unfiltered backprojection
			if(self.account_irf): 
				self.get_unfilt_backproj_W3d = self.get_unfilt_backproj_W3d_with_irf_full
			else:
				self.get_unfilt_backproj_W3d = self.get_unfilt_backproj_W3d_full
			## Compute the expected number of parameters
			expected_num_params = self.k*self.encoding_kernel3d_dims[0]*self.encoding_kernel3d_dims[1]*self.encoding_kernel3d_dims[2]
			expected_num_params_2 = compute_csph3d_expected_num_params(self.encoding_type, self.encoding_kernel3d_dims[0], self.encoding_kernel3d_dims[1]*self.encoding_kernel3d_dims[2], self.k)
		elif(self.encoding_type == 'separable'):
			## Separable convolution encoding
			## Initialize time domain coding layer
			# initalize a random conv3d layer but with last 2 dims as 1 (note first layer is not init as separable)
			self.Cmat_tdim = get_rand_torch_conv3d_W(k, self.encoding_kernel3d_t, separable=False)
			# If we use rand init, leave the matrix as is, but store a copy of the weights
			if(self.tblock_init == 'Rand'):
				self.Cmat_tdim_init = torch.clone(self.Cmat_tdim.data).cpu().numpy() 
			else:
				# init weights for tdim layer
				self.Cmat_tdim.data = torch.from_numpy(self.Cmat_tdim_init.transpose()[:,np.newaxis,:,np.newaxis,np.newaxis]).type(self.Cmat_tdim.dtype)
			# Only optimize the codes if neeeded
			self.Cmat_tdim.requires_grad = self.optimize_tdim_codes and self.optimize_codes
			## Initialize spatial domain encoding layer
			# get a random W2d initialized with pytorch conv layer
			self.Cmat_xydim = get_rand_torch_conv3d_W( k, self.encoding_kernel3d_xy, separable=True)
			self.Cmat_xydim_init = torch.clone(self.Cmat_xydim.data).cpu().numpy()
			# Only optimize code if needed
			self.Cmat_xydim.requires_grad = self.optimize_codes
			## Set the function to compute the compressive histogram
			self.csph_layer = self.csph_layer_separable
			## Select how the coding matrix weights are constrained (zero mean and/or smooth)
			if(self.zero_mean_tdim_codes and self.smooth_tdim_codes):
				self.get_Cmat = self.get_smooth_zero_mean_tdim_Cmat
			elif(self.smooth_tdim_codes):
				self.get_Cmat = self.get_smooth_tdim_Cmat
			elif(self.zero_mean_tdim_codes):
				self.get_Cmat = self.get_zero_mean_tdim_Cmat
			else:
				self.get_Cmat = self.get_tdim_Cmat
			## Set the method to obtain the weights for unfiltered backprojection
			if(self.account_irf): 
				self.get_unfilt_backproj_W3d = self.get_unfilt_backproj_W3d_with_irf_separable
			else:
				self.get_unfilt_backproj_W3d = self.get_unfilt_backproj_W3d_separable
			## Compute the expected number of parameters
			expected_num_params = (self.k*self.tblock_len) + (self.k*self.spatial_down_factor*self.spatial_down_factor)
			expected_num_params_2 = compute_csph3d_expected_num_params(self.encoding_type, self.encoding_kernel3d_dims[0], self.encoding_kernel3d_dims[1]*self.encoding_kernel3d_dims[2], self.k)
		elif(self.encoding_type == 'csph1d'):
			assert(self.spatial_down_factor == 1), 'CSPH1D encoding only works for kernels of dimes (tblock_sizex1x1)'
			## Initialize time domain coding layer
			# initalize a random conv3d layer but with last 2 dims as 1 (note first layer is not init as separable)
			self.Cmat_tdim = get_rand_torch_conv3d_W(k, self.encoding_kernel3d_t, separable=False)
			# If we use rand init, leave the matrix as is, but store a copy of the weights
			if(self.tblock_init == 'Rand'):
				self.Cmat_tdim_init = torch.clone(self.Cmat_tdim.data).cpu().numpy() 
			else:
				# init weights for tdim layer
				self.Cmat_tdim.data = torch.from_numpy(self.Cmat_tdim_init.transpose()[:,np.newaxis,:,np.newaxis,np.newaxis]).type(self.Cmat_tdim.dtype)
			# Only optimize the codes if neeeded
			self.Cmat_tdim.requires_grad = self.optimize_tdim_codes and self.optimize_codes
			## Set the function to compute the compressive histogram
			self.csph_layer = self.csph_layer_csph1d
			## Select how the coding matrix weights are constrained (zero mean and/or smooth)
			if(self.zero_mean_tdim_codes and self.smooth_tdim_codes):
				self.get_Cmat = self.get_smooth_zero_mean_tdim_Cmat
			elif(self.smooth_tdim_codes):
				self.get_Cmat = self.get_smooth_tdim_Cmat
			elif(self.zero_mean_tdim_codes):
				self.get_Cmat = self.get_zero_mean_tdim_Cmat
			else:
				self.get_Cmat = self.get_tdim_Cmat
			## Set the method to obtain the weights for unfiltered backprojection
			if(self.account_irf): 
				self.get_unfilt_backproj_W3d = self.get_unfilt_backproj_W3d_with_irf_csph1d
			else:
				self.get_unfilt_backproj_W3d = self.get_unfilt_backproj_W3d_csph1d
			## Compute the expected number of parameters
			expected_num_params = (self.k*self.tblock_len)	
			expected_num_params_2 = compute_csph3d_expected_num_params(self.encoding_type, self.encoding_kernel3d_dims[0], self.encoding_kernel3d_dims[1]*self.encoding_kernel3d_dims[2], self.k)
		else:
			raise ValueError('Invalid encoding_type ({}) given as input.'.format(self.encoding_type))
		## Verify that the number of parameters we estimated above matches what we get
		expected_num_params += self.irf_layer.n
		expected_num_params_2 += self.irf_layer.n
		num_params = sum(p.numel() for p in self.parameters())
		print("    - Num CSPH Params: {}".format(num_params))
		assert(expected_num_params == num_params), "Expected number of params does not match the coding layer params"
		assert(expected_num_params_2 == num_params), "v2 Expected number of params does not match the coding layer params"
		self.Cmat_size = num_params

		self.csph_out_norm = csph_out_norm
		if(csph_out_norm == 'none'): self.normalize_X = self.normalize_X_none
		elif(csph_out_norm == 'Cmatsize'): self.normalize_X = self.normalize_X_Cmatsize
		elif(csph_out_norm == 'Linf'): self.normalize_X = self.normalize_X_Linf
		elif(csph_out_norm == 'LinfGlobal'): self.normalize_X = self.normalize_X_LinfGlobal
		elif(csph_out_norm == 'L2'): self.normalize_X = self.normalize_X_L2
		else: assert(False), "Invalid csph_out_norm parameter for CSPH3DLayer"

	def csph_layer_full(self, inputs):
		'''
			Assume a full coding matrix representation
		'''
		B = F.conv3d(inputs, self.get_Cmat(), bias=None, stride=self.encoding_kernel3d_txy, padding=0, dilation = 1, groups = 1)
		return B

	def csph_layer_separable(self, inputs):
		'''
			Assume a separable coding matrix representation where the 1st dim is independent from 2nd and 3rd
		'''
		B1 = F.conv3d(inputs, self.get_Cmat(), bias=None, stride=self.encoding_kernel3d_t, padding=0, dilation = 1, groups = 1)
		B = F.conv3d(B1, self.Cmat_xydim, bias=None, stride=self.encoding_kernel3d_xy, padding=0, dilation = 1, groups = self.k)
		return B
	
	def enable_quantization_emulation(self):
		'''
			Change the encoding layer to a quantized one.

			This should only be enabled at test time.
		'''
		self.emulate_int8_quantization = True
		if(self.encoding_type == 'separable'):
			self.csph_layer = self.csph_layer_separable_quantized
		elif(self.encoding_type == 'csph1d'):
			self.csph_layer = self.csph_layer_csph1d_quantized
		else:
			assert(False), "invalid encoding type for quantization emulation"

	def csph_layer_separable_quantized(self, inputs):
		'''
			Assume a separable coding matrix representation where the 1st dim is independent from 2nd and 3rd

			Quantization is emulated here by going to int8 first, and then going back to float32. So the float32 numbers will have quantization errors. The reason we have to cast everyting back to float32 is because we want the output compressed histogram to be 16 or even 32-bits, and not 8-bits. Although, the weights are 8-bits, the compressed histogram should have higher bit-depth because otherwise it would overflow. Unfortunately, to my knowledge pytorch does not support a conv3d that returns a different type with a higher bit-depth to avoid overflow.
		'''
		## Convert inputs and weights to int8 and back to float 32 to emulat quantization
		# can simply cast to int8 since spad inputs are already integers btwn 0-255
		inputs_int8_dequantized = inputs.type(torch.int8).type(torch.float32)
		(Cmat_tdim_int8, Cmat_tdim_scale, Cmat_tdim_zero_point) = quantize_qint8_manual(self.get_Cmat(), X_range=(-1,1))
		Cmat_tdim_int8_dequantized = dequantize_qint8(Cmat_tdim_int8, Cmat_tdim_scale, Cmat_tdim_zero_point)
		(Cmat_xydim_int8, Cmat_xydim_scale, Cmat_xydim_zero_point) = quantize_qint8_manual(self.Cmat_xydim, X_range=(-1,1))
		Cmat_xydim_int8_dequantized = dequantize_qint8(Cmat_xydim_int8, Cmat_xydim_scale, Cmat_xydim_zero_point)
		## Run regular encoding with the emulated quantized layers
		B1 = F.conv3d(inputs_int8_dequantized, Cmat_tdim_int8_dequantized, bias=None, stride=self.encoding_kernel3d_t, padding=0, dilation = 1, groups = 1)
		B = F.conv3d(B1, Cmat_xydim_int8_dequantized, bias=None, stride=self.encoding_kernel3d_xy, padding=0, dilation = 1, groups = self.k)
		return B

	def csph_layer_csph1d(self, inputs):
		'''
			Assume a separable coding matrix representation where the 1st dim is independent from 2nd and 3rd
		'''
		B = F.conv3d(inputs, self.get_Cmat(), bias=None, stride=self.encoding_kernel3d_t, padding=0, dilation = 1, groups = 1)
		return B
	
	def csph_layer_csph1d_quantized(self, inputs):
		'''
			Assume a separable coding matrix representation where the 1st dim is independent from 2nd and 3rd
			
			Quantization is emulated here by going to int8 first, and then going back to float32. So the float32 numbers will have quantization errors. The reason we have to cast everyting back to float32 is because we want the output compressed histogram to be 16 or even 32-bits, and not 8-bits. Although, the weights are 8-bits, the compressed histogram should have higher bit-depth because otherwise it would overflow. Unfortunately, to my knowledge pytorch does not support a conv3d that returns a different type with a higher bit-depth to avoid overflow.
		'''
		## Convert inputs and weights to int8 and back to float 32 to emulat quantization
		# can simply cast to int8 since spad inputs are already integers btwn 0-255
		inputs_int8_dequantized = inputs.type(torch.int8).type(torch.float32)
		(Cmat_tdim_int8, Cmat_tdim_scale, Cmat_tdim_zero_point) = quantize_qint8_manual(self.get_Cmat(), X_range=(-1,1))
		Cmat_tdim_int8_dequantized = dequantize_qint8(Cmat_tdim_int8, Cmat_tdim_scale, Cmat_tdim_zero_point)
		## Run regular encoding with the emulated quantized layers
		B = F.conv3d(inputs_int8_dequantized, Cmat_tdim_int8_dequantized, bias=None, stride=self.encoding_kernel3d_t, padding=0, dilation = 1, groups = 1)
		return B

	def apply_irf_on_W(self, W):
		return self.irf_layer(W).unsqueeze(1)

	def get_full_Cmat(self): 
		return self.Cmat_txydim
	
	def get_zero_mean_full_Cmat(self): 
		return self.Cmat_txydim - self.Cmat_txydim.mean(dim=-3, keepdim=True)

	def get_smooth_full_Cmat(self): 
		smooth_Cmat_txydim = self.apply_irf_on_W(self.Cmat_txydim)
		return smooth_Cmat_txydim

	def get_smooth_zero_mean_full_Cmat(self): 
		smooth_Cmat_txydim = self.apply_irf_on_W(self.Cmat_txydim)
		return smooth_Cmat_txydim - smooth_Cmat_txydim.mean(dim=-3, keepdim=True)

	def get_tdim_Cmat(self): 
		return self.Cmat_tdim
	
	def get_zero_mean_tdim_Cmat(self): 
		return self.Cmat_tdim - self.Cmat_tdim.mean(dim=-3, keepdim=True)

	def get_smooth_tdim_Cmat(self): 
		smooth_Cmat_tdim = self.apply_irf_on_W(self.Cmat_tdim)
		return smooth_Cmat_tdim

	def get_smooth_zero_mean_tdim_Cmat(self): 
		smooth_Cmat_tdim = self.apply_irf_on_W(self.Cmat_tdim)
		return smooth_Cmat_tdim - smooth_Cmat_tdim.mean(dim=-3, keepdim=True)

	def get_unfilt_backproj_W3d_full(self):
		'''
			If we use the full coding matrix, we use these weights for unfiltered backjprojection
		'''
		return self.get_Cmat() # self.Cmat_txydim = nn.Conv3d

	def get_unfilt_backproj_W3d_with_irf_full(self):
		'''
			If we use the full coding matrix, we use these weights for unfiltered backjprojection
		'''
		return self.apply_irf_on_W(self.get_Cmat()) # self.Cmat_txydim = nn.Conv3d

	def get_unfilt_backproj_W3d_separable(self):
		'''
			If we use a separable coding matrix, we use the outer product of the time and spatial domain for the unfiltered backprojection
		'''
		return self.get_Cmat()*self.Cmat_xydim

	def get_unfilt_backproj_W3d_with_irf_separable(self):
		'''
			If we use a separable coding matrix, we use the outer product of the time and spatial domain for the unfiltered backprojection
		'''
		return self.apply_irf_on_W(self.get_Cmat())*self.Cmat_xydim

	def get_unfilt_backproj_W3d_csph1d(self):
		'''
			If we use a separable coding matrix, we use the outer product of the time and spatial domain for the unfiltered backprojection
		'''
		return self.get_Cmat()

	def get_unfilt_backproj_W3d_with_irf_csph1d(self):
		'''
			If we use a separable coding matrix, we use the outer product of the time and spatial domain for the unfiltered backprojection
		'''
		return self.apply_irf_on_W(self.get_Cmat())

	def normalize_X_Linf(self, X):
		return X / (torch.norm(X, p=float('inf'), dim=-3, keepdim=True) + 1e-5)

	def normalize_X_LinfGlobal(self, X):
		return X / (torch.norm(X, p=float('inf'), dim=(-1,-2,-3), keepdim=True) + 1e-5)

	def normalize_X_L2(self, X):
		return X / (torch.norm(X, p=2, dim=-3, keepdim=True) + 1e-5)	

	def normalize_X_Cmatsize(self, X):
		return X / self.Cmat_size

	def normalize_X_none(self, X):
		return X

	def zero_norm(self, X, dims):
		'''
			Apply Zero-Normalization as it is done in Zero-normalization Cross-Corrlation
		'''
		X_zero_mean = X - X.mean(dim=dims, keepdim=True)
		return (X_zero_mean) / (torch.norm(X_zero_mean, p=2, dim=dims, keepdim=True) + 1e-5)	

	def no_zero_norm(self, X, dims): 
		return X

	def forward(self, inputs):
		'''
			Expected input dims == (Batch, Nt, Nr, Nc)
		'''
		## Compute compressive histogram
		B = self.csph_layer(inputs)
		# Get the weights from the first layer (if separable this is the outerproduct of two matrices)
		W = self.get_unfilt_backproj_W3d()
		## Normalize across the channel dimension like in ZNCC if the apply_zncc_norm is enabled
		B_norm = self.zncc_norm(B, dims=-4) # for 3D signals the channel dimension is -4 dimension 
		W_norm = self.zncc_norm(W, dims=0) # for weights the channel dimension is the first channel
		## Upsample using unfiltered backprojection (similar to transposed convolution, but with fixed weights)
		X = self.unfiltered_backproj_layer(y=B_norm, W=W_norm)
		## Normalize X with the specified normalization function
		X = self.normalize_X(X)
		return X

class CSPH3DLayerv2(nn.Module):
	def __init__(self, k=2, num_bins=1024, 
					tblock_init='Rand', h_irf=None, optimize_tdim_codes=False, optimize_codes=True, 
					nt_blocks=1, spatial_down_factor=1, 
					encoding_type='full',
					csph_out_norm='none'
				):
		# Init parent class
		super(CSPH3DLayerv2, self).__init__()
		# Validate some input params
		assert((num_bins % nt_blocks) == 0), "Number of bins should be divisible by number of time blocks"

		self.num_bins = num_bins
		self.k = k
		self.tblock_init=tblock_init
		self.optimize_tdim_codes=optimize_tdim_codes
		self.optimize_codes=optimize_codes
		self.nt_blocks=nt_blocks
		# self.tblock_len = int(self.num_bins / nt_blocks) 
		self.tblock_len = 1024 
		self.spatial_down_factor=spatial_down_factor
		# Pad IRF with zeros if needed
		self.h_irf = pad_h_irf(h_irf, num_bins=self.tblock_len) # This is used to select the frequencies that will be used in HybridGrayFourier
		# Get initialization matrix for the temporal dimension block
		self.Cmat_tdim_init = get_temporal_cmat_init(k=k, num_bins=self.tblock_len, init_id=tblock_init, h_irf=self.h_irf)
		# reshape matrix to have the shape of a convolutional kernel in 3D
		W1d = torch.from_numpy(self.Cmat_tdim_init.transpose()[:,np.newaxis,:,np.newaxis,np.newaxis])
		
		# Initialize the encoding layer
		self.encoding_type = encoding_type
		self.encoding_kernel3d_dims = (self.tblock_len, self.spatial_down_factor, self.spatial_down_factor)
		self.encoding_kernel3d_txy = self.encoding_kernel3d_dims
		self.encoding_kernel3d_t = (self.encoding_kernel3d_dims[0], 1, 1)
		self.encoding_kernel3d_xy = (1, self.encoding_kernel3d_dims[1], self.encoding_kernel3d_dims[2])

		print("Initialization a CSPH Encoding Layer:")
		print("    - Encoding Type: {}".format(self.encoding_type))
		print("    - csph_out_norm: {}".format(csph_out_norm))
		print("    - tblock_init: {}".format(self.tblock_init))
		print("    - Optimized tdim codes: {}".format(self.optimize_tdim_codes and self.optimize_codes))
		print("    - Optimized spatial codes: {}".format(self.optimize_codes))
		print("    - Optimized codes: {}".format(self.optimize_codes))
		print("    - Encoding Kernel Dims: {}".format(self.encoding_kernel3d_dims))
		print("    - Number of Filters/Codes: {}".format(self.k))

		self.unfiltered_backproj_layer = UnfiltBackproj3DTransposeConv()
		if(self.encoding_type == 'full'):
			# initalize a random conv3d layer
			self.Cmat_txydim = get_rand_torch_conv3d_W(k, self.encoding_kernel3d_dims, separable=False)
			# If tblock init was not rand, initialize with the Cmat_tdim_init
			if(not (self.tblock_init == 'Rand')):
				## Initialize the weights
				# change the type  for W1d
				W1d = W1d.type(self.Cmat_txydim.data.dtype)
				# get a random W2d initialized with pytorch conv layer
				W2d = get_rand_torch_conv3d_W(k, self.encoding_kernel3d_xy, separable=True)
				## re-init the parameter tensor
				self.Cmat_txydim = nn.Parameter(W1d*(W2d.data), requires_grad=True)
			## store the initial matrix
			self.Cmat_txydim_init = torch.clone(self.Cmat_txydim.data).cpu().numpy()
			# optimize codes only if this flag is set
			# sometimes we may want to compare against codes that are not optimized or with random codes
			self.Cmat_txydim.requires_grad = self.optimize_codes
			## Set the function to compute the compressive histogram
			self.csph_layer = self.csph_layer_full
			## Set the method to obtain the weights for unfiltered backprojection
			self.get_unfilt_backproj_W3d = self.get_unfilt_backproj_W3d_full
			## Compute the expected number of parameters
			expected_num_params = self.k*self.encoding_kernel3d_dims[0]*self.encoding_kernel3d_dims[1]*self.encoding_kernel3d_dims[2]
		elif(self.encoding_type == 'separable'):
			## Separable convolution encoding
			## Initialize time domain coding layer
			# initalize a random conv3d layer but with last 2 dims as 1 (note first layer is not init as separable)
			self.Cmat_tdim = get_rand_torch_conv3d_W(k, self.encoding_kernel3d_t, separable=False)
			# If we use rand init, leave the matrix as is, but store a copy of the weights
			if(self.tblock_init == 'Rand'):
				self.Cmat_tdim_init = torch.clone(self.Cmat_tdim.data).cpu().numpy() 
			else:
				# init weights for tdim layer
				self.Cmat_tdim.data = torch.from_numpy(self.Cmat_tdim_init.transpose()[:,np.newaxis,:,np.newaxis,np.newaxis]).type(self.Cmat_tdim.dtype)
			# Only optimize the codes if neeeded
			self.Cmat_tdim.requires_grad = self.optimize_tdim_codes and self.optimize_codes
			## Initialize spatial domain encoding layer
			# get a random W2d initialized with pytorch conv layer
			self.Cmat_xydim = get_rand_torch_conv3d_W( k, self.encoding_kernel3d_xy, separable=True)
			self.Cmat_xydim_init = torch.clone(self.Cmat_xydim.data).cpu().numpy()
			# Only optimize code if needed
			self.Cmat_xydim.requires_grad = self.optimize_codes
			## Set the function to compute the compressive histogram
			self.csph_layer = self.csph_layer_separable
			## Set the method to obtain the weights for unfiltered backprojection
			self.get_unfilt_backproj_W3d = self.get_unfilt_backproj_W3d_separable
			## Compute the expected number of parameters
			expected_num_params = (self.k*self.tblock_len) + (self.k*self.spatial_down_factor*self.spatial_down_factor)
		elif(self.encoding_type == 'csph1d'):
			assert(self.spatial_down_factor == 1), 'CSPH1D encoding only works for kernels of dimes (tblock_sizex1x1)'
			## Initialize time domain coding layer
			# initalize a random conv3d layer but with last 2 dims as 1 (note first layer is not init as separable)
			self.Cmat_tdim = get_rand_torch_conv3d_W(k, self.encoding_kernel3d_t, separable=False)
			# If we use rand init, leave the matrix as is, but store a copy of the weights
			if(self.tblock_init == 'Rand'):
				self.Cmat_tdim_init = torch.clone(self.Cmat_tdim.data).cpu().numpy() 
			else:
				# init weights for tdim layer
				self.Cmat_tdim.data = torch.from_numpy(self.Cmat_tdim_init.transpose()[:,np.newaxis,:,np.newaxis,np.newaxis]).type(self.Cmat_tdim.dtype)
			# Only optimize the codes if neeeded
			self.Cmat_tdim.requires_grad = self.optimize_tdim_codes and self.optimize_codes
			## Set the function to compute the compressive histogram
			self.csph_layer = self.csph_layer_csph1d
			## Set the method to obtain the weights for unfiltered backprojection
			self.get_unfilt_backproj_W3d = self.get_unfilt_backproj_W3d_csph1d
			## Compute the expected number of parameters
			expected_num_params = (self.k*self.tblock_len)	
		else:
			raise ValueError('Invalid encoding_type ({}) given as input.'.format(self.encoding_type))
		## Verify that the number of parameters we estimated above matches what we get
		num_params = sum(p.numel() for p in self.parameters())
		print("    - Num CSPH Params: {}".format(num_params))
		assert(expected_num_params == num_params), "Expected number of params does not match the coding layer params"
		self.Cmat_size = num_params

		self.csph_out_norm = csph_out_norm
		if(csph_out_norm == 'none'): self.normalize_X = self.normalize_X_none
		elif(csph_out_norm == 'Cmatsize'): self.normalize_X = self.normalize_X_Cmatsize
		elif(csph_out_norm == 'Linf'): self.normalize_X = self.normalize_X_Linf
		elif(csph_out_norm == 'LinfGlobal'): self.normalize_X = self.normalize_X_LinfGlobal
		elif(csph_out_norm == 'L2'): self.normalize_X = self.normalize_X_L2
		else: assert(False), "Invalid csph_out_norm parameter for CSPH3DLayer"

	def csph_layer_full(self, inputs):
		'''
			Assume a full coding matrix representation
		'''
		B = F.conv3d(inputs, self.Cmat_txydim, bias=None, stride=self.encoding_kernel3d_txy, padding=0, dilation = 1, groups = 1)
		return B

	def csph_layer_separable(self, inputs):
		'''
			Assume a separable coding matrix representation where the 1st dim is independent from 2nd and 3rd
		'''
		B1 = F.conv3d(inputs, self.Cmat_tdim, bias=None, stride=self.encoding_kernel3d_t, padding=0, dilation = 1, groups = 1)
		B = F.conv3d(B1, self.Cmat_xydim, bias=None, stride=self.encoding_kernel3d_xy, padding=0, dilation = 1, groups = self.k)
		return B

	def csph_layer_csph1d(self, inputs):
		'''
			Assume a separable coding matrix representation where the 1st dim is independent from 2nd and 3rd
		'''
		B = F.conv3d(inputs, self.Cmat_tdim, bias=None, stride=self.encoding_kernel3d_t, padding=0, dilation = 1, groups = 1)
		return B

	def get_unfilt_backproj_W3d_full(self):
		'''
			If we use the full coding matrix, we use these weights for unfiltered backjprojection
		'''
		return self.Cmat_txydim # self.Cmat_txydim = nn.Conv3d

	def get_unfilt_backproj_W3d_separable(self):
		'''
			If we use a separable coding matrix, we use the outer product of the time and spatial domain for the unfiltered backprojection
		'''
		return self.Cmat_tdim*self.Cmat_xydim

	def get_unfilt_backproj_W3d_csph1d(self):
		'''
			If we use a separable coding matrix, we use the outer product of the time and spatial domain for the unfiltered backprojection
		'''
		return self.Cmat_tdim

	def normalize_X_Linf(self, X):
		return X / (torch.norm(X, p=float('inf'), dim=-3, keepdim=True) + 1e-5)

	def normalize_X_LinfGlobal(self, X):
		return X / (torch.norm(X, p=float('inf'), dim=(-1,-2,-3), keepdim=True) + 1e-5)

	def normalize_X_L2(self, X):
		return X / (torch.norm(X, p=2, dim=-3, keepdim=True) + 1e-5)	

	def normalize_X_Cmatsize(self, X):
		return X / self.Cmat_size

	def normalize_X_none(self, X):
		return X

	def forward(self, inputs):
		'''
			Expected input dims == (Batch, Nt, Nr, Nc)
		'''
		## Compute compressive histogram
		B = self.csph_layer(inputs)
		# Get the weights from the first layer (if separable this is the outerproduct of two matrices)
		W = self.get_unfilt_backproj_W3d()
		## Upsample using unfiltered backprojection (similar to transposed convolution, but with fixed weights)
		X = self.unfiltered_backproj_layer(y=B, W=W)
		## Normalize X with the specified normalization function
		X = self.normalize_X(X)
		return X

if __name__=='__main__':
	import matplotlib.pyplot as plt

	# Select device
	use_gpu = False
	if(torch.cuda.is_available() and use_gpu): device = torch.device("cuda:0")
	else: device = torch.device("cpu")
	
	# Set random input
	k=32
	optimize_tdim_codes = True
	optimize_codes = True
	batch_size = 4
	nt_blocks = 1
	spatial_down_factor = 4		
	(nr, nc, nt) = (32, 32, 1024) 
	# (nr, nc, nt) = (35, 39, 1536) 
	inputs = torch.randn((batch_size, nt, nr, nc)).to(device) + 2
	inputs[inputs<2] = 0

	simple_hist_input = torch.zeros((2, nt, nr, nc)).to(device)
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
	csph1D_obj.to(device=device)
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
	csph1D2D_obj.to(device=device)
	(outputs, csph_out) = csph1D2D_obj(inputs)
	(simple_hist_output, simple_hist_csph) = csph1D2D_obj(simple_hist_input)
	print("CSPH1D2D Layer:")
	print("    inputs1: {}".format(inputs.shape))
	print("    outputs1: {}".format(outputs.shape))
	print("    inputs2: {}".format(simple_hist_input.shape))
	print("    outputs2: {}".format(simple_hist_output.shape))


	plt.clf()
	plt.subplot(2,1,1)
	plt.plot(simple_hist_input[0,:,0,0].cpu().numpy())
	plt.plot(simple_hist_output[0,:,0,0].detach().cpu().numpy())
	plt.plot(simple_hist_output[0,:,1,1].detach().cpu().numpy())

	plt.subplot(2,1,2)
	plt.plot(simple_hist_input[1,:,0,0].cpu().numpy())
	plt.plot(simple_hist_output[1,:,0,0].detach().cpu().numpy())
	plt.plot(simple_hist_output[1,:,1,1].detach().cpu().numpy())


	## CSPH1DGlobal2DLocalLayer4xDown CSPH Coding Object
	csph1DGlobal2DLocal4xDown_obj = CSPH1DGlobal2DLocalLayer4xDown(k=k, num_bins=nt, init='TruncFourier', optimize_weights=False)
	csph1DGlobal2DLocal4xDown_obj.to(device=device)
	(outputs, csph_out) = csph1DGlobal2DLocal4xDown_obj(inputs)
	(simple_hist_output, simple_hist_csph) = csph1DGlobal2DLocal4xDown_obj(simple_hist_input)
	print("CSPH1DGlobal2DLocalLayer4xDown Layer:")
	print("    inputs1: {}".format(inputs.shape))
	print("    outputs1: {}".format(outputs.shape))
	print("    inputs2: {}".format(simple_hist_input.shape))
	print("    outputs2: {}".format(simple_hist_output.shape))

	## CSPHEncodingLayer Coding Object
	init_params = ['HybridGrayFourier','CoarseHist','TruncFourier']
	k = k
	optimize_tdim_codes = optimize_tdim_codes
	nt_blocks = nt_blocks
	spatial_down_factors = [1, 2, 4]				
	encoding_types=[ 'full', 'separable']
	for init_param in init_params:
		for spatial_block_dim in spatial_down_factors:
			for encoding_type in encoding_types:
				csph_layer_obj = CSPHEncodingLayer(k=k, num_bins=nt 
												, tblock_init=init_param, optimize_tdim_codes=optimize_tdim_codes
												, spatial_down_factor=spatial_block_dim, encoding_type=encoding_type
												, nt_blocks=nt_blocks)
				csph_layer_obj.to(device=device)
				csph_out = csph_layer_obj(inputs.unsqueeze(1))

	csph_layer_weights = csph_layer_obj.csph_coding_layer[0].weight.data.cpu().numpy()
	# (simple_hist_output, simple_hist_csph) = csph1DGlobal2DLocal4xDown_obj(simple_hist_input)
	print("CSPHEncodingLayer Layer:")
	print("    inputs1: {}".format(inputs.unsqueeze(1).shape))
	print("    outputs1: {}".format(csph_out.shape))
	# print("    inputs2: {}".format(simple_hist_input.shape))
	# print("    outputs2: {}".format(simple_hist_output.shape))

	csph_decoding_obj = CSPHDecodingLayer(k=k, num_bins=nt, nt_blocks=csph_layer_obj.nt_blocks, up_factor_xydim=csph_layer_obj.spatial_down_factor)
	csph_decoding_obj.to(device=device)
	csph_decoded_out = csph_decoding_obj(csph_out)
	print("CSPHDecodingLayer Layer:")
	print("    inputs1: {}".format(csph_out.shape))
	print("    outputs1: {}".format(csph_decoded_out.shape))

	## Test output of csph 
	temporally_decoded_csph = csph_decoding_obj.temporal_decoding_block(csph_out)
	(batch, tblock_len, Mt, Mr, Mc) = temporally_decoded_csph.shape
	# temporally_decoded_csph_reshaped = temporally_decoded_csph.view((batch, tblock_len*Mt, Mr, Mc))
	# temporally_decoded_csph_reshaped = torch.reshape(temporally_decoded_csph, (batch, tblock_len*Mt, Mr, Mc))
	# temporally_decoded_csph_reshaped = torch.reshape(temporally_decoded_csph, (batch, Mt, tblock_len, Mr, Mc)).reshape()
	temporally_decoded_csph_reshaped = torch.permute(temporally_decoded_csph, dims=(0, 2, 1, 3, 4)).reshape((batch, tblock_len*Mt, Mr, Mc))

	for i in range(Mt):
		start_idx = i*tblock_len
		end_idx = start_idx + tblock_len
		
		block1 = temporally_decoded_csph[:, :, i, :, :]
		block2 = temporally_decoded_csph_reshaped[:, start_idx:end_idx, :, :]
		abs_diff = torch.abs(block1-block2)
		print("Diff Stats:")
		print("    Max = {}".format(abs_diff.flatten().max()))
		print("    Min = {}".format(abs_diff.flatten().min()))
		print("    Mean = {}".format(abs_diff.flatten().mean()))
		
	from research_utils.timer import Timer
	csph3d_full_layer_obj = CSPH3DLayer(k=k, num_bins=nt, tblock_init=init_param, optimize_tdim_codes=optimize_tdim_codes, optimize_codes=optimize_codes, 	spatial_down_factor=spatial_down_factor, encoding_type='full', nt_blocks=nt_blocks)
	csph3d_full_layer_obj.to(device=device)
	with Timer("CSPH3DLayer full"):
		csph3d_layer_out = csph3d_full_layer_obj(inputs.unsqueeze(1))
		torch.cuda.synchronize()

	csph3d_separable_layer_obj = CSPH3DLayer(k=k, num_bins=nt , tblock_init=init_param, optimize_tdim_codes=optimize_tdim_codes, optimize_codes=optimize_codes, spatial_down_factor=spatial_down_factor, encoding_type='separable', nt_blocks=nt_blocks)
	csph3d_separable_layer_obj.to(device=device)
	with Timer("CSPH3DLayer separable"):
		csph3d_layer_out = csph3d_separable_layer_obj(inputs.unsqueeze(1))
		torch.cuda.synchronize()

	plt.clf()
	plt.subplot(3,1,1)
	plt.plot(csph3d_full_layer_obj.Cmat_txydim.data.detach().cpu().numpy()[0,0,:,0,0], label='{}-full-k=0-0,0'.format(init_param))
	plt.plot(csph3d_separable_layer_obj.Cmat_tdim.data.detach().cpu().numpy()[0,0,:,0,0], label='{}-separable-tdim-k=0'.format(init_param))
	plt.plot(csph3d_separable_layer_obj.Cmat_xydim.data.detach().cpu().numpy()[0,0,0,0,0]*csph3d_separable_layer_obj.Cmat_tdim.data.detach().cpu().numpy()[0,0,:,0,0], label='{}-separable-txydim-k=0-0,0'.format(init_param))
	plt.legend()
	plt.subplot(3,1,2)
	plt.plot(csph3d_full_layer_obj.Cmat_txydim.data.detach().cpu().numpy()[0,0,:,1,1], label='{}-full-k=0-1,1'.format(init_param))
	plt.plot(csph3d_separable_layer_obj.Cmat_tdim.data.detach().cpu().numpy()[0,0,:,0,0], label='{}-separable-tdim-k=0'.format(init_param))
	plt.plot(csph3d_separable_layer_obj.Cmat_xydim.data.detach().cpu().numpy()[0,0,0,1,1]*csph3d_separable_layer_obj.Cmat_tdim.data.detach().cpu().numpy()[0,0,:,0,0], label='{}-separable-txydim-k=0-1,1'.format(init_param))
	plt.legend()
	plt.subplot(3,1,3)
	plt.plot(csph3d_full_layer_obj.Cmat_txydim.data.detach().cpu().numpy()[2,0,:,1,1], label='{}-full-k=2-1,1'.format(init_param))
	plt.plot(csph3d_separable_layer_obj.Cmat_tdim.data.detach().cpu().numpy()[2,0,:,0,0], label='{}-separable-tdim-k=2'.format(init_param))
	plt.plot(csph3d_separable_layer_obj.Cmat_xydim.data.detach().cpu().numpy()[2,0,0,1,1]*csph3d_separable_layer_obj.Cmat_tdim.data.detach().cpu().numpy()[2,0,:,0,0], label='{}-separable-txydim-k=2-1,1'.format(init_param))
	plt.legend()