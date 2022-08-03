'''
Different PyTorch implementations of an unfiltered backprojection applied to a coded 3D signal.

All implementations assume the following process:
1. 3D signal is divided into equally sized non-overlapping 3D blocks
2. We apply a coded projection on each block individually with a coding matrix
	2.1 This can be applied as a non-overlapping strided convolution where the coding matrix are the filters
3. The unfiltered backprojection is done for each non-overlapping 3D block to recover a 3D signal that is the same dimensions as the input
'''
#### Standard Library Imports

#### Library imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports

def get_nonoverlapping_start_end_idx(block_idx, block_size):
	start_idx = block_idx*block_size
	end_idx = start_idx + block_size
	return (start_idx, end_idx)

def verify_input_tensors(y, W):
	assert(y.ndim == 4 or y.ndim == 5), "Invalid input y dims"
	assert(W.ndim == 5 ), "Invalid input W dims"
	assert(y.shape[-4] == W.shape[0]), "Number of filters in W should match number of channels of y"

class UnfiltBackproj3DTransposeConv(nn.Module):
	def __init__(self):
		super(UnfiltBackproj3DTransposeConv, self).__init__()

	def forward(self, y, W):
		'''
			y: coded 3D signal --> (batch_size, K, n1, n2, n3)
			W: coding matrix - 3D conv filters applied to the input to get y --> (K, 1, b1, b2, b3)
			K: number of channels == number of coded measurements/projections of signal x
		'''
		verify_input_tensors(y, W)
		kernel3d_size = W.shape[-3:]
		xhat = F.conv_transpose3d(y, weight=W, bias=None, stride=kernel3d_size)
		return xhat

class UnfiltBackproj3DForLoop(nn.Module):
	def __init__(self):
		super(UnfiltBackproj3DForLoop, self).__init__()

	def forward(self, y, W):
		'''
			y: coded 3D signal --> (batch_size, K, n1, n2, n3)
			W: coding matrix - 3D conv filters applied to the input to get y --> (K, 1, b1, b2, b3)
			K: number of channels == number of coded measurements/projections of signal x
		'''
		verify_input_tensors(y, W)
		# Number of blocks in each dimension of the coded signal
		(n1, n2, n3) = y.shape[-3:]
		# Size of the block along each dimension
		(b1, b2, b3) = W.shape[-3:]
		# Other dimensions
		if(y.ndim == 4): y = y.unsqueeze(0)
		batch_size = y.shape[0]
		K = y.shape[1]
		# Original signal dimensions
		(dim1, dim2, dim3) = (b1*n1, b2*n2, b3*n3)
		## Unfiltered backprojection FULL (for loop implementation)
		xhat = torch.zeros((batch_size, 1, dim1, dim2, dim3), device=y.device)
		for d1 in range(n1):
			(start_d1_idx, end_d1_idx) = get_nonoverlapping_start_end_idx(d1, b1)  
			for d2 in range(n2):
				(start_d2_idx, end_d2_idx) = get_nonoverlapping_start_end_idx(d2, b2)  
				for d3 in range(n3):
					(start_d3_idx, end_d3_idx) = get_nonoverlapping_start_end_idx(d3, b3)  
					# Compute unfiltered backprojection for current block
					xhat_curr = torch.sum(y[:,:,d1:d1+1,d2:d2+1,d3:d3+1]*W.squeeze(1).unsqueeze(0), dim=1, keepdim=True)
					# Concatenate the result for each block
					xhat[:, :, start_d1_idx:end_d1_idx, start_d2_idx:end_d2_idx, start_d3_idx:end_d3_idx] = xhat_curr
		return xhat

class UnfiltBackproj3DBatch(nn.Module):
	def __init__(self):
		super(UnfiltBackproj3DBatch, self).__init__()

	def forward(self, y, W):
		'''
			y: coded 3D signal --> (batch_size, K, n1, n2, n3)
			W: coding matrix - 3D conv filters applied to the input to get y --> (K, 1, b1, b2, b3)
			K: number of channels == number of coded measurements/projections of signal x
		'''
		# Number of blocks in each dimension of the coded signal
		(n1, n2, n3) = y.shape[-3:]
		# Size of the block along each dimension
		(b1, b2, b3) = W.shape[-3:]
		# Upsample y 
		nn_up = torch.nn.Upsample(scale_factor=(b1,b2,b3), mode='nearest')
		y_up = nn_up(y)
		# Repeat the filters to apply them to each block individually
		W_up = W.repeat((1,1, n1, n2, n3))
		xhat = torch.sum(y_up*W_up.squeeze(1).unsqueeze(0), dim=1, keepdim=True)
		return xhat

class UnfiltBackproj3DSeparable1D2DForLoop(nn.Module):
	def __init__(self, apply_W1d_first=True):
		super(UnfiltBackproj3DSeparable1D2DForLoop, self).__init__()

		self.apply_W1d_first=apply_W1d_first
		if(apply_W1d_first): 
			print("Initializing Separable Unfilt Backproj with 1D first")
			self.unfilt_backproj = self.unfilt_backproj_1D2D
		else: 
			print("Initializing Separable Unfilt Backproj with 2D first")
			self.unfilt_backproj = self.unfilt_backproj_2D1D
		
	def unfilt_backproj_1D2D(self, y_block, W1d, W2d):
		return torch.sum((y_block*W1d.squeeze(1).unsqueeze(0))*W2d.squeeze(1).unsqueeze(0), dim=1, keepdim=True)
	
	def unfilt_backproj_2D1D(self, y_block, W1d, W2d):	
		return torch.sum((y_block*W2d.squeeze(1).unsqueeze(0))*W1d.squeeze(1).unsqueeze(0), dim=1, keepdim=True)

	def forward(self, y, W1d, W2d):
		'''
			y: coded 3D signal --> (batch_size, K, n1, n2, n3)
			W: coding matrix - 3D conv filters applied to the input to get y --> (K, 1, b1, b2, b3)
			K: number of channels == number of coded measurements/projections of signal x
		'''
		verify_input_tensors(y, W1d)
		verify_input_tensors(y, W2d)
		# Number of blocks in each dimension of the coded signal
		(n1, n2, n3) = y.shape[-3:]
		# Size of the block along each dimension
		(b1, _, _) = W1d.shape[-3:]
		(_, b2, b3) = W2d.shape[-3:]
		# Other dimensions
		if(y.ndim == 4): y = y.unsqueeze(0)
		batch_size = y.shape[0]
		K = y.shape[1]
		# Original signal dimensions
		(dim1, dim2, dim3) = (b1*n1, b2*n2, b3*n3)
		## Unfiltered backprojection FULL (for loop implementation)
		xhat = torch.zeros((batch_size, 1, dim1, dim2, dim3), device=y.device)
		for d1 in range(n1):
			(start_d1_idx, end_d1_idx) = get_nonoverlapping_start_end_idx(d1, b1)  
			for d2 in range(n2):
				(start_d2_idx, end_d2_idx) = get_nonoverlapping_start_end_idx(d2, b2)  
				for d3 in range(n3):
					(start_d3_idx, end_d3_idx) = get_nonoverlapping_start_end_idx(d3, b3)  
					# Compute unfiltered backprojection for current block
					xhat_curr = self.unfilt_backproj(y[:,:,d1:d1+1,d2:d2+1,d3:d3+1], W1d, W2d)
					# Concatenate the result for each block
					xhat[:, :, start_d1_idx:end_d1_idx, start_d2_idx:end_d2_idx, start_d3_idx:end_d3_idx] = xhat_curr
		return xhat

class UnfiltBackproj3DSeparable1D2DTransposeConv(nn.Module):
	def __init__(self, apply_W1d_first):
		super(UnfiltBackproj3DSeparable1D2DTransposeConv, self).__init__()
		
		self.apply_W1d_first=apply_W1d_first
		if(apply_W1d_first): 
			self.unfilt_backproj = self.unfilt_backproj_1D2D
		else: 
			self.unfilt_backproj = self.unfilt_backproj_2D1D
	
	def unfilt_backproj_1D2D(self, y, W1d, W2d, stride3d_1d, stride3d_2d, k):
		xhat_1d = F.conv_transpose3d(y, weight=W1d, bias=None, stride=stride3d_1d, groups=k)
		xhat = F.conv_transpose3d(xhat_1d, weight=W2d, bias=None, stride=stride3d_2d)
		return xhat
	
	def unfilt_backproj_2D1D(self, y, W1d, W2d, stride3d_1d, stride3d_2d, k):	
		xhat_2d = F.conv_transpose3d(y, weight=W2d, bias=None, stride=stride3d_2d, groups=k)
		xhat = F.conv_transpose3d(xhat_2d, weight=W1d, bias=None, stride=stride3d_1d)
		return xhat
	
	def forward(self, y, W1d, W2d):
		'''
			y: coded 3D signal --> (batch_size, K, n1, n2, n3)
			W: coding matrix - 3D conv filters applied to the input to get y --> (K, 1, b1, b2, b3)
			K: number of channels == number of coded measurements/projections of signal x
		'''
		verify_input_tensors(y, W1d)
		verify_input_tensors(y, W2d)
		kernel3d_1d_size = W1d.shape[-3:]
		kernel3d_2d_size = W2d.shape[-3:]
		k = y.shape[-4]
		xhat = self.unfilt_backproj(y, W1d, W2d, kernel3d_1d_size, kernel3d_2d_size, k)
		return xhat