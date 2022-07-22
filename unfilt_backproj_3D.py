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

