'''
	Layers that generate a 1D signal from a set of parametric inputs
'''
#### Standard Library Imports

#### Library imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports

class Gaussian1DLayer(nn.Module):
	'''
		For each input mu (assume 0 < mu < 1) we generate a 1D gaussian signal with mean=mu and unit sigma
	'''
	def __init__(self, gauss_len=1024, out_dim=-1):
		super(Gaussian1DLayer, self).__init__()

		self.gauss_len = gauss_len
		self.sigma = 1. / self.gauss_len 
		self.sigma_sq = self.sigma ** 2
		# Normalization factor
		self.a = (1. / (self.sigma * np.sqrt(2*np.pi))) / self.gauss_len
		domain = torch.arange(0, self.gauss_len, requires_grad=False) / self.gauss_len
		self.domain = torch.nn.Parameter(domain, requires_grad=False)
		self.out_dim = out_dim

	def forward(self, mu):
		loc = mu.unsqueeze(-1) - self.domain
		if(self.out_dim != -1):
			loc_reshaped = torch.moveaxis(loc, source=-1, destination=self.out_dim)
		else:
			loc_reshaped = loc
		out_gaussian = self.a*torch.exp(-0.5*torch.square(loc_reshaped / self.sigma))
		return out_gaussian

class IRF1DLayer(nn.Module):
	'''
		Convolve/Smooth the specified dimension with a 1D filter that is pre-specified
		The dimension can either be 0, 1, or 2
		If you want the convolution to happen along the first dimension of your signal then set dim=0
		If you want the convolution to happen along the last dimension of your signal then set dim=0
	
		Input dimensions: (B, 1, D0, D1, D2)
			- We assume there is only 1 channel
			- If conv_dim == 0, then convolution is done on D0 dimension
			- If conv_dim == 1, then convolution is done on D1 dimension
			- If conv_dim == 2, then convolution is done on D2 dimension
	'''
	def __init__(self, irf, conv_dim=0):
		super(IRF1DLayer, self).__init__()

		assert(irf.ndim==1), "Input IRF should have only 1 dimension"

		self.n = irf.size # Number of bins in irf
		self.is_even = int((self.n % 2) == 0)
		self.pad_l = self.n // 2
		self.pad_r = self.n // 2 - self.is_even
		breakpoint()
		if(conv_dim == 0): 
			self.kernel_dims = (self.n, 1, 1)
			self.conv_padding = (0, 0) + (0, 0) + (self.pad_l, self.pad_r)
		elif(conv_dim == 1): 
			self.kernel_dims = (1, self.n, 1)
			self.conv_padding = (0, 0) + (self.pad_l, self.pad_r) + (0, 0)
		elif(conv_dim == 2): 
			self.kernel_dims = (1, 1, self.n)
			self.conv_padding = (self.pad_l, self.pad_r) + (0, 0) + (0, 0)
		else: assert(False), "Invalid input conv_dim. conv_dim should be one of 0, 1, 2"

		# Normalize
		irf = irf / irf.sum()
		# Create a 3D convolutional filter that will effectively only operate on a single dimension
		# The filter will have the in/out channel dimensions as 1
		# The last 3 dimensions are the signal dimension
		irf_tensor = torch.from_numpy(irf.reshape((1,1)+self.kernel_dims).astype(np.float32))
		self.irf_weights = torch.nn.Parameter(irf_tensor, requires_grad=False)

	def forward(self, inputs):
		'''
			Inputs should have dims (B, 1, D0, D1, D2)
		'''
		# Pad inputs appropriately so that the convolution is the same as a circular convolution
		padded_input = F.pad(inputs, pad=self.conv_padding, mode='circular')
		# Apply convolution
		# No need to pad here since we padded above to make sure that the out has the same dimension
		out = F.conv3d(padded_input, self.irf_weights, stride=1, padding=0)
		return out


if __name__=='__main__':
	import matplotlib.pyplot as plt
	from model_utils import count_parameters
	from research_utils.signalproc_ops import gaussian_pulse

	# Set random input
	batch_size = 3
	(nr, nc, nt) = (32, 32, 1024) 
	inputs2D = torch.rand((batch_size, 1, nr, nc))

	# Eval Network Components
	inputs = inputs2D
	model = Gaussian1DLayer(gauss_len=nt, out_dim = -3)
	outputs = model(inputs)
	print("{} Parameters: {}".format(model.__class__.__name__, count_parameters(model)))
	print("    input shape: {}".format(inputs.shape))
	print("    output shape: {}".format(outputs.shape))

	# Generate Gaussian Pulse
	pulse_len = 25
	pulse_domain = np.arange(0, pulse_len*5)
	pulse = gaussian_pulse(pulse_domain, mu=pulse_domain[-1]//2, width=pulse_len, circ_shifted=True)
	irf_layer = IRF1DLayer(irf=pulse, conv_dim=0)
	outputs_irf_layer = irf_layer(outputs)

	outputs_np = outputs.cpu().numpy()
	outputs_irf_layer_np = outputs_irf_layer.cpu().numpy()
	plt.clf()
	(r,c) = (np.random.randint(0,nr), np.random.randint(0,nc))
	plt.plot(outputs_np[0,0, :, r, c])
	plt.plot(pulse)
	plt.plot(outputs_irf_layer_np[0,0,:,r,c])
	plt.title("Row: {}, Col: {}".format(r,c))



