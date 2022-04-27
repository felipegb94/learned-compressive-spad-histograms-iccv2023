'''
	Layers that generate a 1D signal from a set of parametric inputs
'''
#### Standard Library Imports

#### Library imports
import numpy as np
import torch
import torch.nn as nn
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


if __name__=='__main__':
	import matplotlib.pyplot as plt
	from model_utils import count_parameters

	# Set random input
	batch_size = 3
	(nr, nc, nt) = (32, 32, 1024) 
	in_ch  = 2
	inputs2D = torch.rand((batch_size, in_ch, nr, nc))

	# Eval Network Components
	inputs = inputs2D
	model = Gaussian1DLayer(gauss_len=nt, out_dim = -3)
	outputs = model(inputs)
	print("{} Parameters: {}".format(model.__class__.__name__, count_parameters(model)))
	print("    input shape: {}".format(inputs.shape))
	print("    output shape: {}".format(outputs.shape))

	outputs_np = outputs.cpu().numpy()
	plt.clf()
	(r,c) = (np.random.randint(0,nr), np.random.randint(0,nc))
	plt.plot(outputs_np[0,0, :, r, c])
	plt.title("Row: {}, Col: {}".format(r,c))

