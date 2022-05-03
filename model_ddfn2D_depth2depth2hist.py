#### Standard Library Imports

#### Library imports
from random import gauss
import torch
import torch.nn as nn
import torch.nn.init as init
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_base_spad_lit import LITBaseSPADModel
from layers_parametric1D import Gaussian1DLayer, IRF1DLayer
from model_ddfn2D_depth2depth import PlainDeepBoosting2DDepth2Depth

class PlainDeepBoosting2DDepth2Depth2Hist(PlainDeepBoosting2DDepth2Depth):
	def __init__(self, irf, in_channels=1, outchannel_MS=2, n_ddfn_blocks=10, num_bins=1024):
		
		super(PlainDeepBoosting2DDepth2Depth2Hist, self).__init__(
			in_channels=in_channels, 
			outchannel_MS=outchannel_MS, 
			n_ddfn_blocks=n_ddfn_blocks, 
			num_bins=num_bins
		)

		## Replace the hist reconstruction layer
		gauss_layer = Gaussian1DLayer(gauss_len=num_bins, out_dim=-3)
		irf_layer = IRF1DLayer(irf, conv_dim=0)
		self.hist_rec_layer = torch.nn.Sequential(
			gauss_layer, 
			irf_layer
		)
		self.gauss1D_layer = Gaussian1DLayer(gauss_len=num_bins, out_dim=-3)


	def forward(self, inputs):

		# Feature extraction
		msfeat = self.msfeat(inputs) 
		c1 = self.C1(msfeat)

		# Feature integration 
		b_out = self.dfu_block_group(c1)

		# depth reconstruction 
		rec = self.C_rec(b_out)

		# hist reconstruction
		denoise_out = self.hist_rec_layer(rec)

		return denoise_out, rec


class LITPlainDeepBoosting2DDepth2Depth2Hist01Inputs(LITBaseSPADModel):
	def __init__(self, 
		irf,
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9,
		in_channels=1,
		outchannel_MS=4,
		n_ddfn_blocks=12,
		num_bins=1024
		):
		
		deep_boosting_model = PlainDeepBoosting2DDepth2Depth2Hist(
			irf=irf,
			in_channels=in_channels, 
			outchannel_MS=outchannel_MS, 
			n_ddfn_blocks=n_ddfn_blocks, 
			num_bins=num_bins)

		super(LITPlainDeepBoosting2DDepth2Depth2Hist01Inputs, self).__init__(backbone_net=deep_boosting_model,
												init_lr = init_lr,
												p_tv = p_tv, 
												lr_decay_gamma = lr_decay_gamma)
		
		# Overwrite example input array
		self.example_input_array = torch.randn([1, 1, 32, 32])
	
	def get_input_data(self, sample):
		return sample["est_bins_argmax"]

if __name__=='__main__':
	import matplotlib.pyplot as plt
	import numpy as np
	from model_utils import count_parameters
	# Set random input
	batch_size = 2
	(nr, nc, nt) = (64, 64, 1024) 
	outchannel_MS = 4
	inputs3D = torch.randn((batch_size, 1, nt, nr, nc))
	inputs2D = torch.randn((batch_size, 1, nr, nc))

	## Generate a sample IRF
	from research_utils.signalproc_ops import gaussian_pulse
	irf_len = 40
	irf_mu = irf_len // 2
	irf = gaussian_pulse(time_domain=np.arange(0,irf_len), mu=irf_mu, width=3)

	# Set compression params
	inputs = inputs2D
	model = PlainDeepBoosting2DDepth2Depth2Hist(irf=irf, num_bins=nt)
	print("{} Parameters: {}".format(model.__class__.__name__, count_parameters(model)))
	outputs = model(inputs)
	print("		inputs shape: {}".format(inputs.shape))
	print("		outputs1 shape: {}".format(outputs[0].shape))
	print("		outputs2 shape: {}".format(outputs[1].shape))

	# Set compression params
	inputs = inputs2D
	model = LITPlainDeepBoosting2DDepth2Depth2Hist01Inputs(irf=irf, outchannel_MS=outchannel_MS, num_bins=nt)
	print("{} Parameters: {}".format(model.__class__.__name__, count_parameters(model)))
	outputs = model(inputs)
	print("		inputs shape: {}".format(inputs.shape))
	print("		outputs1 shape: {}".format(outputs[0].shape))
	print("		outputs2 shape: {}".format(outputs[1].shape))

