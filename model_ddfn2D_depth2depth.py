#### Standard Library Imports

#### Library imports
from random import gauss
import torch
import torch.nn as nn
import torch.nn.init as init
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_base_spad_lit import LITBaseSPADModel, LITL1LossBaseSpadModel, make_zeromean_normalized_bins
from model_ddfn import Block2DGroup, MsFeat2D
from layers_parametric1D import Gaussian1DLayer

class PlainDeepBoosting2DDepth2Depth(nn.Module):
	def __init__(self, in_channels=1, outchannel_MS=2, n_ddfn_blocks=10, num_bins=1024):
		super(PlainDeepBoosting2DDepth2Depth, self).__init__()
		# feature extraction
		self.msfeat = MsFeat2D(in_channels, outchannel_MS=outchannel_MS)
		
		# 1x1 convolution
		self.C1 = nn.Sequential(
			nn.Conv2d(self.msfeat.module_out_channels, self.msfeat.module_out_channels, kernel_size=1, stride=1, bias=True),
			nn.ReLU(inplace=True)
			)
		init.kaiming_normal_(self.C1[0].weight, 0, 'fan_in', 'relu') 
		init.constant_(self.C1[0].bias, 0.0)

		## ddfn blocks
		self.n_ddfn_blocks = n_ddfn_blocks
		self.dfu_block_group = Block2DGroup(in_channels=self.msfeat.module_out_channels, n_ddfn_blocks=self.n_ddfn_blocks)

		# Reconstruction kernel
		self.C_rec = nn.Sequential(
			nn.Conv2d(self.dfu_block_group.module_out_channels, 1, kernel_size=1, stride=1, bias=True),
			# nn.ReLU(inplace=True)
			nn.Sigmoid()
			# nn.Tanh()
		)
		# init.kaiming_normal_(self.C_rec[0].weight, 0, 'fan_in', 'relu') 
		# init.normal_(self.C_rec[0].weight, mean=0.0, std=1)
		init.xavier_uniform_(self.C_rec[0].weight)
		init.constant_(self.C_rec[0].bias, 0.0)

		self.gauss1D_layer = Gaussian1DLayer(gauss_len=num_bins, out_dim=-3)

	def forward(self, inputs):

		# Feature extraction
		msfeat = self.msfeat(inputs) 
		c1 = self.C1(msfeat)

		# Feature integration 
		b_out = self.dfu_block_group(c1)

		# depth reconstruction 
		rec = self.C_rec(b_out)
		# rec = (0.5*self.C_rec(b_out)) + 1 # If we use tanh make sure to make the rec between 0 and 1
		# rec = self.C_rec(torch.cat((inputs, b_out), 1))
		# breakpoint()

		# hist reconstruction
		denoise_out = self.gauss1D_layer(rec)

		return denoise_out, rec


class LITPlainDeepBoosting2DDepth2Depth(LITL1LossBaseSpadModel):
	def __init__(self, 
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9,
		in_channels=1,
		outchannel_MS=4,
		n_ddfn_blocks=12,
		num_bins=1024
		):
		
		deep_boosting_model = PlainDeepBoosting2DDepth2Depth(
			in_channels=in_channels, 
			outchannel_MS=outchannel_MS, 
			n_ddfn_blocks=n_ddfn_blocks, 
			num_bins=num_bins)

		super(LITPlainDeepBoosting2DDepth2Depth, self).__init__(backbone_net=deep_boosting_model,
												init_lr = init_lr,
												p_tv = p_tv, 
												lr_decay_gamma = lr_decay_gamma)
		
		# Overwrite example input array
		self.example_input_array = torch.randn([1, 1, 32, 32])
	
	def get_input_data(self, sample):
		return make_zeromean_normalized_bins(sample["est_bins_argmax"])
		# return sample["est_bins_argmax"]


if __name__=='__main__':
	import matplotlib.pyplot as plt
	from model_utils import count_parameters
	# Set random input
	batch_size = 2
	(nr, nc, nt) = (64, 64, 512) 
	outchannel_MS = 4
	inputs3D = torch.randn((batch_size, 1, nt, nr, nc))
	inputs2D = torch.randn((batch_size, 1, nr, nc))

	# Set compression params
	inputs = inputs2D
	model = PlainDeepBoosting2DDepth2Depth()
	print("{} Parameters: {}".format(model.__class__.__name__, count_parameters(model)))
	outputs = model(inputs)
	print("		inputs shape: {}".format(inputs.shape))
	print("		outputs1 shape: {}".format(outputs[0].shape))
	print("		outputs2 shape: {}".format(outputs[1].shape))

	# Set compression params
	inputs = inputs2D
	model = LITPlainDeepBoosting2DDepth2Depth(outchannel_MS=outchannel_MS, num_bins=nt)
	print("{} Parameters: {}".format(model.__class__.__name__, count_parameters(model)))
	outputs = model(inputs)
	print("		inputs shape: {}".format(inputs.shape))
	print("		outputs1 shape: {}".format(outputs[0].shape))
	print("		outputs2 shape: {}".format(outputs[1].shape))
