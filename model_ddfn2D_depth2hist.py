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


class PlainDeepBoosting2DDepth2Hist(nn.Module):
	def __init__(self, in_channels=1, outchannel_MS=2, n_ddfn_blocks=10, num_bins=1024):
		super(PlainDeepBoosting2DDepth2Hist, self).__init__()
		# feature extraction
		self.msfeat = MsFeat2D(in_channels, outchannel_MS=outchannel_MS)
		self.num_hist_bins = num_bins
		
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
			nn.Conv2d(self.dfu_block_group.module_out_channels, self.num_hist_bins, kernel_size=1, stride=1, bias=True)
			# , nn.ReLU(inplace=True)
			# , nn.Sigmoid()
		)
		# init.kaiming_normal_(self.C_rec[0].weight, 0, 'fan_in', 'relu'); 
		init.normal_(self.C_rec[0].weight, mean=0.0, std=0.001)
		init.constant_(self.C_rec[0].bias, 0.0)

		self.smax = torch.nn.Softmax2d()
		softargmax_weights = torch.arange(0, self.num_hist_bins).unsqueeze(1).unsqueeze(1) / self.num_hist_bins
		self.register_buffer('softargmax_weights', softargmax_weights)

	def forward(self, inputs):

		# Feature extraction
		msfeat = self.msfeat(inputs) 
		c1 = self.C1(msfeat)

		# Feature integration 
		b_out = self.dfu_block_group(c1)

		# depth reconstruction 
		hist_rec = self.C_rec(b_out)
		# hist reconstruction
		denoise_out = hist_rec

		# softargmax reconstruction
		weighted_smax = self.softargmax_weights * self.smax(denoise_out)
		softargmax_rec = weighted_smax.sum(1).unsqueeze(1)


		return denoise_out, softargmax_rec


class LITPlainDeepBoosting2DDepth2Hist01Inputs(LITBaseSPADModel):
	def __init__(self, 
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9,
		in_channels=1,
		outchannel_MS=4,
		n_ddfn_blocks=12,
		num_bins=1024
		):
		
		deep_boosting_model = PlainDeepBoosting2DDepth2Hist(
			in_channels=in_channels, 
			outchannel_MS=outchannel_MS, 
			n_ddfn_blocks=n_ddfn_blocks, 
			num_bins=num_bins)

		super(LITPlainDeepBoosting2DDepth2Hist01Inputs, self).__init__(backbone_net=deep_boosting_model,
												init_lr = init_lr,
												p_tv = p_tv, 
												lr_decay_gamma = lr_decay_gamma)
		
		# Overwrite example input array
		self.example_input_array = torch.randn([1, 1, 32, 32])
	
	def get_input_data(self, sample):
		return sample["est_bins_argmax"]


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
	model = LITPlainDeepBoosting2DDepth2Hist01Inputs(outchannel_MS=outchannel_MS, num_bins=nt)
	print("{} Parameters: {}".format(model.__class__.__name__, count_parameters(model)))
	outputs = model(inputs)
	print("		inputs shape: {}".format(inputs.shape))
	print("		outputs1 shape: {}".format(outputs[0].shape))
	print("		outputs2 shape: {}".format(outputs[1].shape))
