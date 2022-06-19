#### Standard Library Imports

#### Library imports
import torch
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_ddfn_64_B10_CGNL_ori import LITPlainDeepBoosting
from csph_layers import CSPHEncodingLayer, CSPHDecodingLayer



class LITPlainDeepBoostingCSPH(LITPlainDeepBoosting):
	def __init__(self 
		, init_lr = 1e-4
		, p_tv = 1e-5 
		, lr_decay_gamma = 0.9
		, in_channels=1
		, k=16
		, spatial_down_factor=1
		, num_bins=1024
		, nt_blocks=1
		, tblock_init = 'TruncFourier'
		, optimize_tdim_codes = False
		, h_irf = None
		, encoding_type = 'separable'
		):
		# Init parent class
		super(LITPlainDeepBoostingCSPH, self).__init__(init_lr=init_lr,p_tv=p_tv,lr_decay_gamma=lr_decay_gamma,in_channels=in_channels)
		self.csph_encoding_layer = CSPHEncodingLayer(
					k=k, num_bins=num_bins, 
					tblock_init=tblock_init, h_irf=h_irf, optimize_tdim_codes=optimize_tdim_codes,
					nt_blocks=nt_blocks, spatial_down_factor=spatial_down_factor, 
					encoding_type=encoding_type
					)
					
		self.csph_decoding_layer = CSPHDecodingLayer(
					k=k, num_bins=num_bins, 
					nt_blocks=nt_blocks, up_factor_xydim=spatial_down_factor 
					)


	def forward(self, x):
		B = self.csph_encoding_layer(x)
		est_hist_img = self.csph_decoding_layer(B)
		# use forward for inference/predictions
		out = self.backbone_net(est_hist_img)
		return out
