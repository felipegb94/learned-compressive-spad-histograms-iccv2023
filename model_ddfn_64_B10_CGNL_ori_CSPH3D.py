#### Standard Library Imports

#### Library imports
import torch
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_ddfn_64_B10_CGNL_ori import LITPlainDeepBoosting
from csph_layers import CSPH3DLayer, CSPH3DLayerv1, CSPH3DLayerv2

class LITPlainDeepBoostingCSPH3D(LITPlainDeepBoosting):
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
		, optimize_codes = True
		, h_irf = None
		, encoding_type = 'separable'
		, csph_out_norm = 'none'
		, account_irf = False
		, smooth_tdim_codes=False
		, apply_zncc_norm=False
		, zero_mean_tdim_codes=False
		):
		# Init parent class
		super(LITPlainDeepBoostingCSPH3D, self).__init__(init_lr=init_lr,p_tv=p_tv,lr_decay_gamma=lr_decay_gamma,in_channels=in_channels)
		self.csph3d_layer = CSPH3DLayer(k=k, num_bins=num_bins, 
					tblock_init=tblock_init, h_irf=h_irf, optimize_tdim_codes=optimize_tdim_codes, optimize_codes = optimize_codes, 
					nt_blocks=nt_blocks, spatial_down_factor=spatial_down_factor, 
					encoding_type=encoding_type,
					csph_out_norm=csph_out_norm,
					account_irf=account_irf
					, smooth_tdim_codes=smooth_tdim_codes
					, apply_zncc_norm=apply_zncc_norm
					, zero_mean_tdim_codes=zero_mean_tdim_codes
					)
					
	def forward(self, x):
		x_hat = self.csph3d_layer(x)
		# use forward for inference/predictions
		out = self.backbone_net(x_hat)
		return out
	
	def enable_quantization_emulation(self):
		super(LITPlainDeepBoostingCSPH3D, self).enable_quantization_emulation()
		self.csph3d_layer.enable_quantization_emulation()

	# def get_sample_spad_data_ids(self, sample):
	# 	'''
	# 		if quantization is enabled modify them
	# 	'''
	# 	spad_data_ids = super(LITPlainDeepBoostingCSPH3D, self).get_sample_spad_data_ids(sample)
	# 	if(self.csph3d_layer.emulate_int8_quantization):
	# 		for i in range(len(spad_data_ids)):
	# 			spad_data_ids[i] = 'quantized_{}'.format(spad_data_ids[i])
	# 	return spad_data_ids
class LITPlainDeepBoostingCSPH3Dv2(LITPlainDeepBoosting):
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
		, optimize_codes = True
		, h_irf = None
		, encoding_type = 'separable'
		, csph_out_norm = 'none'
		):
		# Init parent class
		super(LITPlainDeepBoostingCSPH3Dv2, self).__init__(init_lr=init_lr,p_tv=p_tv,lr_decay_gamma=lr_decay_gamma,in_channels=in_channels)
		self.csph3d_layer = CSPH3DLayerv2(k=k, num_bins=num_bins, 
					tblock_init=tblock_init, h_irf=h_irf, optimize_tdim_codes=optimize_tdim_codes, optimize_codes = optimize_codes, 
					nt_blocks=nt_blocks, spatial_down_factor=spatial_down_factor, 
					encoding_type=encoding_type,
					csph_out_norm=csph_out_norm
					)
					
	def forward(self, x):
		x_hat = self.csph3d_layer(x)
		# use forward for inference/predictions
		out = self.backbone_net(x_hat)
		return out

class LITPlainDeepBoostingCSPH3Dv1(LITPlainDeepBoosting):
	'''
		Same as LITPlainDeepBoostingCSPH3D but it uses an older CSPH3D layer that stores the coding matrix weight differently. But the overall performance should be the same
	'''
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
		, optimize_codes = True
		, h_irf = None
		, encoding_type = 'separable'
		):
		# Init parent class
		super(LITPlainDeepBoostingCSPH3Dv1, self).__init__(init_lr=init_lr,p_tv=p_tv,lr_decay_gamma=lr_decay_gamma,in_channels=in_channels)
		self.csph3d_layer = CSPH3DLayerv1(k=k, num_bins=num_bins, 
					tblock_init=tblock_init, h_irf=h_irf, optimize_tdim_codes=optimize_tdim_codes, optimize_codes = optimize_codes, 
					nt_blocks=nt_blocks, spatial_down_factor=spatial_down_factor, 
					encoding_type=encoding_type
					)
					
	def forward(self, x):
		x_hat = self.csph3d_layer(x)
		# use forward for inference/predictions
		out = self.backbone_net(x_hat)
		return out
