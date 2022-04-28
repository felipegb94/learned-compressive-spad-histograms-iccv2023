'''
	Base lightning model that contains the share loss functions and training parameters across all models
'''

#### Standard Library Imports

#### Library imports
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_ddfn_64_B10_CGNL_ori import LITDeepBoosting, LITPlainDeepBoosting
from model_ddfn_64_B10_CGNL_ori_depth2depth import LITDeepBoostingDepth2Depth
from model_ddfn_64_B10_CGNL_ori_compressive import LITDeepBoostingCompressive
from model_ddfn2D_depth2depth import LITPlainDeepBoosting2DDepth2Depth, LITPlainDeepBoosting2DDepth2Depth01Inputs
from model_ddfn2D_depth2hist import LITPlainDeepBoosting2DDepth2Hist01Inputs


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_model_from_id(cfg):
	if(cfg.model.model_id == 'DDFN_C64B10_NL'):
		lit_model = LITDeepBoosting(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						)
	elif(cfg.model.model_id == 'DDFN_C64B10'):
		lit_model = LITPlainDeepBoosting(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						)
	elif(cfg.model.model_id == 'DDFN_C64B10_NL_Depth2Depth'):
		lit_model = LITDeepBoostingDepth2Depth(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						)
	elif(cfg.model.model_id == 'DDFN_C64B10_NL_Compressive'):
		lit_model = LITDeepBoostingCompressive(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						, k = cfg.model.model_params.k
						)
	elif(cfg.model.model_id == 'DDFN2D_Depth2Depth'):
		lit_model = LITPlainDeepBoosting2DDepth2Depth(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						, outchannel_MS = cfg.model.model_params.outchannel_MS
						, n_ddfn_blocks = cfg.model.model_params.n_ddfn_blocks
						, num_bins = cfg.dataset.nt
						)
	elif(cfg.model.model_id == 'DDFN2D_Depth2Depth_01Inputs'):
		lit_model = LITPlainDeepBoosting2DDepth2Depth01Inputs(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						, outchannel_MS = cfg.model.model_params.outchannel_MS
						, n_ddfn_blocks = cfg.model.model_params.n_ddfn_blocks
						, num_bins = cfg.dataset.nt
						)
	elif(cfg.model.model_id == 'DDFN2D_Depth2Hist_01Inputs'):
		lit_model = LITPlainDeepBoosting2DDepth2Hist01Inputs(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						, outchannel_MS = cfg.model.model_params.outchannel_MS
						, n_ddfn_blocks = cfg.model.model_params.n_ddfn_blocks
						, num_bins = cfg.dataset.nt
						)
	else:
		assert(False), "Incorrect model_id"

	return lit_model