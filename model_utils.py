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


def init_model_from_id(cfg):
	if(cfg.model.model_id == 'DDFN_C64B10_NL'):
		lit_model = LITDeepBoosting(
						init_lr = cfg.params.lri
						, lr_decay_gamma = cfg.params.lr_decay_gamma
						, p_tv = cfg.params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						)
	elif(cfg.model.model_id == 'DDFN_C64B10'):
		lit_model = LITPlainDeepBoosting(
						init_lr = cfg.params.lri
						, lr_decay_gamma = cfg.params.lr_decay_gamma
						, p_tv = cfg.params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						)
	elif(cfg.model.model_id == 'DDFN_C64B10_NL_Depth2Depth'):
		lit_model = LITDeepBoostingDepth2Depth(
						init_lr = cfg.params.lri
						, lr_decay_gamma = cfg.params.lr_decay_gamma
						, p_tv = cfg.params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						)
	elif(cfg.model.model_id == 'DDFN_C64B10_NL_Compressive'):
		lit_model = LITDeepBoostingCompressive(
						init_lr = cfg.params.lri
						, lr_decay_gamma = cfg.params.lr_decay_gamma
						, p_tv = cfg.params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						, k = cfg.model.model_params.k
						)
	else:
		assert(False), "Incorrect model_id"

	return lit_model