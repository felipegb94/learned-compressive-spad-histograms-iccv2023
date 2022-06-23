'''
	Base lightning model that contains the share loss functions and training parameters across all models
'''

#### Standard Library Imports
import os

#### Library imports
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_ddfn_64_B10_CGNL_ori import LITDeepBoosting, LITPlainDeepBoosting
from model_ddfn_64_B10_CGNL_ori_old import LITDeepBoostingOriginal
from model_ddfn_64_B10_CGNL_ori_depth2depth import LITDeepBoostingDepth2Depth, LITPlainDeepBoostingDepth2Depth
from model_ddfn_64_B10_CGNL_ori_CSPH import LITPlainDeepBoostingCSPH
from model_ddfn_64_B10_CGNL_ori_CSPH1D import LITPlainDeepBoostingCSPH1D
from model_ddfn_64_B10_CGNL_ori_CSPH1D2D import LITPlainDeepBoostingCSPH1D2D, LITPlainDeepBoostingCSPH1DGlobal2DLocal4xDown
from model_ddfn_64_B10_CGNL_ori_compressive import LITDeepBoostingCompressive, LITDeepBoostingCompressiveWithBias
from model_ddfn2D_depth2depth import LITPlainDeepBoosting2DDepth2Depth01Inputs, LITPlainDeepBoosting2DPhasor2Depth, LITPlainDeepBoosting2DPhasor2Depth,LITPlainDeepBoosting2DPhasor2Depth7Freqs,LITPlainDeepBoosting2DPhasor2Depth1Freq
from model_ddfn2D_depth2depth2hist import LITPlainDeepBoosting2DDepth2Depth2Hist01Inputs
from model_unet2D_csph import LITUnet2DCSPH1D, LITUnet2DCSPH1D2Phasor, LITUnet2DCSPH1DLinearOut, LITUnet2DCSPH1D2FFTHist


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_model_from_id(cfg, irf=None):
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
	elif(cfg.model.model_id == 'DDFN_C64B10_Depth2Depth'):
		lit_model = LITPlainDeepBoostingDepth2Depth(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						)
	elif(cfg.model.model_id == 'DDFN_C64B10_CSPH'):
		lit_model = LITPlainDeepBoostingCSPH(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						, k = cfg.model.model_params.k
						, spatial_down_factor = cfg.model.model_params.spatial_down_factor
						, nt_blocks = cfg.model.model_params.nt_blocks
						, tblock_init = cfg.model.model_params.tblock_init
						, optimize_tdim_codes = cfg.model.model_params.optimize_tdim_codes
						, encoding_type = cfg.model.model_params.encoding_type
						, num_bins = cfg.dataset.nt
						, h_irf = irf
						)
	elif(cfg.model.model_id == 'DDFN_C64B10_CSPH1D'):
		lit_model = LITPlainDeepBoostingCSPH1D(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						, init = cfg.model.model_params.init
						, k = cfg.model.model_params.k
						, num_bins = cfg.dataset.nt
						, h_irf = irf
						)
	elif(cfg.model.model_id == 'DDFN_C64B10_CSPH1D2D'):
		lit_model = LITPlainDeepBoostingCSPH1D2D(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						, init = cfg.model.model_params.init
						, k = cfg.model.model_params.k
						, down_factor = cfg.model.model_params.down_factor
						, num_bins = cfg.dataset.nt
						, h_irf = irf
						)
	elif(cfg.model.model_id == 'DDFN_C64B10_CSPH1DGlobal2DLocal4xDown'):
		lit_model = LITPlainDeepBoostingCSPH1DGlobal2DLocal4xDown(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						, init = cfg.model.model_params.init
						, k = cfg.model.model_params.k
						, num_bins = cfg.dataset.nt
						, h_irf = irf
						)
	elif(cfg.model.model_id == 'DDFN_C64B10_NL_Compressive'):
		lit_model = LITDeepBoostingCompressive(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						, k = cfg.model.model_params.k
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
	elif(cfg.model.model_id == 'DDFN2D_Phasor2Depth'):
		lit_model = LITPlainDeepBoosting2DPhasor2Depth(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, outchannel_MS = cfg.model.model_params.outchannel_MS
						, n_ddfn_blocks = cfg.model.model_params.n_ddfn_blocks
						, num_bins = cfg.dataset.nt
						)
	elif(cfg.model.model_id == 'DDFN2D_Phasor2Depth7Freqs'):
		lit_model = LITPlainDeepBoosting2DPhasor2Depth7Freqs(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, outchannel_MS = cfg.model.model_params.outchannel_MS
						, n_ddfn_blocks = cfg.model.model_params.n_ddfn_blocks
						, num_bins = cfg.dataset.nt
						)
	elif(cfg.model.model_id == 'DDFN2D_Phasor2Depth1Freq'):
		lit_model = LITPlainDeepBoosting2DPhasor2Depth1Freq(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, outchannel_MS = cfg.model.model_params.outchannel_MS
						, n_ddfn_blocks = cfg.model.model_params.n_ddfn_blocks
						, num_bins = cfg.dataset.nt
						)
	elif(cfg.model.model_id == 'DDFN2D_Depth2Depth2Hist_01Inputs'):
		lit_model = LITPlainDeepBoosting2DDepth2Depth2Hist01Inputs(
						irf = irf
						, init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, in_channels = cfg.model.model_params.in_channels
						, outchannel_MS = cfg.model.model_params.outchannel_MS
						, n_ddfn_blocks = cfg.model.model_params.n_ddfn_blocks
						, num_bins = cfg.dataset.nt
						)
	elif(cfg.model.model_id == 'Unet2D_CSPH1D'):
		lit_model = LITUnet2DCSPH1D(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, data_loss_id = cfg.model.data_loss_id
						, init = cfg.model.model_params.init
						, optimize_csph = cfg.model.model_params.optimize_csph
						, k = cfg.model.model_params.k
						, num_bins = cfg.dataset.nt
						, h_irf = irf
						)
	elif(cfg.model.model_id == 'Unet2D_CSPH1D2Phasor'):
		lit_model = LITUnet2DCSPH1D2Phasor(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, data_loss_id = cfg.model.data_loss_id
						, init = cfg.model.model_params.init
						, optimize_csph = cfg.model.model_params.optimize_csph
						, k = cfg.model.model_params.k
						, num_bins = cfg.dataset.nt
						, h_irf = irf
						)
	elif(cfg.model.model_id == 'Unet2D_CSPH1D2FFTHist'):
		lit_model = LITUnet2DCSPH1D2FFTHist(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, data_loss_id = cfg.model.data_loss_id
						, init = cfg.model.model_params.init
						, optimize_csph = cfg.model.model_params.optimize_csph
						, k = cfg.model.model_params.k
						, num_bins = cfg.dataset.nt
						, h_irf = irf
						)
	elif(cfg.model.model_id == 'Unet2D_CSPH1DLinearOut'):
		lit_model = LITUnet2DCSPH1DLinearOut(
						init_lr = cfg.train_params.lri
						, lr_decay_gamma = cfg.train_params.lr_decay_gamma
						, p_tv = cfg.train_params.p_tv
						, data_loss_id = cfg.model.data_loss_id
						, init = cfg.model.model_params.init
						, optimize_csph = cfg.model.model_params.optimize_csph
						, k = cfg.model.model_params.k
						, num_bins = cfg.dataset.nt
						, h_irf = irf
						)
	else:
		assert(False), "Incorrect model_id"

	return lit_model

def load_model_from_ckpt(model_name, ckpt_id, logger=None, model_dirpath=None):
	if(model_dirpath is None):
		model_dirpath = './'
	ckpt_fpath = os.path.join(model_dirpath, 'checkpoints/', ckpt_id)
	print(os.listdir(os.path.join(model_dirpath, "checkpoints/")))
	print(os.getcwd())
	assert(os.path.exists(ckpt_fpath)), "Input checkpoint does not exist ({})".format(ckpt_id)
	if(logger is None):
		print("Loading {} model".format(model_name))
	else:
		logger.info("Loading {} model".format(model_name))
	if(model_name == 'DDFN_C64B10_NL_Depth2Depth'):
		model = LITDeepBoostingDepth2Depth.load_from_checkpoint(ckpt_fpath)
	elif(model_name == 'DDFN_C64B10_Depth2Depth'):
		model = LITPlainDeepBoostingDepth2Depth.load_from_checkpoint(ckpt_fpath)
	elif('DDFN_C64B10_CSPH/' in model_name):
		model = LITPlainDeepBoostingCSPH.load_from_checkpoint(ckpt_fpath)
	elif('DDFN_C64B10_CSPH1D/' in model_name):
		model = LITPlainDeepBoostingCSPH1D.load_from_checkpoint(ckpt_fpath)
	elif('DDFN_C64B10_CSPH1D2D/' in model_name):
		model = LITPlainDeepBoostingCSPH1D2D.load_from_checkpoint(ckpt_fpath)
	elif('DDFN_C64B10_CSPH1DGlobal2DLocal4xDown/' in model_name):
		model = LITPlainDeepBoostingCSPH1DGlobal2DLocal4xDown.load_from_checkpoint(ckpt_fpath)
	elif(model_name == 'DDFN_C64B10_NL_Compressive'):
		model = LITDeepBoostingCompressive.load_from_checkpoint(ckpt_fpath)
	elif(model_name == 'DDFN_C64B10_NL_CompressiveWithBias'):
		model = LITDeepBoostingCompressiveWithBias.load_from_checkpoint(ckpt_fpath)
	elif(model_name == 'DDFN_C64B10_NL_original'):
		model = LITDeepBoostingOriginal.load_from_checkpoint(ckpt_fpath)
	elif(model_name == 'DDFN_C64B10'):
		model = LITPlainDeepBoosting.load_from_checkpoint(ckpt_fpath)
	elif('DDFN2D_Depth2Depth_01Inputs' in model_name):
		model = LITPlainDeepBoosting2DDepth2Depth01Inputs.load_from_checkpoint(ckpt_fpath, strict=False)
	elif('DDFN2D_Phasor2Depth' in model_name):
		if('7Freqs' in model_name):
			model = LITPlainDeepBoosting2DPhasor2Depth7Freqs.load_from_checkpoint(ckpt_fpath, strict=False)
		if('1Freq' in model_name):
			model = LITPlainDeepBoosting2DPhasor2Depth1Freq.load_from_checkpoint(ckpt_fpath, strict=False)
		else:
			model = LITPlainDeepBoosting2DPhasor2Depth.load_from_checkpoint(ckpt_fpath, strict=False)
	elif(model_name == 'DDFN2D_Depth2Depth2Hist_01Inputs/B-12_MS-8'):
		model = LITPlainDeepBoosting2DDepth2Depth2Hist01Inputs.load_from_checkpoint(ckpt_fpath, strict=False)
	else:
		assert(False), "Invalid model_name: {}".format(model_name)
	
	return model, ckpt_fpath