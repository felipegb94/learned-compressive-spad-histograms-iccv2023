#### Standard Library Imports
import logging
import os

#### Library imports
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from spad_dataset import Lindell2018LinoSpadDataset, SpadDataset
from model_utils import count_parameters, load_model_from_ckpt
from train import setup_tb_logger
from model_ddfn_64_B10_CGNL_ori_CSPH3D import *
from research_utils.io_ops import load_json, write_json

# A logger for this file (not for the pytorch logger)
logger = logging.getLogger(__name__)

@hydra.main(config_path="./conf", config_name="test")
def test(cfg):

	if('middlebury' in cfg.params.test_datalist_fpath):
		assert(cfg.params.batch_size==1), 'For middlebury batch size should be 1 since not all images are the same size so the cant be run as batch'

	pl.seed_everything(1234)

	logger.info("\n" + OmegaConf.to_yaml(cfg))
	logger.info("Number of assigned GPUs: {}".format(cfg.params.gpu_num))
	logger.info("Number of available GPUs: {} {}".format(torch.cuda.device_count(), torch.cuda.get_device_name(torch.cuda.current_device())))

	if(torch.cuda.is_available() and cfg.params.cuda): device = torch.device("cuda:0")
	else: 
		device = torch.device("cpu")
		cfg.params.cuda = False

	tb_logger = setup_tb_logger() 

	## Callbacks
	callbacks = [ ] 

	# uses in_dim=32, out_dim=10
	if(cfg.ckpt_id):
		logger.info("Loading {} ckpt".format(cfg.ckpt_id))
		if('.ckpt' in cfg.ckpt_id):
			ckpt_id = cfg.ckpt_id
		else:
			ckpt_id = cfg.ckpt_id + '.ckpt'
	else:
		logger.info("Loading last.ckpt because no ckpt was given")
		ckpt_id = 'last.ckpt'

	model, ckpt_fpath = load_model_from_ckpt(cfg.model_name, ckpt_id, logger=logger)

	encoding_kernel_dims = None
	if(isinstance(model, LITPlainDeepBoostingCSPH3D)):
		encoding_kernel_dims = model.csph3d_layer.encoding_kernel3d_dims
	elif(isinstance(model, LITPlainDeepBoostingCSPH3Dv2)):
		encoding_kernel_dims = model.csph3d_layer.encoding_kernel3d_dims
	elif(isinstance(model, LITPlainDeepBoostingCSPH3Dv1)):
		encoding_kernel_dims = model.csph3d_layer.encoding_kernel_dims

	## int8 quantization only impacts CSPH3D models
	if(cfg.emulate_int8_quantization):
		model.enable_quantization_emulation()

	logger.info("Loading test data...")
	logger.info("Test Datalist: {}".format(cfg.params.test_datalist_fpath))
	if('lindell2018_linospad_captured' in cfg.dataset.name):
		## If model is a csph3d model, input the encoding kernel dimensions
		test_data = Lindell2018LinoSpadDataset(cfg.params.test_datalist_fpath, dims=(cfg.dataset.nt,cfg.dataset.nr,cfg.dataset.nc), tres_ps=cfg.dataset.tres_ps, encoding_kernel_dims=encoding_kernel_dims)
	else:
		test_data = SpadDataset(cfg.params.test_datalist_fpath, cfg.params.noise_idx, output_size=None, disable_rand_crop=True)
	
	
	test_loader = DataLoader(test_data, batch_size=cfg.params.batch_size, 
		shuffle=False, num_workers=cfg.params.num_workers, pin_memory=cfg.params.cuda)
	logger.info("Load test data complete - {} test samples!".format(len(test_data)))
	logger.info("+++++++++++++++++++++++++++++++++++++++++++")

	logger.info("cuda flags BEFORE Trainer object:")
	logger.info("    torch.backends.cudnn.benchmark: {}".format(torch.backends.cudnn.benchmark))
	logger.info("    torch.backends.cudnn.enabled: {}".format(torch.backends.cudnn.enabled))
	logger.info("    torch.backends.cudnn.deterministic: {}".format(torch.backends.cudnn.deterministic))

	if(('RTX 2070' in torch.cuda.get_device_name(torch.cuda.current_device())) and ('k64_down4_Mt1' in cfg.model_name or ('k256_down4_Mt1' in cfg.model_name))):
		logger.info("WARNING k=256_down4_Mt1 runs very slow in RTX 2070 for some reason...")
		logger.info("WARNING k=64_down4_Mt1 runs very slow in RTX 2070 for some reason...")

	if(cfg.params.cuda):
		trainer = pl.Trainer(accelerator="gpu", devices=1, logger=tb_logger, callbacks=callbacks, benchmark=True) # 
	else:
		trainer = pl.Trainer(logger=tb_logger, callbacks=callbacks) # 

	logger.info("Number of Model Params: {}".format(count_parameters(model)))

	# ## get new IRf and overwrite model IRF
	# from layers_parametric1D import IRF1DLayer
	# h_irf = test_loader.dataset.psf 
	# irf_layer_new = IRF1DLayer(irf=h_irf, conv_dim=0) 
	# model.csph3d_layer.irf_layer = irf_layer_new

	trainer.test(model, dataloaders=test_loader)

	print("Results for:")
	print("     Model Name: {}".format(cfg.model_name))
	print("     Model Dirpath: {}".format(cfg.model_dirpath))
	print("     ckpt: {}".format(cfg.ckpt_id))

	## add model to pre-trained model list that we can query from other scripts
	pretrained_models_db_fname = 'pretrained_models_rel_dirpaths.json'
	pretrained_models_db_fpath = os.path.join(get_original_cwd(), pretrained_models_db_fname)
	if(pretrained_models_db_fpath): pretrained_models_db = load_json(pretrained_models_db_fpath)
	else: pretrained_models_db = {}
	## add entry (only if it doesnt exist)
	if(not (cfg.model_name in pretrained_models_db)):
		pretrained_models_db[cfg.model_name] = {}
	## over-write everything else about the entry
	pretrained_models_db[cfg.model_name]['rel_dirpath'] = cfg.model_dirpath
	pretrained_models_db[cfg.model_name]['ckpt_id'] = cfg.ckpt_id
	## over-write test set results if needed
	pretrained_models_db[cfg.model_name][cfg.dataset.name] = {}
	pretrained_models_db[cfg.model_name][cfg.dataset.name]['depths/test_mae'] = float("{:.5}".format(float(np.around(trainer.logged_metrics['depths/test_mae'].cpu().numpy(), decimals=5))))
	pretrained_models_db[cfg.model_name][cfg.dataset.name]['depths/test_mse'] = float("{:.5}".format(float(np.around(trainer.logged_metrics['depths/test_mse'].cpu().numpy(), decimals=5))))
	pretrained_models_db[cfg.model_name][cfg.dataset.name]['depths/test_rmse'] = float("{:.5}".format(float(np.around(trainer.logged_metrics['depths/test_rmse'].cpu().numpy(), decimals=5))))

	write_json(pretrained_models_db_fpath, pretrained_models_db)



if __name__=='__main__':
	test()

