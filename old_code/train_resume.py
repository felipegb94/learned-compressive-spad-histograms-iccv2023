#### Standard Library Imports
import logging
import os

#### Library imports
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_utils import count_parameters, load_model_from_ckpt
from train_test_utils import config_train_val_dataloaders, setup_train_callbacks, setup_tb_logger

# A logger for this file (not for the pytorch logger)
logger = logging.getLogger(__name__)

@hydra.main(config_path="./conf", config_name="train_resume")
def train_resume(cfg):

	pl.seed_everything(cfg.random_seed)
	logger.info("Running {} experiment mode".format(cfg.experiment))
	logger.info("Fixed random seed {}".format(cfg.random_seed))

	logger.info("\n" + OmegaConf.to_yaml(cfg))
	logger.info("Number of assigned GPUs: {}".format(cfg.train_params.gpu_num))
	logger.info("Number of available GPUs: {} {}".format(torch.cuda.device_count(), torch.cuda.get_device_name(torch.cuda.current_device())))


	if(torch.cuda.is_available() and cfg.train_params.cuda): device = torch.device("cuda:0")
	else: 
		device = torch.device("cpu")
		cfg.train_params.cuda = False

	## Config dataloaders. Use a function that can be re-used in train_resume.py
	(train_data, train_loader, val_data, val_loader) = config_train_val_dataloaders(cfg, logger)

	tb_logger = setup_tb_logger()

	(callbacks, lr_monitor_callback, ckpt_callback, resume_ckpt_callback) = setup_train_callbacks()

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

	lit_model, ckpt_fpath = load_model_from_ckpt(cfg.model_name, ckpt_id, logger)

	if(cfg.train_params.cuda):
		trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=cfg.train_params.epoch, 
			logger=tb_logger, callbacks=callbacks, 
			log_every_n_steps=10, val_check_interval=0.5) # 
	else:
		trainer = pl.Trainer(max_epochs=cfg.train_params.epoch, 
			logger=tb_logger, callbacks=callbacks, 
			log_every_n_steps=5, val_check_interval=1.0) # 

	logger.info("Number of Model Params: {}".format(count_parameters(lit_model)))

	trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_fpath)





if __name__=='__main__':
	train_resume()

