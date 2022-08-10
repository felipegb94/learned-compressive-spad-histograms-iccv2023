#### Standard Library Imports
import logging

#### Library imports
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from spad_dataset import SpadDataset
from model_utils import init_model_from_id
from train_test_utils import config_train_val_dataloaders, setup_tb_logger, setup_train_callbacks


# A logger for this file (not for the pytorch logger)
logger = logging.getLogger(__name__)

@hydra.main(config_path="./conf", config_name="train")
def train(cfg):
	# if('debug' in cfg.experiment):
	# 	pl.seed_everything(cfg.random_seed)
	# 	logger.info("Running debug experiment mode. Fixed Random Seed")
	# else:
	# 	logger.info("Running {} experiment mode".format(cfg.experiment))

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

	logger.info("Initializing {} model".format(cfg.model.model_name))
	lit_model = init_model_from_id(cfg, irf=train_data.psf)
	
	if(cfg.train_params.overfit_batches):
		# trainer = pl.Trainer(fast_dev_run=True, logger=tb_logger, callbacks=[lr_monitor_callback]) # 
		if(cfg.train_params.cuda):
			trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=cfg.train_params.epoch, 
				logger=tb_logger, callbacks=[lr_monitor_callback], 
				log_every_n_steps=2, val_check_interval=1.0, overfit_batches=0.03,track_grad_norm=2) # 
		else:
			trainer = pl.Trainer(max_epochs=cfg.train_params.epoch, 
				logger=tb_logger, callbacks=[lr_monitor_callback], 
				log_every_n_steps=2, val_check_interval=1.0, overfit_batches=0.03,track_grad_norm=2) # 
	else:
		if(cfg.train_params.cuda):
			# trainer = pl.Trainer(accelerator="gpu", devices=1, 
			# 	limit_train_batches=30, limit_val_batches=10, max_epochs=3, 
			# 	logger=tb_logger, callbacks=callbacks,
			# 	log_every_n_steps=1, val_check_interval=0.5
			#  	) # Runs single batch
			trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=cfg.train_params.epoch, 
				logger=tb_logger, callbacks=callbacks, 
				log_every_n_steps=10, val_check_interval=0.5, benchmark=False
				# ,track_grad_norm=2
				) # 
		else:
			# trainer = pl.Trainer(
			# 	limit_train_batches=15, limit_val_batches=3, max_epochs=3, log_every_n_steps=1, 
			# 	logger=tb_logger, callbacks=callbacks) # Runs single batch
			# trainer = pl.Trainer(max_epochs=cfg.train_params.epoch, 
			# 	logger=tb_logger, callbacks=callbacks, 
			# 	log_every_n_steps=10, val_check_interval=1.0, track_grad_norm=2) # 
			trainer = pl.Trainer(max_epochs=cfg.train_params.epoch, 
				logger=tb_logger, callbacks=callbacks, 
				log_every_n_steps=10, val_check_interval=0.5
				# ,track_grad_norm=2 
				) # 

	trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__=='__main__':
	train()

