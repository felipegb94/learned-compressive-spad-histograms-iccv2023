#### Standard Library Imports
import logging
import os
import shutil

#### Library imports
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from research_utils import io_ops
from spad_dataset import SpadDataset
from model_utils import init_model_from_id, load_model_from_ckpt, get_latest_end_of_epoch_ckpt
from train_test_utils import config_train_val_dataloaders, setup_tb_logger, setup_train_callbacks


# A logger for this file (not for the pytorch logger)
logger = logging.getLogger(__name__)

@hydra.main(config_path="./conf", config_name="train")
def train(cfg):

	pl.seed_everything(cfg.random_seed)
	latest_end_of_epoch_ckpt_id = None

	## Try to resume training by moving the checkpoints folder from a previous run and finding the latest end of epoch file
	if(cfg.resume_train):
		## At this point the only files in curr_dirpath are: train.log and .hydra
		curr_dirpath = os.getcwd()
		model_dirpath = os.path.dirname(curr_dirpath) # parent directory 
		latest_run_dirpaths = io_ops.get_dirnames_in_dir(model_dirpath, str_in_dirname='run-latest', include_full_dirpath=True)
		assert(len(latest_run_dirpaths) <= 2), "There should at most 2 runs with the prefix run-latest in the model directory: {}".format(model_dirpath)
		latest_run_dirpaths.remove(curr_dirpath)
		if(len(latest_run_dirpaths) == 0):
			logger.info("No other run-latest directory found. Starting training from scratch")
		else:
			# Rename the previous run directory
			prev_run_dirpath = latest_run_dirpaths[0]
			new_prev_run_dirpath = prev_run_dirpath.replace('run-latest_', 'run-prev_')
			os.rename(prev_run_dirpath, new_prev_run_dirpath)
			prev_run_dirpath = new_prev_run_dirpath
			# Move all checkpoints from the previous run here
			prev_ckpt_dirpath = os.path.join(prev_run_dirpath, "checkpoints")
			if(os.path.exists(prev_ckpt_dirpath)):
				# move checkpoints
				shutil.move(prev_ckpt_dirpath, curr_dirpath)
				# Copy all logs, tensorboard and .yaml files
				io_ops.copy_all_files(prev_run_dirpath, curr_dirpath, ignore_fname_list=['.hydra'])
				logger.info("=======================================")
				logger.info("Copying Files and Attempting to Resume latest end of epoch ckpt")
				latest_end_of_epoch_ckpt_id = get_latest_end_of_epoch_ckpt(os.path.join(curr_dirpath, 'checkpoints'))
			else:
				logger.info("No checkpoint in run-latest directory found. Starting training from scratch")

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

	## Initializing trainer/optimizer object
	logger.info("Initializing Trainer")
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
				log_every_n_steps=10, val_check_interval=0.25, benchmark=True
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
				log_every_n_steps=10, val_check_interval=0.25
				# ,track_grad_norm=2 
				) # 

	## Initialze model from ckpt or from scratch
	logger.info("Initializing model: {}".format(cfg.model.model_name))
	if(latest_end_of_epoch_ckpt_id is None):
		lit_model = init_model_from_id(cfg, irf=train_data.psf)
		trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
	else:
		lit_model, ckpt_fpath = load_model_from_ckpt(cfg.model.model_name, latest_end_of_epoch_ckpt_id, logger)
		trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_fpath)

	## Wrap up training and change the folder name to reflect that the model finished
	logger.info("Finished training model: {}".format(cfg.model.model_name))
	complete_model_dirpath = curr_dirpath.replace('run-latest_', 'run-complete_')
	os.rename(curr_dirpath, complete_model_dirpath)


if __name__=='__main__':
	train()

