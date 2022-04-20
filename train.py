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
from model_ddfn_64_B10_CGNL_ori import LITDeepBoosting


# A logger for this file (not for the pytorch logger)
logger = logging.getLogger(__name__)

@hydra.main(config_path="./conf", config_name="train")
def train(cfg):
	if('debug' in cfg.experiment):
		pl.seed_everything(1234)
		logger.info("Running debug experiment mode. Fixed Random Seed")
	else:
		logger.info("Running {} experiment mode".format(cfg.experiment))
	logger.info("\n" + OmegaConf.to_yaml(cfg))
	logger.info("Number of assigned GPUs: {}".format(cfg.params.gpu_num))
	logger.info("Number of available GPUs: {} {}".format(torch.cuda.device_count(), torch.cuda.get_device_name(torch.cuda.current_device())))

	if(torch.cuda.is_available() and cfg.params.cuda): device = torch.device("cuda:0")
	else: 
		device = torch.device("cpu")
		cfg.params.cuda = False

	logger.info("Loading training data...")
	logger.info("Train Datalist: {}".format(cfg.params.train_datalist_fpath))
	# data preprocessing
	train_data = SpadDataset(cfg.params.train_datalist_fpath, cfg.params.noise_idx, output_size=cfg.params.crop_patch_size)
	train_loader = DataLoader(train_data, batch_size=cfg.params.batch_size, 
							shuffle=True, num_workers=cfg.params.workers, 
							pin_memory=cfg.params.cuda)
	logger.info("Load training data complete - {} train samples!".format(len(train_data)))
	logger.info("Loading validation data...")
	logger.info("Val Datalist: {}".format(cfg.params.val_datalist_fpath))
	val_data = SpadDataset(cfg.params.val_datalist_fpath, cfg.params.noise_idx, output_size=cfg.params.crop_patch_size)
	val_loader = DataLoader(val_data, batch_size=cfg.params.batch_size, 
							shuffle=False, num_workers=cfg.params.workers, 
							pin_memory=cfg.params.cuda)
	logger.info("Load validation data complete - {} val samples!".format(len(val_data)))
	logger.info("+++++++++++++++++++++++++++++++++++++++++++")


	# Let hydra manage directory outputs. We over-write the save_dir to . so that we use the ones that hydra configures
	# tensorboard = pl.loggers.TensorBoardLogger(save_dir=".", name="", version="", log_graph=True, default_hp_metric=False)
	tb_logger = pl.loggers.TensorBoardLogger(save_dir=".", name="", version="", log_graph=True, default_hp_metric=False)

	# NOTE: using rmse/avg_val instead of rmse_avg_val, allows to group things in tensorboard 
	# To avoid creating new directories when using / in the monitor metric name, we need to set auto_insert_metric_name=False
	# and set the filename we want to show
	ckpt_callback = pl.callbacks.ModelCheckpoint(
		monitor='rmse/avg_val'	
		, filename='epoch={epoch:02d}-step={step:02d}-avgvalrmse={rmse/avg_val:.4f}'
		, auto_insert_metric_name=False
		, save_last=True
		, save_top_k=-1 # Of -1 it saves model at end of epoch
		# , every_n_epochs=1 # How often to check the value we are monitoring
		, mode='min'
	) 

	lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
	
	callbacks = [ ckpt_callback, lr_monitor_callback ] 

	lit_model = LITDeepBoosting(
					init_lr = cfg.params.lri,
					lr_decay_gamma = cfg.params.lr_decay_gamma,
					p_tv = cfg.params.p_tv
					)
	# 
	# trainer = pl.Trainer(fast_dev_run=True ) # Runs single batch
	if(cfg.params.cuda):
		# trainer = pl.Trainer(accelerator="gpu", devices=1, 
		# 	limit_train_batches=80, limit_val_batches=10, max_epochs=5, 
		# 	logger=tb_logger, callbacks=callbacks,
		# 	log_every_n_steps=1, val_check_interval=0.25
		#  	) # Runs single batch
		trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=cfg.params.epoch, 
			logger=tb_logger, callbacks=callbacks, 
			log_every_n_steps=10, val_check_interval=0.25) # 
	else:
		# trainer = pl.Trainer(
		# 	limit_train_batches=2, limit_val_batches=3, max_epochs=8, log_every_n_steps=1, 
		# 	logger=tb_logger, callbacks=callbacks) # Runs single batch
		trainer = pl.Trainer(max_epochs=cfg.params.epoch, 
			logger=tb_logger, callbacks=callbacks, 
			log_every_n_steps=10, val_check_interval=0.25) # 

	trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__=='__main__':
	train()

