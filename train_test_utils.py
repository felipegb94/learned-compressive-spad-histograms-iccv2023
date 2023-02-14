#### Standard Library Imports

#### Library imports
from torch.utils.data import DataLoader
import pytorch_lightning as pl

#### Local imports
from spad_dataset import SpadDataset

def config_train_val_dataloaders(cfg, logger):
	logger.info("Loading training data...")
	logger.info("Train Datalist: {}".format(cfg.params.train_datalist_fpath))
	# data preprocessing
	train_data = SpadDataset(cfg.params.train_datalist_fpath
							, cfg.train_params.noise_idx 
							, disable_rand_crop=cfg.train_params.disable_rand_crop
							, output_size=cfg.train_params.crop_patch_size
							, logger=logger
	)
	train_loader = DataLoader(train_data, 
							batch_size=cfg.train_params.batch_size, 
							shuffle=True, 
							num_workers=cfg.train_params.workers, 
							pin_memory=cfg.train_params.cuda)
	logger.info("Load training data complete - {} train samples!".format(len(train_data)))
	logger.info("Loading validation data...")
	logger.info("Val Datalist: {}".format(cfg.params.val_datalist_fpath))
	val_data = SpadDataset(cfg.params.val_datalist_fpath
							, cfg.train_params.noise_idx
							, disable_rand_crop=True
							, logger=logger
	)
	val_loader = DataLoader(val_data, 
							batch_size=cfg.train_params.val_batch_size, 
							shuffle=False, 
							num_workers=cfg.train_params.val_workers, 
							pin_memory=cfg.train_params.cuda)
	logger.info("Load validation data complete - {} val samples!".format(len(val_data)))
	logger.info("+++++++++++++++++++++++++++++++++++++++++++")
	return (train_data, train_loader, val_data, val_loader)

def setup_tb_logger():
	# Let hydra manage directory outputs. We over-write the save_dir to . so that we use the ones that hydra configures
	# tensorboard = pl.loggers.TensorBoardLogger(save_dir=".", name="", version="", log_graph=True, default_hp_metric=False)
	tb_logger = pl.loggers.TensorBoardLogger(save_dir=".", name="", version="", log_graph=True, default_hp_metric=False)
	return tb_logger

def setup_train_callbacks():
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
		# , save_on_train_epoch_end=True
	) 
	
	# This ckpt callback only saves the checkpoint at the end of each epoch and can be used to resume training
	# NOTE: We need this because pytorch-lightning is a bit weird when loading checkpoints in the middle of an epoch
	resume_ckpt_callback = pl.callbacks.ModelCheckpoint(
		filename='epoch={epoch:02d}-step={step:02d}-end-of-epoch'
		, auto_insert_metric_name=False
		, save_last=False
		, save_on_train_epoch_end=True
	)

	lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
	
	callbacks = [ ckpt_callback, lr_monitor_callback, resume_ckpt_callback ] 
	return (callbacks, lr_monitor_callback, ckpt_callback, resume_ckpt_callback)

