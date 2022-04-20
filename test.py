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

@hydra.main(config_path="./conf", config_name="test")
def test(cfg):

	if('middlebury' in cfg.params.test_datalist_fpath):
		assert(cfg.params.batch_size==1), 'For middlebury batch size should be 1 since not all images are the same size so the cant be run as batch'

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

	logger.info("Loading test data...")
	logger.info("Test Datalist: {}".format(cfg.params.test_datalist_fpath))
	test_data = SpadDataset(cfg.params.test_datalist_fpath, cfg.params.noise_idx, 
		output_size=None, disable_rand_crop=True, tres_ps=cfg.params.tres_ps)
	test_loader = DataLoader(test_data, batch_size=cfg.params.batch_size, 
		shuffle=False, num_workers=0, pin_memory=cfg.params.cuda)
	logger.info("Load test data complete - {} test samples!".format(len(test_data)))
	logger.info("+++++++++++++++++++++++++++++++++++++++++++")

	# Let hydra manage directory outputs. We over-write the save_dir to . so that we use the ones that hydra configures
	# tensorboard = pl.loggers.TensorBoardLogger(save_dir=".", name="", version="", log_graph=True, default_hp_metric=False)
	tb_logger = pl.loggers.TensorBoardLogger(save_dir=".", name="", version="", log_graph=True, default_hp_metric=False)
	
	# uses in_dim=32, out_dim=10
	model = LITDeepBoosting.load_from_checkpoint("checkpoints/epoch=03-step=976-avgvalrmse=0.0326.ckpt")

	if(cfg.params.cuda):
		trainer = pl.Trainer(accelerator="gpu", devices=1, logger=tb_logger) # 
	else:
		trainer = pl.Trainer(logger=tb_logger) # 

	trainer.test(model, dataloaders=test_loader)


if __name__=='__main__':
	test()

