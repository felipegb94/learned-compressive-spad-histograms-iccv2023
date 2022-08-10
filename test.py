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
from model_utils import count_parameters, load_model_from_ckpt
from train import setup_tb_logger

# A logger for this file (not for the pytorch logger)
logger = logging.getLogger(__name__)

@hydra.main(config_path="./conf", config_name="test")
def test(cfg):

	if('middlebury' in cfg.params.test_datalist_fpath):
		assert(cfg.params.batch_size==1), 'For middlebury batch size should be 1 since not all images are the same size so the cant be run as batch'

	pl.seed_everything(1234)

	torch.backends.cudnn.benchmark = False

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
		output_size=None, disable_rand_crop=True)
	test_loader = DataLoader(test_data, batch_size=cfg.params.batch_size, 
		shuffle=False, num_workers=cfg.params.num_workers, pin_memory=cfg.params.cuda)
	logger.info("Load test data complete - {} test samples!".format(len(test_data)))
	logger.info("+++++++++++++++++++++++++++++++++++++++++++")

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

	logger.info("cuda flags BEFORE Trainer object:")
	logger.info("    torch.backends.cudnn.benchmark: {}".format(torch.backends.cudnn.benchmark))
	logger.info("    torch.backends.cudnn.enabled: {}".format(torch.backends.cudnn.enabled))
	logger.info("    torch.backends.cudnn.deterministic: {}".format(torch.backends.cudnn.deterministic))

	if(cfg.params.cuda):
		trainer = pl.Trainer(accelerator="gpu", devices=1, logger=tb_logger, callbacks=callbacks, benchmark=torch.backends.cudnn.benchmark) # 
	else:
		trainer = pl.Trainer(logger=tb_logger, callbacks=callbacks) # 

	logger.info("Number of Model Params: {}".format(count_parameters(model)))

	trainer.test(model, dataloaders=test_loader)

	# # Testing validation loop
	# trainer.validate(model, dataloaders=test_loader)



if __name__=='__main__':
	test()

