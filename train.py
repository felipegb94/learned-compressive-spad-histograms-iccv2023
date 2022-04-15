#### Standard Library Imports


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


@hydra.main(config_path="./conf", config_name="train")
def train(cfg):
	print(OmegaConf.to_yaml(cfg))

	print("Number of assigned GPUs: {}".format(cfg.params.gpu_num))
	print("Number of available GPUs: {} {}".format(torch.cuda.device_count(), torch.cuda.get_device_name(torch.cuda.current_device())))

	if(torch.cuda.is_available() and cfg.params.cuda): device = torch.device("cuda:0")
	else: 
		device = torch.device("cpu")
		cfg.params.cuda = False

	print("Loading training data...")
	# data preprocessing
	train_data = SpadDataset(cfg.params.train_datalist_fpath, cfg.params.noise_idx, output_size=cfg.params.crop_patch_size)
	train_loader = DataLoader(train_data, batch_size=cfg.params.batch_size, 
							shuffle=True, num_workers=cfg.params.workers, 
							pin_memory=cfg.params.cuda)
	print("Load training data complete - {} train samples!".format(len(train_data)))
	print("Loading validation data...")
	val_data = SpadDataset(cfg.params.val_datalist_fpath, cfg.params.noise_idx, output_size=cfg.params.crop_patch_size)
	val_loader = DataLoader(val_data, batch_size=cfg.params.batch_size, 
							shuffle=False, num_workers=cfg.params.workers, 
							pin_memory=cfg.params.cuda)
	print("Load validation data complete - {} val samples!".format(len(val_data)))
	print("+++++++++++++++++++++++++++++++++++++++++++")

	lit_model = LITDeepBoosting(cfg)

	# 
	# trainer = pl.Trainer(fast_dev_run=True ) # Runs single batch
	trainer = pl.Trainer(limit_train_batches=5, limit_val_batches=5, max_epochs=5) # Runs single batch
	# if(cfg.params.cuda):
	# 	trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=cfg.params.epoch) # 
	# else:
	# 	trainer = pl.Trainer() # 

	trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__=='__main__':
	train()

