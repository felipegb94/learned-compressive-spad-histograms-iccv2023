# The train file for network
# Based on pytorch 1.0
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms
import os
import sys
from tqdm import tqdm

from tensorboardX import SummaryWriter
from datetime import datetime
import skimage.io
import scipy.io as scio

from util.SpadDataset import SpadDataset
from util.ParseArgs import parse_args
from util.SaveChkp import save_checkpoint
from pro.Train import train
from pro.ParamCounter import ParamCounter
from models import model_ddfn_64_B10_CGNL_ori

## For debugging
from IPython.core import debugger
breakpoint = debugger.set_trace
import matplotlib.pyplot as plt

dtype = torch.cuda.FloatTensor

if __name__=="__main__":
	cfg_fpath='./config.ini'
	model_dirpath = './output/logfile/DDFN_C64B10_NL_batch-4_lrdec-no_addtvloss-no_date_03_29-12_21'
	model_fname = 'epoch_150_750_END.pth'
	model_fpath = os.path.join(model_dirpath, model_fname)
	assert(os.path.exists(model_fpath)), "Input model fpath does not exist"

	# parse arguments
	opt = parse_args(cfg_fpath, is_train=False)


	# configure network
	print("Constructing Models...")
	model = model_ddfn_64_B10_CGNL_ori.DeepBoosting()
	model.cuda()

	# Load model checkpoint
	checkpoint_model = torch.load(model_fpath)
	checkpoint_model_state_dict = checkpoint_model['state_dict']
	model.load_state_dict(checkpoint_model_state_dict)
	n_iter = checkpoint_model["n_iter"]
	epoch = checkpoint_model["epoch"]
	print("Loaded model checkpoint!")

	print("+++++++++++++++++++++++++++++++++++++++++++")
	# load data
	print("Loading training data...")
	# data preprocessing
	train_data = SpadDataset(opt["train_file"], opt["noise_idx"], 32)
	train_loader = DataLoader(train_data, batch_size=opt["batch_size"], 
							shuffle=True, num_workers=opt["workers"], 
							pin_memory=True)
	print("Load training data complete!\nLoading validation data...")
	# val_data = SpadDataset(opt["val_file"], opt["noise_idx"], 32)
	# val_loader = DataLoader(val_data, batch_size=opt["batch_size"], 
	# 						shuffle=False, num_workers=opt["workers"], 
	# 						pin_memory=True)
	# print("Load validation data complete!")
	print("+++++++++++++++++++++++++++++++++++++++++++")

	with torch.no_grad():
		for sample in tqdm(train_loader):
			M_mea = sample["spad"].type(dtype)
			M_gt = sample["rates"].type(dtype)
			dep = sample["bins"].type(dtype)
			idx = sample["idx"]
			M_mea_re, dep_re = model(M_mea)
			
			# M_mea_re_lsmx = lsmx(M_mea_re).unsqueeze(1)
			# loss_kl = criterion_KL(M_mea_re_lsmx, M_gt).data.cpu().numpy()
			# loss_tv = criterion_TV(dep_re).data.cpu().numpy()
			# rmse = criterion_L2(dep_re, dep).data.cpu().numpy()
			# loss = loss_kl + params["p_tv"]*loss_tv
			for i in range(len(idx)):
				C = 3e8
				Tp = 100e-9			
				plt.clf()
				plt.subplot(1,2,1)
				plt.imshow(dep.data.cpu().numpy()[i, 0, :, :] * Tp * C / 2, vmin=0, vmax=5)
				plt.title("GT, idx: {}, iter: {}".format(idx[i], n_iter))
				plt.subplot(1,2,2)
				plt.imshow(dep_re.data.cpu().numpy()[i, 0, :, :] * Tp * C / 2, vmin=0, vmax=5)
				plt.title("Rec, idx: {}, iter: {}".format(idx[i], n_iter))
				plt.colorbar()
				plt.pause(0.1)
				breakpoint()
				# model_id = params['log_file'].split('/')[-1]
				# out_dirpath = '/home/felipe/research_projects/2022_compressive-sp3Dc/results/week_2022-03-28/overfitting_20-ex_validate/'+model_id
				# out_fname = "rec_idx-{}_itr-{}".format(idx[i], n_iter)
				# save_currfig(dirpath = out_dirpath, filename = out_fname, file_ext = 'png', use_imsave=False)
			del M_mea_re
			del dep_re
			torch.cuda.empty_cache()
			


