#### Standard Library Imports
import sys
import os
sys.path.append('../')
sys.path.append('./')

#### Library imports
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from train_test_utils import config_train_val_dataloaders
from spad_dataset import SpadDataset
from tof_utils import *


if __name__=='__main__':

	# data preprocessing
	datalist_fpath = './datalists/nyuv2_train_SimSPADDataset_nr-64_nc-64_nt-1024_tres-80ps_dark-1_psf-1.txt'
	dataset = SpadDataset(datalist_fpath
							, noise_idx=None 
							, disable_rand_crop=False
							, output_size=(64,64)
	)
	loader = DataLoader(dataset, 
							batch_size=1, 
							shuffle=True, 
							num_workers=4)

	n_samples = 1000
	(nr,nc,nt) = (64,64,1024)
	n_pixels=nr*nc
	tres = 80e-12
	max_depths = np.zeros((n_samples,))
	percent_depths_gt_than_4m = np.zeros((n_samples,))
	percent_depths_gt_than_6m = np.zeros((n_samples,))
	percent_depths_gt_than_8m = np.zeros((n_samples,))
	percent_depths_gt_than_9m = np.zeros((n_samples,))
	percent_depths_gt_than_10m = np.zeros((n_samples,))
	percent_depths_gt_than_12m = np.zeros((n_samples,))
	

	for idx, sample in enumerate(loader):
		if(idx == n_samples):
			break
		gt_bins = sample['bins']*nt 
		gt_depths = bin2depth(gt_bins, num_bins=nt, tau=tres*nt).numpy()
		max_depths[idx] = np.max(gt_depths)
		percent_depths_gt_than_4m[idx] = 100*np.sum(gt_depths>4.) / n_pixels
		percent_depths_gt_than_6m[idx] = 100*np.sum(gt_depths>6.) / n_pixels
		percent_depths_gt_than_8m[idx] = 100*np.sum(gt_depths>8.) / n_pixels
		percent_depths_gt_than_9m[idx] = 100*np.sum(gt_depths>9.) / n_pixels
		percent_depths_gt_than_10m[idx] = 100*np.sum(gt_depths>10.) / n_pixels
		percent_depths_gt_than_12m[idx] = 100*np.sum(gt_depths>12.) / n_pixels
		# print("Max Depth {}: {}m".format(idx, max_depths[idx]))
	
	print("Percent Depths >= 4m: {}".format(percent_depths_gt_than_4m.mean()))
	print("Percent Depths >= 6m: {}".format(percent_depths_gt_than_6m.mean()))
	print("Percent Depths >= 8m: {}".format(percent_depths_gt_than_8m.mean()))
	print("Percent Depths >= 9m: {}".format(percent_depths_gt_than_9m.mean()))
	print("Percent Depths >= 10m: {}".format(percent_depths_gt_than_10m.mean()))
	print("Percent Depths >= 12m: {}".format(percent_depths_gt_than_12m.mean()))








