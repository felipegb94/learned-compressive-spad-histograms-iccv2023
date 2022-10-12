'''
	This script plots the test set results for models that were trained with different settings introduced in CSPH3D model
	The goal of this visualization is to find a good CSPH3D design that provides a good trade-off between performance and model size.
'''
#### Standard Library Imports
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import io
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from tof_utils import *
from research_utils import plot_utils, np_utils, io_ops
from spad_dataset import SpadDataset
import analyze_test_results_utils
from analyze_test_results_utils import init_model_metrics_dict, process_middlebury_test_results, get_hydra_io_dirpaths


if __name__=='__main__':

	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_csph3d_test_set_metrics')

	## output dirpaths
	experiment_name = 'middlebury/test_set_metrics'
	# out_dirpath = os.path.join(io_dirpaths.results_weekly_dirpath, experiment_name)
	out_dirpath = os.path.join(io_dirpaths.results_figures_dirpath, experiment_name)
	os.makedirs(out_dirpath, exist_ok=True)

	## Scene ID and Params
	test_set_id = 'test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
	middlebury_test_set_info = analyze_test_results_utils.middlebury_test_set_info
	scene_ids = middlebury_test_set_info['scene_ids']
	sbr_params_low_flux = middlebury_test_set_info['sbr_params_low_flux']
	sbr_params_high_flux = middlebury_test_set_info['sbr_params_high_flux']
	sbr_params = sbr_params_low_flux + sbr_params_high_flux

	## Create spad dataloader object to get ground truth data for each scene
	spad_dataset = SpadDataset(datalist_fpath=os.path.join(io_dirpaths.datalists_dirpath, test_set_id+'.txt'), disable_rand_crop=True)
	(nr,nc,nt,tres_ps) = spad_dataset.get_spad_data_sample_params(0)

	## Set model type we want to plot
	compression_ratio = 64
	spatial_down_factor = 4
	encoding_type_all = ['full', 'separable', 'separable']
	tdim_init_all = ['Rand', 'Rand', 'HybridGrayFourier']
	num_tdim_blocks_all = [1, 1, 1]
	n_csph3d_models = len(encoding_type_all)

	## Model IDs of the models we want to plot
	model_names = []

	## Generate names for all csph3d models
	for i in range(n_csph3d_models):
		tdim_init = tdim_init_all[i]
		encoding_type = encoding_type_all[i]
		num_tdim_blocks = num_tdim_blocks_all[i]
		(block_nr, block_nc) = (spatial_down_factor, spatial_down_factor)
		assert((nt % num_tdim_blocks) == 0), "nt should be divisible by num_tdim_blocks"
		block_nt = int(nt / num_tdim_blocks)
		k = analyze_test_results_utils.csph3d_compression2k(compression_ratio, block_nr, block_nc, block_nt)
		print(k)
		model_name = 'DDFN_C64B10_CSPH3D/k{}_down{}_Mt{}_{}-optCt=True-optC=True_{}_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0'.format(k, spatial_down_factor, num_tdim_blocks, tdim_init, encoding_type)
		model_names.append(model_name)

	## Get pretrained models dirpaths
	model_dirpaths = analyze_test_results_utils.get_model_dirpaths(model_names)
