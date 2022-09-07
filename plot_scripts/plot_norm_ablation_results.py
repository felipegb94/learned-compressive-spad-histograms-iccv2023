'''
    This script plots the test set results for models that were trained with different normalization strategies
    The test commands for these models appear under `scripts_test/test_csph3d_norm_strategies.sh`
'''
#### Standard Library Imports
import os
import sys

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import hydra
from omegaconf import OmegaConf
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from tof_utils import *
from research_utils import plot_utils, np_utils, io_ops
from spad_dataset import SpadDataset
from analyze_test_results_utils import init_model_metrics_dict, process_middlebury_test_results, get_hydra_io_dirpaths

if __name__=='__main__':

	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_norm_ablation_results')

	## output dirpaths
	experiment_name = 'middlebury/norm_ablation'
	out_dirpath = os.path.join(io_dirpaths.results_weekly_dirpath, experiment_name)
	os.makedirs(out_dirpath, exist_ok=True)

	## Scene ID and Params
	test_set_id = 'test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
	scene_ids = ['spad_Art', 'spad_Reindeer', 'spad_Books', 'spad_Moebius', 'spad_Bowling1', 'spad_Dolls', 'spad_Laundry', 'spad_Plastic']
	sbr_params = ['2_2','2_10','2_50','5_2','5_10','5_50','10_2','10_10','10_50']
	sbr_params_high_flux = ['10_200', '10_500', '10_1000', '50_50', '50_200', '50_500', '50_1000'] ## more than 100 photons per pixel on average

	## Create spad dataloader object to get ground truth data for each scene
	spad_dataset = SpadDataset(datalist_fpath=os.path.join(io_dirpaths.datalists_dirpath, test_set_id+'.txt'), disable_rand_crop=True)

	## Model IDs of the models we want to plot
	model_names = []
	model_names.append('DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0')
	model_names.append('DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-False_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0')
	model_names.append('DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0')
	model_names.append('DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0')
	model_names.append('DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-0.0')
	model_names.append('DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-L2/loss-kldiv_tv-0.0')
	model_names.append('DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-Linf/loss-kldiv_tv-0.0')
	# model_names.append('DDFN_C64B10_CSPH3Dv1/k128_down4_Mt1_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0')
	# model_names.append('DDFN_C64B10_CSPH3Dv2/k128_down4_Mt1_Rand-optCt=True-optC=True_full_norm-L2/loss-kldiv_tv-0.0')
	# model_names.append('DDFN_C64B10_CSPH3Dv2/k128_down4_Mt1_Rand-optCt=True-optC=True_full_norm-Linf/loss-kldiv_tv-0.0')

	## Get pretrained models dirpaths
	pretrained_models_all = io_ops.load_json('pretrained_models_rel_dirpaths.json')
	model_dirpaths = []
	for model_name in model_names:
		model_dirpaths.append(pretrained_models_all[model_name]['rel_dirpath'])

	## init model metrics dict
	model_metrics_all = init_model_metrics_dict(model_names, model_dirpaths)

	## process results and append metrics
	model_metrics_all, rec_depths_all = process_middlebury_test_results(scene_ids, sbr_params, model_metrics_all, spad_dataset, out_dirpath=None, save_depth_images=False, return_rec_depths=False)


	# ## For each scene id and sbr params
	# for i in range(len(scene_ids)):
	# 	for j in range(len(sbr_params)):
	# 		curr_scene_id = scene_ids[i] 
	# 		curr_sbr_params = sbr_params[j] 

	# 		scene_fname = '{}_{}'.format(curr_scene_id, curr_sbr_params)
	# 		print("Processing: {}".format(scene_fname))






