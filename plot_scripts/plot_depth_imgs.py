'''
	This script creates lines plots of compression vs. different test set metrics
'''
#### Standard Library Imports
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from tof_utils import *
from research_utils import plot_utils
from spad_dataset import SpadDataset
import analyze_test_results_utils
from analyze_test_results_utils import init_model_metrics_dict, process_middlebury_test_results, get_hydra_io_dirpaths, compose_csph3d_model_names_list

if __name__=='__main__':

	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_depth_imgs')

	## output dirpaths
	experiment_name_base = 'middlebury/depth_imgs'
	out_dirpath = os.path.join(io_dirpaths.results_figures_dirpath, experiment_name_base)
	os.makedirs(out_dirpath, exist_ok=True)

	## Scene ids and signal and SBR parameters we want to plot for
	test_set_id = 'test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
	scene_ids = ['spad_Art']
	sbr_params = ['50_1000']

	## Check that input scene ids and sbr params are available in test set
	middlebury_test_set_info = analyze_test_results_utils.middlebury_test_set_info
	scene_ids_all = middlebury_test_set_info['scene_ids']
	sbr_params_low_flux_all = middlebury_test_set_info['sbr_params_low_flux']
	sbr_params_high_flux_all = middlebury_test_set_info['sbr_params_high_flux']
	sbr_params_all = sbr_params_low_flux_all + sbr_params_high_flux_all
	for scene_id in scene_ids: assert(scene_id in scene_ids_all), "{} not in test scene ids".format(scene_id)
	for sbr_param in sbr_params: assert(sbr_param in sbr_params_all), "{} not in test SBR params".format(sbr_param)

	## Create spad dataloader object to get ground truth data for each scene
	spad_dataset = SpadDataset(datalist_fpath=os.path.join(io_dirpaths.datalists_dirpath, test_set_id+'.txt'), disable_rand_crop=True)
	(nr,nc,nt,tres_ps) = spad_dataset.get_spad_data_sample_params(0)

	## Generate model_names for the models we want to plots depths for
	model_names = []
	no_compression_baseline = 'DDFN_C64B10/loss-kldiv_tv-1e-5'
	argmax_compression_baseline = 'DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-5'
	model_names.append(no_compression_baseline)
	model_names.append(argmax_compression_baseline)
	# Set parameters for CSPH3D models
	encoding_type_all = ['csph1d', 'csph1d', 'separable', 'separable', 'separable']
	tdim_init_all = ['HybridGrayFourier', 'TruncFourier', 'Rand', 'Rand', 'Rand']
	optCt_all = [False, False, True, True, True]
	optC_all = [False, False, True, True, True]
	spatial_down_factor_all = [1, 1, 4, 4, 4]
	num_tdim_blocks_all = [1, 1, 1, 4, 16]
	compression_ratio_all = [32, 64, 128]
	# compression_ratio_all = [128]
	# generate the csph3d model names
	(csph3d_model_names, csph3d_num_model_params) = compose_csph3d_model_names_list(compression_ratio_all
									, spatial_down_factor_all
									, num_tdim_blocks_all
									, tdim_init_all
									, optCt_all
									, optC_all
									, encoding_type_all
									, nt = nt
		)
	model_names += csph3d_model_names

	## Get pretrained models dirpaths
	model_dirpaths = analyze_test_results_utils.get_model_dirpaths(model_names)

	## For each scene and sbr param load depth image
	## We need to do one by one so that rec_depths contain the correct scene + SBR params
	for scene_id in scene_ids:
		for sbr_param in sbr_params:
			scene_fname = '{}_{}'.format(scene_id, sbr_param)
			spad_dataset_sample = spad_dataset.get_item_by_scene_name(scene_fname)
			# plot intensity
			intensity_img = spad_dataset_sample['intensity']
			plt.clf()
			plt.imshow(intensity_img, cmap='gray')
			plot_utils.remove_ticks()
			plot_utils.save_currfig(dirpath=os.path.join(out_dirpath, scene_fname), filename='intensity_img', file_ext='svg')
			plt.pause(0.05)
			# init dict 
			model_metrics_all = init_model_metrics_dict(model_names, model_dirpaths)
			# only process a single scene
			model_metrics_all, rec_depths_all = process_middlebury_test_results([scene_id], [sbr_param], model_metrics_all, spad_dataset, return_rec_depths=True)
			# get gt depth to calculate the colorbar range
			gt_depths = rec_depths_all['gt']['depths']
			min_depth = gt_depths.flatten().min() - 0.5*gt_depths.flatten().std() 
			max_depth = gt_depths.flatten().max() + 0.5*gt_depths.flatten().std()
			# for each model plot depths
			for model_id in rec_depths_all.keys():
				curr_model = rec_depths_all[model_id]
				model_depths = curr_model['depths']
				plt.clf()
				plt.imshow(model_depths, vmin=min_depth, vmax=max_depth)
				plot_utils.remove_ticks()
				out_fname = model_id.replace('/','_')
				plot_utils.save_currfig(dirpath=os.path.join(out_dirpath, scene_fname), filename=out_fname, file_ext='svg')
				plt.pause(0.05)
				# save with colorbar
				plt.colorbar()
				plot_utils.save_currfig(dirpath=os.path.join(out_dirpath, scene_fname + '/withcbar'), filename=out_fname, file_ext='svg')
				plt.pause(0.05)
				plt.title(model_id)



