'''
	This script plots the test set results for models that were trained with different settings introduced in CSPH3D model
	The goal of this visualization is to find a good CSPH3D design that provides a good trade-off between performance and model size.
'''
#### Standard Library Imports
from itertools import compress
import os
from humanize import metric

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
from csph_layers import compute_csph3d_expected_num_params
from research_utils import plot_utils, np_utils, io_ops
from spad_dataset import SpadDataset
import analyze_test_results_utils
from analyze_test_results_utils import init_model_metrics_dict, process_middlebury_test_results, get_hydra_io_dirpaths


def simplify_model_name(model_name):
	## only keep model params
	model_name_min = model_name.replace('DDFN_C64B10_CSPH3D', '')
	model_name_min = model_name_min.replace('/loss-kldiv_tv-0.0', '')
	model_name_min = model_name_min.replace('False', '0')
	model_name_min = model_name_min.replace('True', '1')
	model_name_min = model_name_min.replace('_smoothtdimC-0', '')
	model_name_min = model_name_min.replace('_irf-0', '')
	model_name_min = model_name_min.replace('down4','4x4')
	model_name_min = model_name_min.replace('down2','2x2')
	model_name_min = model_name_min.replace('down1','1x1')
	model_name_min = model_name_min.replace('_Mt1_','x1024_')
	model_name_min = model_name_min.replace('_Mt4_','x256_')
	model_name_min = model_name_min.replace('_Mt16_','x64_')
	return model_name_min

# def extract_model_type(model_name):
# 	model_name_min = simplify_model_name(model_name)
# 	model_type = '_'.join(model_name_min.split('_')[1:])
# 	model_type = model_type.split('_norm')[0]
# 	return model_type

if __name__=='__main__':

	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_compression_vs_test_metrics')

	compression_ratio_all = [32, 64, 128]
	# compression_ratio_all = [128]

	no_compression_baseline = 'DDFN_C64B10/loss-kldiv_tv-1e-5'
	argmax_compression_baseline = 'DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-5'

	model_names = []
	num_model_params = []
	model_names.append(no_compression_baseline)
	model_names.append(argmax_compression_baseline)
	num_model_params = [0] * len(model_names)
	model_metrics_df = pd.DataFrame()

	## output dirpaths
	experiment_name = 'middlebury/compression_vs_test_metrics'
	out_dirpath = os.path.join(io_dirpaths.results_figures_dirpath, experiment_name)
	os.makedirs(out_dirpath, exist_ok=True)
	out_fname_base = 'compression_vs'

	## Scene ID and Params
	test_set_id = 'test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
	middlebury_test_set_info = analyze_test_results_utils.middlebury_test_set_info
	scene_ids = middlebury_test_set_info['scene_ids']
	sbr_params_low_flux = middlebury_test_set_info['sbr_params_low_flux']
	sbr_params_high_flux = middlebury_test_set_info['sbr_params_high_flux']
	sbr_params = sbr_params_low_flux

	## Create spad dataloader object to get ground truth data for each scene
	spad_dataset = SpadDataset(datalist_fpath=os.path.join(io_dirpaths.datalists_dirpath, test_set_id+'.txt'), disable_rand_crop=True)
	(nr,nc,nt,tres_ps) = spad_dataset.get_spad_data_sample_params(0)

	## generate model names for all compression ratios
	for compression_ratio in compression_ratio_all:
	
		## Parameters for: Does decreasing the number of parameters hurt performance?
		# experiment_id = ''
		# encoding_type_all = ['csph1d', 'csph1d', 'csph1d', 'separable', 'separable', 'separable']
		# tdim_init_all = ['CoarseHist', 'HybridGrayFourier', 'TruncFourier', 'Rand', 'Rand', 'Rand']
		# tdim_init_all = ['CoarseHist', 'HybridGrayFourier', 'TruncFourier', 'Rand', 'Rand', 'Rand']
		# optCt_all = [False, False, False, True, True, True]
		# optC_all = [False, False, False, True, True, True]
		# spatial_down_factor_all = [1, 1, 1, 4, 4, 4]
		# num_tdim_blocks_all = [1, 1, 1, 1, 4, 16]

		experiment_id = ''
		encoding_type_all = ['csph1d', 'csph1d', 'separable', 'separable', 'separable']
		tdim_init_all = ['HybridGrayFourier', 'TruncFourier', 'Rand', 'Rand', 'Rand']
		optCt_all = [False, False, True, True, True]
		optC_all = [False, False, True, True, True]
		spatial_down_factor_all = [1, 1, 4, 4, 4]
		num_tdim_blocks_all = [1, 1, 1, 4, 16]

		# experiment_id = ''
		# encoding_type_all = ['csph1d', 'csph1d', 'separable', 'separable', 'separable', 'separable', 'separable']
		# tdim_init_all = ['HybridGrayFourier', 'TruncFourier', 'Rand', 'Rand', 'Rand', 'HybridGrayFourier', 'HybridGrayFourier']
		# optCt_all = [False, False, True, True, True, False, False]
		# optC_all = [False, False, True, True, True, True, True]
		# spatial_down_factor_all = [1, 1, 4, 4, 4, 4, 4]
		# num_tdim_blocks_all = [1, 1, 1, 4, 16, 1, 16]

		## Model IDs of the models we want to plot
		n_csph3d_models = len(encoding_type_all)
		
		## Generate names for all csph3d models at current compression
		for i in range(n_csph3d_models):
			spatial_down_factor = spatial_down_factor_all[i]
			(block_nt, block_nr, block_nc) = analyze_test_results_utils.compute_block_dims(spatial_down_factor, nt, num_tdim_blocks_all[i])
			k = analyze_test_results_utils.csph3d_compression2k(compression_ratio, block_nr, block_nc, block_nt)
			# Compose name
			model_name = analyze_test_results_utils.compose_csph3d_model_name(k=k, spatial_down_factor=spatial_down_factor, tdim_init=tdim_init_all[i], encoding_type=encoding_type_all[i], num_tdim_blocks=num_tdim_blocks_all[i], optCt=optCt_all[i], optC=optC_all[i])
			model_names.append(model_name)
			num_model_params.append(compute_csph3d_expected_num_params(encoding_type_all[i], block_nt, block_nr*block_nc, k))
			print("Model Info: ")
			print("	   Name: {}".format(model_names[i]))
			print("    CSPH3DLayer Num Params: {}".format(num_model_params[i]))


	## Get pretrained models dirpaths
	model_dirpaths = analyze_test_results_utils.get_model_dirpaths(model_names)
	# model_types = model_names
	# for model_name in model_names:
	# 	model_types.append(extract_model_type(model_name))

	## init model metrics dict
	model_metrics_all = init_model_metrics_dict(model_names, model_dirpaths)

	## process results and append metrics
	model_metrics_all, rec_depths_all = process_middlebury_test_results(scene_ids, sbr_params, model_metrics_all, spad_dataset, out_dirpath=None, save_depth_images=False, return_rec_depths=False)

	## Make into data frame
	curr_model_metrics_df = analyze_test_results_utils.metrics2dataframe(model_names, model_metrics_all)
	model_metrics_df = pd.concat((model_metrics_df, curr_model_metrics_df), axis=0)
	
	## scale MAE to make its units mm
	model_metrics_df['mae'] = model_metrics_df['mae']*1000
	
	## separate into csph3d and baselines
	model_metrics_df_csph3d = model_metrics_df[model_metrics_df['compression_ratio']>1]
	model_metrics_df_baselines = model_metrics_df[model_metrics_df['compression_ratio']==1]
	baseline_model_names = model_metrics_df_baselines['model_name'].unique()
	baselines_mean_metrics = model_metrics_df_baselines.groupby('model_name').mean()

	## Make plot for all metrics
	plt.clf()
	metric_id = 'mae'
	out_fname = out_fname_base + '_' + metric_id
	metric_ylim = None
	ax = sns.lineplot(data=model_metrics_df_csph3d, x="compression_ratio", y=metric_id, hue="model_type", marker="o", err_style="bars", errorbar=("se", 2), linewidth=2)
	ax.axhline(baselines_mean_metrics.loc[no_compression_baseline][metric_id], linewidth=2, linestyle='--', label=no_compression_baseline, color='red')
	ax.axhline(baselines_mean_metrics.loc[argmax_compression_baseline][metric_id], linewidth=2, linestyle='--', label=argmax_compression_baseline, color='blue')
	plt.legend()
	plt.grid(linestyle='--', linewidth=0.5)
	plot_utils.set_ticks(fontsize=12)
	plt.xlabel("Compression Ratio", fontsize=12)
	plt.ylabel("Mean Absolute Error (mm)", fontsize=12)
	plot_utils.set_xy_box()
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
