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
from csph_layers import compute_csph3d_expected_num_params
from research_utils import plot_utils
from spad_dataset import SpadDataset
import analyze_test_results_utils
from analyze_test_results_utils import init_model_metrics_dict, process_middlebury_test_results, get_hydra_io_dirpaths, compose_csph3d_model_names_list


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

if __name__=='__main__':

	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_compression_vs_test_metrics')

	compression_ratio_all = [32, 64, 128]
	# compression_ratio_all = [128]

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

	## generate all model names
	model_names = []
	num_model_params = []
	# add baselines first
	# no_compression_baseline = 'DDFN_C64B10/loss-kldiv_tv-1e-5'
	no_compression_baseline = 'DDFN_C64B10/norm-none/loss-kldiv_tv-1e-05'
	argmax_compression_baseline = 'DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-5'
	model_names.append(no_compression_baseline)
	model_names.append(argmax_compression_baseline)
	num_model_params = [0] * len(model_names)
	# generate all csph3d model names 
	encoding_type_all = ['csph1d', 'csph1d', 'separable', 'separable', 'separable']
	tdim_init_all = ['HybridGrayFourier', 'TruncFourier', 'Rand', 'Rand', 'Rand']
	optCt_all = [False, False, True, True, True]
	optC_all = [False, False, True, True, True]
	spatial_down_factor_all = [1, 1, 4, 4, 4]
	num_tdim_blocks_all = [1, 1, 1, 4, 16]
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
	num_model_params += csph3d_num_model_params

	## Get pretrained models dirpaths
	model_dirpaths = analyze_test_results_utils.get_model_dirpaths(model_names)

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
	plt.figure()
	metric_id = 'mae'
	out_fname = out_fname_base + '_' + metric_id
	metric_ylim = None
	ax = sns.lineplot(data=model_metrics_df_csph3d, x="compression_ratio", y=metric_id, hue="model_type", marker="o", err_style="bars", errorbar=("se", 2), linewidth=2)
	ax.axhline(baselines_mean_metrics.loc[no_compression_baseline][metric_id], linewidth=2, linestyle='--', label=no_compression_baseline, color='red')
	ax.axhline(baselines_mean_metrics.loc[argmax_compression_baseline][metric_id], linewidth=2, linestyle='--', label=argmax_compression_baseline, color='blue')
	plt.grid(linestyle='--', linewidth=0.5)
	plot_utils.set_ticks(fontsize=12)
	plt.xlabel("Compression Ratio", fontsize=14)
	plt.ylabel("Mean Absolute Error (mm)", fontsize=14)
	plot_utils.set_xy_box()
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	plt.legend()


	## Make plot for all metrics
	plt.figure()
	metric_id = 'ssim'
	out_fname = out_fname_base + '_' + metric_id
	metric_ylim = None
	ax = sns.lineplot(data=model_metrics_df_csph3d, x="compression_ratio", y=metric_id, hue="model_type", marker="o", err_style="bars", errorbar=("se", 2), linewidth=2)
	ax.axhline(baselines_mean_metrics.loc[no_compression_baseline][metric_id], linewidth=2, linestyle='--', label=no_compression_baseline, color='red')
	ax.axhline(baselines_mean_metrics.loc[argmax_compression_baseline][metric_id], linewidth=2, linestyle='--', label=argmax_compression_baseline, color='blue')
	plt.grid(linestyle='--', linewidth=0.5)
	plot_utils.set_ticks(fontsize=12)
	plt.xlabel("Compression Ratio", fontsize=14)
	plt.ylabel("SSIM", fontsize=14)
	plot_utils.set_xy_box()
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	plt.legend()


	## Make plot for all metrics
	plt.figure()
	metric_id = '5mm_tol_err'
	out_fname = out_fname_base + '_' + metric_id
	metric_ylim = None
	ax = sns.lineplot(data=model_metrics_df_csph3d, x="compression_ratio", y=metric_id, hue="model_type", marker="o", err_style="bars", errorbar=("se", 2), linewidth=2)
	ax.axhline(baselines_mean_metrics.loc[no_compression_baseline][metric_id], linewidth=2, linestyle='--', label=no_compression_baseline, color='red')
	ax.axhline(baselines_mean_metrics.loc[argmax_compression_baseline][metric_id], linewidth=2, linestyle='--', label=argmax_compression_baseline, color='blue')
	plt.grid(linestyle='--', linewidth=0.5)
	plot_utils.set_ticks(fontsize=12)
	plt.xlabel("Compression Ratio", fontsize=14)
	plt.ylabel("Percent Pixels with Errors < 5mm", fontsize=14)
	plot_utils.set_xy_box()
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	plt.legend()

	## Make plot for all metrics
	plt.figure()
	metric_id = '10mm_tol_err'
	out_fname = out_fname_base + '_' + metric_id
	metric_ylim = None
	ax = sns.lineplot(data=model_metrics_df_csph3d, x="compression_ratio", y=metric_id, hue="model_type", marker="o", err_style="bars", errorbar=("se", 2), linewidth=2)
	ax.axhline(baselines_mean_metrics.loc[no_compression_baseline][metric_id], linewidth=2, linestyle='--', label=no_compression_baseline, color='red')
	ax.axhline(baselines_mean_metrics.loc[argmax_compression_baseline][metric_id], linewidth=2, linestyle='--', label=argmax_compression_baseline, color='blue')
	plt.grid(linestyle='--', linewidth=0.5)
	plot_utils.set_ticks(fontsize=12)
	plt.xlabel("Compression Ratio", fontsize=14)
	plt.ylabel("Percent Pixels with Errors < 10mm", fontsize=14)
	plot_utils.set_xy_box()
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	plt.legend()

	## Make plot for all metrics
	plt.figure()
	metric_id = 'inverse_10mm_tol_err'
	out_fname = out_fname_base + '_' + metric_id
	metric_ylim = None
	ax = sns.lineplot(data=model_metrics_df_csph3d, x="compression_ratio", y=metric_id, hue="model_type", marker="o", err_style="bars", errorbar=("se", 2), linewidth=2)
	ax.axhline(baselines_mean_metrics.loc[no_compression_baseline][metric_id], linewidth=2, linestyle='--', label=no_compression_baseline, color='red')
	ax.axhline(baselines_mean_metrics.loc[argmax_compression_baseline][metric_id], linewidth=2, linestyle='--', label=argmax_compression_baseline, color='blue')
	plt.grid(linestyle='--', linewidth=0.5)
	plot_utils.set_ticks(fontsize=12)
	plt.xlabel("Compression Ratio", fontsize=14)
	plt.ylabel("Percent Pixels with Errors > 10mm", fontsize=14)
	plot_utils.set_xy_box()
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	plt.legend()
