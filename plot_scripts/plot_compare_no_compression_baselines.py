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
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from tof_utils import *
from csph_layers import compute_csph3d_expected_num_params
from research_utils import plot_utils
from spad_dataset import SpadDataset
import analyze_test_results_utils
from analyze_test_results_utils import init_model_metrics_dict, process_middlebury_test_results, get_hydra_io_dirpaths, compose_csph3d_model_names_list


if __name__=='__main__':

	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_compare_no_compression_baselines')

	##
	model_names = []
	num_model_params = []
	# add baselines first
	old_no_compression_baseline = 'DDFN_C64B10/loss-kldiv_tv-1e-5'
	old_argmax_compression_baseline = 'DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-5'
	model_names.append(old_no_compression_baseline)
	model_names.append(old_argmax_compression_baseline)

	# hyperparam search
	no_compression_baseline_lr1em4_tv1em5_nonorm = 'DDFN_C64B10/norm-none/loss-kldiv_tv-1e-05'
	model_names.append(no_compression_baseline_lr1em4_tv1em5_nonorm)
	# no_compression_baseline_lr1em4_tv1em5_linfnorm = 'DDFN_C64B10/norm-Linf/loss-kldiv_tv-1e-05'
	# model_names.append(no_compression_baseline_lr1em4_tv1em5_linfnorm)
	# no_compression_baseline_lr1em4_tv0_linfnorm = 'DDFN_C64B10/norm-Linf/loss-kldiv_tv-0.0'
	# model_names.append(no_compression_baseline_lr1em4_tv0_linfnorm)
	no_compression_baseline_lr1em3_tv1em5_linfnorm = 'DDFN_C64B10/norm-Linf/loss-kldiv_tv-1e-05'
	model_names.append(no_compression_baseline_lr1em3_tv1em5_linfnorm)
	no_compression_baseline_lr1em3_tv0_linfnorm = 'DDFN_C64B10/norm-Linf/loss-kldiv_tv-0.0'
	model_names.append(no_compression_baseline_lr1em3_tv0_linfnorm)

	# argmax_compression_baseline_lr1em4_tv1em5_nonorm = 'DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-05'
	# model_names.append(argmax_compression_baseline_lr1em4_tv1em5_nonorm)
	# argmax_compression_baseline_lr1em4_tv0_nonorm = 'DDFN_C64B10_Depth2Depth/loss-kldiv_tv-0.0'
	# model_names.append(argmax_compression_baseline_lr1em4_tv0_nonorm)
	# argmax_compression_baseline_lr1em3_tv1em5_nonorm = 'DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-05'
	# model_names.append(argmax_compression_baseline_lr1em3_tv1em5_nonorm)
	# argmax_compression_baseline_lr1em3_tv0_nonorm = 'DDFN_C64B10_Depth2Depth/loss-kldiv_tv-0.0'
	# model_names.append(argmax_compression_baseline_lr1em3_tv0_nonorm)


	## output dirpaths
	experiment_name = 'middlebury/test_set_metrics/no_compression_baselines'
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


	## Get pretrained models dirpaths
	model_dirpaths = analyze_test_results_utils.get_model_dirpaths(model_names)
	model_names_min = []
	for model_name in model_names:
		model_names_min.append(model_name)

	## init model metrics dict
	model_metrics_all = init_model_metrics_dict(model_names_min, model_dirpaths)

	## process results and append metrics
	model_metrics_all, rec_depths_all = process_middlebury_test_results(scene_ids, sbr_params, model_metrics_all, spad_dataset, out_dirpath=None, save_depth_images=False, return_rec_depths=False)

	## Make into data frame
	model_metrics_df = analyze_test_results_utils.metrics2dataframe(model_names_min, model_metrics_all)

	## scale MAE to make its units mm
	model_metrics_df['mae'] = model_metrics_df['mae']*1000

	## Make plot for all metrics
	plt.figure()
	metric_id = 'mae'
	out_fname =  metric_id
	metric_ylim = (0, 50)
	# metric_ylim = None
	analyze_test_results_utils.plot_test_dataset_metrics(model_metrics_df, metric_id=metric_id, ylim=metric_ylim, title='')
	# remove ticks and legend for saving
	# plot_utils.remove_xticks()
	# plt.xlabel(''); plt.ylabel('')
	plt.grid(linestyle='--', linewidth=0.5)
	# save figure with legend
	# plot_utils.save_currfig(dirpath=out_dirpath, filename='legend_'+out_fname, file_ext='svg')
	# save figure without legend
	plt.gca().get_legend().remove()
	# plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# add title after saving so we know what we plotted
	plt.title('{} - No Compression Baselines'.format(metric_id))

	# ## Make plot for all metrics
	# plt.figure()
	# metric_id = 'ssim'
	# out_fname =  metric_id
	# metric_ylim = None
	# # metric_ylim = None
	# analyze_test_results_utils.plot_test_dataset_metrics(model_metrics_df, metric_id=metric_id, ylim=metric_ylim, title='')
	# # remove ticks and legend for saving
	# # plot_utils.remove_xticks()
	# # plt.xlabel(''); plt.ylabel('')
	# plt.grid(linestyle='--', linewidth=0.5)
	# # save figure with legend
	# # plot_utils.save_currfig(dirpath=out_dirpath, filename='legend_'+out_fname, file_ext='svg')
	# # save figure without legend
	# plt.gca().get_legend().remove()
	# # plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# # add title after saving so we know what we plotted
	# plt.title('{} - No Compression Baselines'.format(metric_id))
		
	# 	## Make plot for all metrics
	# 	plt.figure()
	# 	metric_id = '5mm_tol_err'
	# 	metric_ylim = (0.1, 0.9)
	# 	# metric_ylim = None
	# 	out_fname =  metric_id
	# 	analyze_test_results_utils.plot_test_dataset_metrics(model_metrics_df, metric_id=metric_id, ylim=metric_ylim, title='')
	# 	# remove ticks and legend for saving
	# 	plot_utils.remove_xticks()
	# 	plt.xlabel(''); plt.ylabel('')
	# 	plt.grid(linestyle='--', linewidth=0.5)
	# 	# save figure with legend
	# 	# plot_utils.save_currfig(dirpath=out_dirpath, filename='legend_'+out_fname, file_ext='svg')
	# 	# save figure without legend
	# 	plt.gca().get_legend().remove()
	# 	# plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
	# 	# add title after saving so we know what we plotted
	# plt.title('{} - No Compression Baselines'.format(metric_id))

