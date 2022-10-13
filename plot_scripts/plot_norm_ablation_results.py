'''
    This script plots the test set results for models that were trained with different normalization strategies
    The test commands for these models appear under `scripts_test/test_csph3d_norm_strategies.sh`
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


def parse_csph3d_params(model_name):
	if('CSPH3D' in model_name):
		down_factor = int(model_name.split('_down')[-1].split('_')[0])
		k = int(model_name.split('/k')[-1].split('_')[0])
		nt_blocks = int(model_name.split('_Mt')[-1].split('_')[0])
	else:
		down_factor = None
		k = None
		nt_blocks = None
	return (down_factor, nt_blocks, k)

def simplify_model_name(model_name):
	## parset params
	(down_factor, nt_blocks, k) = parse_csph3d_params(model_name)
	## simplify
	# model_name_min = model_name.replace('DDFN_C64B10', 'db3D')
	# model_name_min = model_name.replace('loss-kldiv_', '')
	# model_name_min = model_name_min.replace('_down{}_Mt{}'.format(down_factor, nt_blocks), '_{}x{}x{}'.format(1024//nt_blocks, down_factor, down_factor))

	model_name_min = model_name.split('/')[1]
	model_name_min = 'norm' + model_name_min.split('_norm')[-1]
	model_name_min = model_name_min.replace('False', '0')
	model_name_min = model_name_min.replace('True', '1')
	model_name_min = model_name_min.replace('_smoothtdimC-0', '')
	model_name_min = model_name_min.replace('_irf-0', '')

	return model_name_min

if __name__=='__main__':

	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_norm_ablation_results')

	## Add high flux test results
	plot_high_flux = True

	## output dirpaths
	experiment_name = 'middlebury/norm_ablation'
	out_dirpath = os.path.join(io_dirpaths.results_weekly_dirpath, experiment_name)
	os.makedirs(out_dirpath, exist_ok=True)

	## Scene ID and Params
	test_set_id = 'test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
	middlebury_test_set_info = analyze_test_results_utils.middlebury_test_set_info 
	scene_ids = middlebury_test_set_info['scene_ids']
	sbr_params_low_flux = middlebury_test_set_info['sbr_params_low_flux']
	sbr_params_high_flux = middlebury_test_set_info['sbr_params_high_flux']
	
	if(plot_high_flux):
		sbr_params = sbr_params_low_flux + sbr_params_high_flux
		base_fname = 'high_flux_test_results'
	else:
		sbr_params = sbr_params_low_flux
		base_fname = 'test_results'

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
	model_dirpaths = analyze_test_results_utils.get_model_dirpaths(model_names)
	model_names_min = []
	for model_name in model_names:
		model_names_min.append(simplify_model_name(model_name))

	## init model metrics dict
	# model_metrics_all = init_model_metrics_dict(model_names, model_dirpaths)
	model_metrics_all = init_model_metrics_dict(model_names_min, model_dirpaths)

	## process results and append metrics
	model_metrics_all, rec_depths_all = process_middlebury_test_results(scene_ids, sbr_params, model_metrics_all, spad_dataset, out_dirpath=None, save_depth_images=False, return_rec_depths=False)

	## Make into data frame
	model_metrics_df = analyze_test_results_utils.metrics2dataframe(model_names_min, model_metrics_all)


	plt.clf()
	plot_utils.update_fig_size(height=8, width=14)
	# cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True, reverse=True, light=0.8, dark=0.3)
	# cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
	cmap = sns.color_palette("mako", n_colors=len(model_metrics_df['mean_sbr'].unique()))
	## legend="full" is needed to display the full name of the hue variable
	ax = sns.swarmplot(data=model_metrics_df, x='model_name', y='mae', orient="v", hue="mean_sbr", dodge=True, legend="full", palette=cmap)
	ax.legend(title='Mean SBR', fontsize=14, title_fontsize=14)
	plt.xticks(rotation=15)
	plot_utils.save_currfig_png(dirpath=out_dirpath, filename=base_fname + '_sbr-hue')


	plt.clf()
	plot_utils.update_fig_size(height=8, width=14)
	# cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True, reverse=True, light=0.8, dark=0.3)
	# cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
	cmap = sns.color_palette("mako_r", n_colors=2)
	## legend="full" is needed to display the full name of the hue variable
	ax = sns.swarmplot(data=model_metrics_df, x='model_name', y='mae', orient="v", hue="is_high_flux", dodge=False, legend="full", palette=cmap)
	ax.legend(title='Flux > 100 photons', fontsize=14, title_fontsize=14)
	plt.xticks(rotation=15)
	plot_utils.save_currfig_png(dirpath=out_dirpath, filename=base_fname + '_highflux-hue')
	
	## plot only the zero mean codes models
	plt.clf()
	plot_utils.update_fig_size(height=8, width=14)
	# cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True, reverse=True, light=0.8, dark=0.3)
	# cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
	cmap = sns.color_palette("mako", n_colors=len(model_metrics_df['mean_sbr'].unique()))
	## legend="full" is needed to display the full name of the hue variable
	ax = sns.swarmplot(data=model_metrics_df[model_metrics_df['model_name'].str.contains('zeromu-1')], x='model_name', y='mae', orient="v", hue="mean_sbr", dodge=True, legend="full", palette=cmap)
	ax.legend(title='Mean SBR', fontsize=14, title_fontsize=14)
	plt.xticks(rotation=15)
	plot_utils.save_currfig_png(dirpath=out_dirpath, filename=base_fname + '_sbr-hue_zeromu-1')

	
	plt.clf()
	plot_utils.update_fig_size(height=8, width=14)
	# cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True, reverse=True, light=0.8, dark=0.3)
	# cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
	cmap = sns.color_palette("mako_r", n_colors=2)
	## legend="full" is needed to display the full name of the hue variable
	ax = sns.swarmplot(data=model_metrics_df[model_metrics_df['model_name'].str.contains('zeromu-1')], x='model_name', y='mae', orient="v", hue="is_high_flux", dodge=False, legend="full", palette=cmap)
	ax.legend(title='Flux > 100 photons', fontsize=14, title_fontsize=14)
	plt.xticks(rotation=15)
	plot_utils.save_currfig_png(dirpath=out_dirpath, filename=base_fname + '_highflux-hue_zeromu-1')
	
	# plt.ylim(0,0.5)


	# sns.swarmplot(data=model_metrics_all['DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0']['mae'],orient='v')
	# sns.swarmplot(data=model_metrics_all['argmax']['mae'],orient='v')






