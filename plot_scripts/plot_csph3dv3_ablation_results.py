'''
	This script plots the test set results for models that were trained with different settings introduced in CSPH3Dv3 model (currently the latest version)
	The test commands for these models appear under `scripts_test/test_csph3dv3_ablation.sh`
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
from analyze_test_results_utils import init_model_metrics_dict, process_middlebury_test_results, get_hydra_io_dirpaths


def simplify_model_name(model_name):
	## only keep model params
	model_name_min = model_name.split('/')[1]
	model_name_min = 'norm' + model_name_min.split('_norm')[-1]
	model_name_min = model_name_min.replace('False', '0')
	model_name_min = model_name_min.replace('True', '1')
	# model_name_min = model_name_min.replace('_smoothtdimC-0', '')
	# model_name_min = model_name_min.replace('_irf-0', '')

	return model_name_min

def plot_test_dataset_metrics(model_metrics, metric_id='mae', ylim=None):
	plt.clf()
	plot_utils.update_fig_size(height=8, width=16)
	ax = plt.gca()
	# cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True, reverse=True, light=0.8, dark=0.3)
	# cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
	cmap = sns.color_palette("mako", n_colors=len(model_metrics_df['mean_sbr'].unique()))
	## legend="full" is needed to display the full name of the hue variable
	## set zorder to 0 to make sur eit appears below boxplot
	ax = sns.swarmplot(data=model_metrics, x='model_name', y=metric_id, orient="v", hue="mean_sbr", dodge=True, legend="full", palette=cmap)
	boxprops = {'facecolor':'black', 'linewidth': 1, 'alpha': 0.3}
	# medianprops = {'linewidth': 4, 'color': '#ff5252'}
	# medianprops = {'linewidth': 4, 'color': '#4ba173'}
	medianprops = {'linewidth': 3, 'color': '#424242', "solid_capstyle": "butt"}
	# meanprops={"linestyle":"--","linewidth": 3, "color":"white"}
	meanprops={"marker":"o",
					"markerfacecolor":"white", 
					"markeredgecolor":"black",
					"markersize":"14"}
	ax = sns.boxplot(data=model_metrics, x='model_name', y=metric_id, ax=ax, orient="v", showfliers = False, boxprops=boxprops, medianprops=medianprops, meanprops=meanprops, showmeans=True)
	ax.legend(title='Mean SBR', fontsize=14, title_fontsize=14)
	plt.xticks(rotation=10)
	if(not (ylim is None)):
		plt.ylim(ylim)
	plot_utils.save_currfig_png(dirpath=out_dirpath, filename=base_fname + '{}_sbr-hue'.format(metric_id))

if __name__=='__main__':

	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_norm_ablation_results')

	## Add high flux test results
	plot_high_flux = True

	## output dirpaths
	experiment_name = 'middlebury/csph3dv3_ablation'
	out_dirpath = os.path.join(io_dirpaths.results_weekly_dirpath, experiment_name)
	os.makedirs(out_dirpath, exist_ok=True)

	## Scene ID and Params
	test_set_id = 'test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
	scene_ids = ['spad_Art', 'spad_Reindeer', 'spad_Books', 'spad_Moebius', 'spad_Bowling1', 'spad_Dolls', 'spad_Laundry', 'spad_Plastic']
	sbr_params = ['2_2','2_10','2_50','5_2','5_10','5_50','10_2','10_10','10_50']
	sbr_params_high_flux = ['10_200', '10_500', '10_1000', '50_50', '50_200', '50_500', '50_1000'] ## more than 100 photons per pixel on average
	
	if(plot_high_flux):
		sbr_params = sbr_params + sbr_params_high_flux
		base_fname = 'high_flux_test_results'
	else:
		sbr_params = sbr_params
		base_fname = 'test_results'

	# base_fname = 'test_set_irf_' + base_fname

	## Create spad dataloader object to get ground truth data for each scene
	spad_dataset = SpadDataset(datalist_fpath=os.path.join(io_dirpaths.datalists_dirpath, test_set_id+'.txt'), disable_rand_crop=True)

	## Model IDs of the models we want to plot
	model_names = []
	# model_names.append('DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0')
	model_names.append('DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-0.0')
	model_names.append('DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal_irf-True_zn-False_zeromu-False_smoothtdimC-False/loss-kldiv_tv-0.0')
	model_names.append('DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0')
	model_names.append('DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-False_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0')
	model_names.append('DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0')
	model_names.append('DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-True_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0')
	model_names.append('DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-True_zn-True_zeromu-True_smoothtdimC-True/loss-kldiv_tv-0.0')

	## Get pretrained models dirpaths
	pretrained_models_all = io_ops.load_json('pretrained_models_rel_dirpaths.json')
	model_dirpaths = []
	model_names_min = []
	for model_name in model_names:
		model_dirpaths.append(pretrained_models_all[model_name]['rel_dirpath'])
		model_names_min.append(simplify_model_name(model_name))

	## init model metrics dict
	# model_metrics_all = init_model_metrics_dict(model_names, model_dirpaths)
	model_metrics_all = init_model_metrics_dict(model_names_min, model_dirpaths)

	## process results and append metrics
	model_metrics_all, rec_depths_all = process_middlebury_test_results(scene_ids, sbr_params, model_metrics_all, spad_dataset, out_dirpath=None, save_depth_images=False, return_rec_depths=False)

	## Make into data frame
	model_metrics_df = pd.DataFrame()	
	for model_name in model_names_min:
		model_metrics_df_curr = pd.DataFrame()
		model_metrics_df_curr['mae'] = model_metrics_all[model_name]['mae']
		model_metrics_df_curr['mse'] = model_metrics_all[model_name]['mse']
		model_metrics_df_curr['1mm_tol_err'] = model_metrics_all[model_name]['1mm_tol_err']
		model_metrics_df_curr['5mm_tol_err'] = model_metrics_all[model_name]['5mm_tol_err']
		model_metrics_df_curr['10mm_tol_err'] = model_metrics_all[model_name]['10mm_tol_err']
		model_metrics_df_curr['model_name'] = [model_name]*len(model_metrics_all[model_name]['mae'])
		model_metrics_df_curr['mean_sbr'] = model_metrics_all['sbr_params']['mean_sbr']
		model_metrics_df_curr['mean_signal_photons'] = model_metrics_all['sbr_params']['mean_signal_photons']
		model_metrics_df_curr['mean_bkg_photons'] = model_metrics_all['sbr_params']['mean_bkg_photons']
		model_metrics_df_curr['is_high_flux'] = (model_metrics_df_curr['mean_signal_photons'] + model_metrics_df_curr['mean_bkg_photons']) > 100
		model_metrics_df = pd.concat((model_metrics_df, model_metrics_df_curr), axis=0)

	# model_metrics_df_filtered = model_metrics_df[model_metrics_df['model_name'].str.contains('zeromu-1')]
	model_metrics_df_filtered = model_metrics_df


	plot_test_dataset_metrics(model_metrics_df_filtered, metric_id='mae', ylim=(0.0025,0.05) )

	# plt.figure()
	# plot_test_dataset_metrics(model_metrics_df_filtered, metric_id='mse')

	# plt.figure()
	# plot_test_dataset_metrics(model_metrics_df_filtered, metric_id='10mm_tol_err')

	# plt.figure()
	# plot_test_dataset_metrics(model_metrics_df_filtered, metric_id='5mm_tol_err')

	# plt.figure()
	# plot_test_dataset_metrics(model_metrics_df_filtered, metric_id='1mm_tol_err')
