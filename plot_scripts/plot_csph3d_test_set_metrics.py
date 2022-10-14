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


def simplify_model_name(model_name):
	## only keep model params
	model_name_min = model_name.replace('DDFN_C64B10_CSPH3D', '')
	model_name_min = model_name_min.replace('/loss-kldiv_tv-0.0', '')
	model_name_min = model_name_min.replace('False', '0')
	model_name_min = model_name_min.replace('True', '1')
	model_name_min = model_name_min.replace('_smoothtdimC-0', '')
	model_name_min = model_name_min.replace('_irf-0', '')
	return model_name_min

if __name__=='__main__':

	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_csph3d_test_set_metrics')

	compression_ratio_all = [32, 64, 128]
	compression_ratio_all = [128]

	## plot for each compression ratio
	for compression_ratio in compression_ratio_all:
	
		out_fname_base = 'CR-{}'.format(compression_ratio) 
		
		## Parameters for: Does decreasing the number of parameters hurt performance?
		experiment_id = 'Cmat_size_effect'
		encoding_type_all = ['full', 'separable', 'separable', 'separable']
		tdim_init_all = ['Rand']*len(encoding_type_all)
		optCt_all = [True]*len(encoding_type_all)
		optC_all = [True]*len(encoding_type_all)
		spatial_down_factor_all = [4]*len(encoding_type_all)
		num_tdim_blocks_all = [1, 1, 4, 16]

		# ## Parameters for: Spatial kernel size effect?
		# experiment_id = 'spatial_block_dims_effect'
		# encoding_type_all = ['csph1d', 'csph1d', 'separable', 'separable',  'separable']
		# spatial_down_factor_all = [1, 1, 2, 4, 8]
		# tdim_init_all = ['Rand']*len(encoding_type_all)
		# optCt_all = [True]*len(encoding_type_all)
		# optC_all = [True]*len(encoding_type_all)
		# tdim_init_all[0] = 'HybridGrayFourier'
		# optCt_all[0] = False
		# optC_all[0] = False
		# num_tdim_blocks_all = [1]*len(encoding_type_all)

		# ## Parameters for: Does a good initialization Help Performance?
		# experiment_id = 'tdim_init_effect'
		# tdim_init_all = ['Rand', 'HybridGrayFourier', 'HybridGrayFourier', 'Rand', 'HybridGrayFourier', 'HybridGrayFourier']
		# optCt_all = [True, True, False, True, True, False]
		# optC_all = [True]*len(tdim_init_all)
		# encoding_type_all = ['separable']*len(tdim_init_all)
		# spatial_down_factor_all = [4]*len(tdim_init_all)
		# num_tdim_blocks_all = [1, 1, 1, 16, 16, 16]

		## output dirpaths
		experiment_name = 'middlebury/test_set_metrics/{}'.format(experiment_id)
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

		## Model IDs of the models we want to plot
		n_csph3d_models = len(encoding_type_all)
		model_names = []
		num_model_params = []
		(model_names, num_model_params) = compose_csph3d_model_names_list(compression_ratio_all
										, spatial_down_factor_all
										, num_tdim_blocks_all
										, tdim_init_all
										, optCt_all
										, optC_all
										, encoding_type_all
										, nt = nt
			)

		## Get pretrained models dirpaths
		model_dirpaths = analyze_test_results_utils.get_model_dirpaths(model_names)
		model_names_min = []
		for model_name in model_names:
			model_names_min.append(simplify_model_name(model_name))

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
		out_fname = out_fname_base + '_' + metric_id
		metric_ylim = (0, 50)
		# metric_ylim = None
		analyze_test_results_utils.plot_test_dataset_metrics(model_metrics_df, metric_id=metric_id, ylim=metric_ylim, title='')
		# remove ticks and legend for saving
		plot_utils.remove_xticks()
		plt.xlabel(''); plt.ylabel('')
		plt.grid(linestyle='--', linewidth=0.5)
		# save figure with legend
		# plot_utils.save_currfig(dirpath=out_dirpath, filename='legend_'+out_fname, file_ext='svg')
		# save figure without legend
		plt.gca().get_legend().remove()
		# plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
		# add title after saving so we know what we plotted
		plt.title('{} - Compressive Histograms at {}x Compression'.format(metric_id, compression_ratio))
		
		## Make plot for all metrics
		plt.figure()
		metric_id = '5mm_tol_err'
		metric_ylim = (0.1, 0.9)
		# metric_ylim = None
		out_fname = out_fname_base + '_' + metric_id
		analyze_test_results_utils.plot_test_dataset_metrics(model_metrics_df, metric_id=metric_id, ylim=metric_ylim, title='')
		# remove ticks and legend for saving
		plot_utils.remove_xticks()
		plt.xlabel(''); plt.ylabel('')
		plt.grid(linestyle='--', linewidth=0.5)
		# save figure with legend
		# plot_utils.save_currfig(dirpath=out_dirpath, filename='legend_'+out_fname, file_ext='svg')
		# save figure without legend
		plt.gca().get_legend().remove()
		# plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')
		# add title after saving so we know what we plotted
		plt.title('{} - Compressive Histograms at {}x Compression'.format(metric_id, compression_ratio))

