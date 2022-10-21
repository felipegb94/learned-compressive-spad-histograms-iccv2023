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
from model_utils import load_model_from_ckpt

def get_pretty_C(Cmat, col2row_ratio=1.35):
	n_maxres = Cmat.shape[1]
	n_codes = Cmat.shape[0]
	if((n_maxres // 2) <= n_codes): col2row_ratio=1
	n_row_per_code = int(np.floor(n_maxres / n_codes) / col2row_ratio)
	n_rows = n_row_per_code*n_codes
	n_cols = n_maxres
	pretty_C = np.zeros((n_rows, n_cols))
	for i in range(n_codes):
		start_row = i*n_row_per_code
		end_row = start_row + n_row_per_code
		pretty_C[start_row:end_row, :] = Cmat[i:i+1, :] 
	return pretty_C

def plot_tdim_coding_matrix(Cmat):
	pretty_C = get_pretty_C(Cmat_tdim_scaled, col2row_ratio=3)
	plt.clf()
	ax = plt.gca()
	plt.imshow(pretty_C, cmap='gray', vmin=-1, vmax=1)
	plot_utils.remove_ticks()
	plt.pause(0.1)
	return ax

def get_model_name_min(model_name):
	model_name_min = model_name.replace("DDFN_C64B10_CSPH3D/","")
	model_name_min = model_name_min.replace("/loss-kldiv_tv-0.0","")
	return model_name_min

if __name__=='__main__':


	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_coding_tensors')

	compression_ratio = 128
	# compression_ratio_all = [128]

	max_n_codes_to_plot = 16

	model_metrics_df = pd.DataFrame()

	## output dirpaths
	experiment_name = 'coding_tensors/CR-{}'.format(compression_ratio)
	out_dirpath = os.path.join(io_dirpaths.results_figures_dirpath, experiment_name)
	os.makedirs(out_dirpath, exist_ok=True)

	## generate all model names
	model_names = []
	num_model_params = []
	# generate all csph3d model names 
	encoding_type_all = ['csph1d', 'csph1d', 'csph1d', 'separable', 'separable']
	tdim_init_all = ['TruncFourier', 'HybridGrayFourier', 'Rand', 'Rand', 'Rand']
	optCt_all = [False, False, True, True, True]
	optC_all = [False, False, True, True, True]
	spatial_down_factor_all = [1, 1, 1, 4, 4]
	num_tdim_blocks_all = [1, 1, 1, 1, 4]
	# encoding_type_all = ['csph1d']
	# tdim_init_all = ['Rand']
	# optCt_all = [True]
	# optC_all = [True]
	# spatial_down_factor_all = [1, 1, 1, 4, 4]
	# num_tdim_blocks_all = [1, 1, 1, 1, 4]

	(csph3d_model_names, csph3d_num_model_params) = compose_csph3d_model_names_list([compression_ratio]
									, spatial_down_factor_all
									, num_tdim_blocks_all
									, tdim_init_all
									, optCt_all
									, optC_all
									, encoding_type_all
									, nt = 1024
		)
	model_names += csph3d_model_names
	num_model_params += csph3d_num_model_params

	## Get pretrained models dirpaths
	model_dirpaths = analyze_test_results_utils.get_model_dirpaths(model_names)

	## Get the fpath for model checkpoints
	model_ckpt_fpaths, model_ckpt_fnames = analyze_test_results_utils.get_model_ckpt_fpaths(model_names)

	## Load each model and save its coding tensors
	Cmat_txydim_all = []
	Cmat_tdim_all = []
	Cmat_xydim_all = []
	K_all = []
	for i, ckpt_fpath in enumerate(model_ckpt_fpaths):
		model_name = model_names[i]
		ckpt_fname = model_ckpt_fnames[i]
		model_dirpath = model_dirpaths[i]
		model, _ = load_model_from_ckpt(model_name, ckpt_fname, logger=None, model_dirpath=model_dirpath)
		encoding_layer = model.csph3d_layer
		encoding_type = model.csph3d_layer.encoding_type
		K = encoding_layer.k
		if(encoding_type == 'separable'):
			Cmat_tdim = encoding_layer.Cmat_tdim.detach().cpu().numpy()
			Cmat_xydim = encoding_layer.Cmat_xydim.detach().cpu().numpy()
			Cmat_txydim = Cmat_xydim*Cmat_tdim
		elif(encoding_type == 'full'):
			Cmat_txydim = encoding_layer.Cmat_txydim.detach().cpu().numpy()
			Cmat_tdim = Cmat_txydim.mean(axis=(-1,-2), keepdims=True)
			Cmat_xydim = Cmat_txydim.mean(axis=(-3), keepdims=True)
		elif(encoding_type == 'csph1d'):
			Cmat_tdim = encoding_layer.Cmat_tdim.detach().cpu().numpy()
			Cmat_xydim = np.ones((K, 1, 1, 1, 1)).astype(Cmat_tdim.dtype)
			Cmat_txydim = Cmat_xydim*Cmat_tdim
		K_all.append(K)
		Cmat_txydim_all.append(Cmat_txydim)
		Cmat_tdim_all.append(Cmat_tdim)
		Cmat_xydim_all.append(Cmat_xydim)

	for i, model_name in enumerate(model_names):
		print("Cmats for: {}".format(model_name))
		print("    K: {}".format(K_all[i]))
		print("    Cmat_tdim: {}".format(Cmat_tdim_all[i].shape))
		print("    Cmat_xydim: {}".format(Cmat_xydim_all[i].shape))
		print("    Cmat_txydim: {}".format(Cmat_txydim_all[i].shape))

		## Plot coding matrix
		K = K_all[i]
		Cmat_tdim = Cmat_tdim_all[i].squeeze()
		model_name_min = get_model_name_min(model_name)

		## Pre-process matrix for visualization
		# Scale codes between -1 and 1
		Cmat_tdim_scaled = (Cmat_tdim - Cmat_tdim.min(axis=-1, keepdims=True)) / (Cmat_tdim.max(axis=-1, keepdims=True) - Cmat_tdim.min(axis=-1, keepdims=True)) 
		Cmat_tdim_scaled = (Cmat_tdim_scaled*2)-1
		# Cap the number of codes we want to plot 
		n_codes_to_plot = min(max_n_codes_to_plot, K)
		# Cmat_tdim_scaled = Cmat_tdim_scaled[0:n_codes_to_plot, ]

		## Plot coding matrix
		plt.figure()
		ax = plot_tdim_coding_matrix(Cmat_tdim_scaled)
		plot_utils.update_fig_size(height=4, width=12)
		plot_utils.save_currfig(dirpath=out_dirpath, filename='{}_Cmat'.format(model_name_min), file_ext='svg')
		plt.title(model_name_min)

		## plot the first 8 rows in 4 plots
		plt.figure()
		for j in range(4):
			start_row = j*2
			end_row = start_row + 2
			# plot first 4 rows
			plt.clf()
			plot_utils.update_fig_size(height=0.75, width=12)
			# plt.plot(Cmat_tdim_scaled[0:4,:].transpose())
			plt.plot(Cmat_tdim_scaled[start_row:end_row,:].transpose(), linewidth=3)
			# plot_utils.set_ticks(fontsize=12)
			plot_utils.remove_xticks()
			plt.grid(linestyle='--', linewidth=0.5)
			plt.xlim((0, Cmat_tdim_scaled.shape[-1]))
			plot_utils.save_currfig(dirpath=out_dirpath, filename='{}_rows-{}-{}'.format(model_name_min, start_row, end_row-1).format(model_name_min), file_ext='svg')
			plt.title(model_name_min)
			plt.pause(0.1)


