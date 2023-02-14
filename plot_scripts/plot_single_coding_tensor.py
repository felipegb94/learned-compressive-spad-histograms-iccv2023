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

	## Modulo 55ps dataset 1024x4x4 separable
	model_ckpt_fpath = 'outputs/modulo_nyuv2_64x64x1024_55ps/validate_new_modulo_dataset_20230207/DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-latest_2023-02-07_205141/checkpoints/epoch=13-step=46750-avgvalrmse=0.0449.ckpt'
	
	## 80 ps dataset 1024x4x4 separable
	model_ckpt_fpath = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models_rerun/DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-10-07_193425/checkpoints/epoch=29-step=103889-avgvalrmse=0.0177.ckpt'

	# ## 80 ps dataset 1024x1x1 learned
	# model_ckpt_fpath = 'outputs/nyuv2_64x64x1024_80ps/csph3D_tdim_baselines/DDFN_C64B10_CSPH3D/k8_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-10-05_115838/checkpoints/epoch=25-step=88306-avgvalrmse=0.0201.ckpt'

	# ## 55ps dataset 1024x1x1 learned
	# model_ckpt_fpath = 'outputs_fmu/nyuv2_64x64x1024_55ps/validate_new_dataset_20230116/DDFN_C64B10_CSPH3D/k8_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-01-18_072412/checkpoints/epoch=20-step=72722-avgvalrmse=0.0209.ckpt'
	
	
	
	model_ckpt_fname = os.path.basename(model_ckpt_fpath)
	model_dirpath = model_ckpt_fpath.split('/checkpoints/')[0]
	model_name = '/'.join(model_dirpath.split('/')[3:-1])

	model, _ = load_model_from_ckpt(model_name, model_ckpt_fname, logger=None, model_dirpath=model_dirpath)
	
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

	## Pre-process Cmat before plotting
	Cmat_tdim = Cmat_tdim.squeeze()
	model_name_min = get_model_name_min(model_name)

	## Pre-process matrix for visualization
	# Scale codes between -1 and 1
	Cmat_tdim_scaled = (Cmat_tdim - Cmat_tdim.min(axis=-1, keepdims=True)) / (Cmat_tdim.max(axis=-1, keepdims=True) - Cmat_tdim.min(axis=-1, keepdims=True)) 
	Cmat_tdim_scaled = (Cmat_tdim_scaled*2)-1
	# Cap the number of codes we want to plot 
	max_n_codes_to_plot = 16
	n_codes_to_plot = min(max_n_codes_to_plot, K)
	# Cmat_tdim_scaled = Cmat_tdim_scaled[0:n_codes_to_plot, ]

	plt.figure()
	ax = plot_tdim_coding_matrix(Cmat_tdim_scaled)
	plt.title(model_dirpath)



