'''
	This script creates lines plots of compression vs. different test set metrics
'''
#### Standard Library Imports
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from IPython.core import debugger
from torch import norm
breakpoint = debugger.set_trace

#### Local imports
from tof_utils import *
from research_utils import plot_utils
from spad_dataset import SpadDataset
import analyze_test_results_utils
from analyze_test_results_utils import init_model_metrics_dict, process_middlebury_test_results, get_hydra_io_dirpaths, compose_csph3d_model_names_list
from csph_layers import CSPH3DLayer

def down2x2_H(meas_H):
	down2x2_H = meas_H[..., 0::2, 0::2] + meas_H[..., 1::2, 0::2] + meas_H[..., 0::2, 1::2] + meas_H[..., 1::2, 1::2]
	return down2x2_H

def down4x4_H(meas_H):
	return down2x2_H(down2x2_H(meas_H))

def down8x8_H(meas_H):
	return down4x4_H(down4x4_H(meas_H))


def plot_depth_img(model_depths, min_scene_depth, max_scene_depth, model_id, out_dirpath, scene_fname, file_ext='svg'):
	plt.imshow(model_depths, vmin=min_scene_depth, vmax=max_scene_depth)
	plot_utils.remove_ticks()
	out_fname = model_id.replace('/','_')
	plot_utils.save_currfig(dirpath=os.path.join(out_dirpath, scene_fname), filename=out_fname, file_ext=file_ext)
	plt.pause(0.05)
	# save with colorbar
	plt.colorbar()
	plot_utils.save_currfig(dirpath=os.path.join(out_dirpath, scene_fname + '/withcbar'), filename=out_fname, file_ext=file_ext)
	plt.pause(0.05)
	plt.title(model_id)

if __name__=='__main__':

	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_tof_modalities_comparison')

	## Choose the test set to plot with
	# ## Middlebury with 8x downsampling (regualr test set)
	# experiment_name_base = 'tof_modalities_comparison/middlebury'
	# test_set_id = 'test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
	## Middlebury with 4x downsampling
	experiment_name_base = 'tof_modalities_comparison/middlebury_down4'
	test_set_id = 'test_middlebury_down4_SimSPADDataset_nr-144_nc-176_nt-1024_tres-98ps_dark-0_psf-0'
	## Middlebury with 2x downsampling
	experiment_name_base = 'tof_modalities_comparison/middlebury_down2'
	test_set_id = 'test_middlebury_down2_SimSPADDataset_nr-288_nc-352_nt-1024_tres-98ps_dark-0_psf-0'
	

	## output dirpaths
	out_dirpath = os.path.join(io_dirpaths.results_figures_dirpath, experiment_name_base)
	os.makedirs(out_dirpath, exist_ok=True)

	## Scene ids and signal and SBR parameters we want to plot for
	scene_ids = ['spad_Art']
	sbr_params = ['50_50']

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
	tau = nt*tres_ps*1e-12
	depth_range = time2depth(tau)
	delta_depth = depth_range / nt
	h_irf = spad_dataset.psf

	## Create csph layers
	itof_csph_layer = CSPH3DLayer(k=6, num_bins=nt, 
					tblock_init='TruncFourier', h_irf=h_irf, optimize_tdim_codes=False, optimize_codes=False, 
					nt_blocks=1, spatial_down_factor=1, 
					encoding_type='csph1d',
					csph_out_norm='none',
					account_irf=True,
					smooth_tdim_codes=False,
					apply_zncc_norm=True,
					zero_mean_tdim_codes=True)
	## Create csph layers
	dtof_csph_layer = CSPH3DLayer(k=nt, num_bins=nt, 
					tblock_init='CoarseHist', h_irf=h_irf, optimize_tdim_codes=False, optimize_codes=False, 
					nt_blocks=1, spatial_down_factor=1, 
					encoding_type='csph1d',
					csph_out_norm='none',
					account_irf=True,
					smooth_tdim_codes=False,
					apply_zncc_norm=True,
					zero_mean_tdim_codes=True)


	plt.close('all')

	for scene_id in scene_ids:
		for sbr_param in sbr_params:
			scene_fname = '{}_{}'.format(scene_id, sbr_param)
			spad_dataset_sample = spad_dataset.get_item_by_scene_name(scene_fname)
			
			## get lmf, argmax and gt depths
			gt_bins = spad_dataset_sample['bins'].squeeze().cpu().numpy()*nt
			gt_depths = bin2depth(gt_bins, num_bins=nt, tau=tau)
			min_scene_depth = gt_depths.flatten().min() - 0.5*gt_depths.flatten().std()
			max_scene_depth = gt_depths.flatten().max() + 0.5*gt_depths.flatten().std()
			## get lmf, argmax and gt depths
			argmax_bins = spad_dataset_sample['est_bins_argmax'].squeeze().cpu().numpy()*nt
			argmax_depths = bin2depth(argmax_bins, num_bins=nt, tau=tau)
			lmf_bins = spad_dataset_sample['est_bins_lmf'].squeeze().cpu().numpy()*nt
			lmf_depths = bin2depth(lmf_bins, num_bins=nt, tau=tau)

			## Get histogram image
			meas_H = spad_dataset_sample['spad']
			meas_H_down4x4 = down4x4_H(meas_H)


			## Processs with itof layer
			itof_Hhat = itof_csph_layer(meas_H).squeeze().cpu().numpy()
			itof_est_bins = np.argmax(itof_Hhat, axis=0)
			itof_depths = bin2depth(itof_est_bins, num_bins=nt, tau=tau)

			## Spad tof takes as input the full-re
			spadtof_Hhat = dtof_csph_layer(meas_H).squeeze().cpu().numpy()
			spadtof_est_bins = np.argmax(spadtof_Hhat, axis=0)
			spadtof_depths = bin2depth(spadtof_est_bins, num_bins=nt, tau=tau)

			## Spad tof takes as input the full-re
			dtof_Hhat = dtof_csph_layer(meas_H_down4x4).squeeze().cpu().numpy()
			dtof_est_bins = np.argmax(dtof_Hhat, axis=0)
			dtof_depths = bin2depth(dtof_est_bins, num_bins=nt, tau=tau)


			## plot intensity
			intensity_img = spad_dataset_sample['intensity'].squeeze().cpu().numpy()
			plt.clf()
			plt.imshow(intensity_img, cmap='gray')
			plot_utils.remove_ticks()
			plot_utils.save_currfig(dirpath=os.path.join(out_dirpath, scene_fname), filename='intensity_img', file_ext='svg')
			plt.pause(0.05)

			plt.figure()
			plot_depth_img(gt_depths, min_scene_depth, max_scene_depth, 'gt', out_dirpath, scene_fname, file_ext='svg')
			gt_depths_mae = np.abs(gt_depths-gt_depths).flatten().mean()
			print("gt_depths_mae = {}".format(gt_depths_mae))
			plt.pause(0.1)

			plt.figure()
			plot_depth_img(argmax_depths, min_scene_depth, max_scene_depth, 'argmax', out_dirpath, scene_fname, file_ext='svg')
			argmax_depths_mae = np.abs(argmax_depths-gt_depths).flatten().mean()
			print("argmax_depths_mae = {}".format(argmax_depths_mae))
			plt.pause(0.1)


			plt.figure()
			plot_depth_img(itof_depths, min_scene_depth, max_scene_depth, 'itof', out_dirpath, scene_fname, file_ext='svg')
			itof_depths_mae = np.abs(itof_depths-gt_depths).flatten().mean()
			print("itof_depths_mae = {}".format(itof_depths_mae))
			plt.pause(0.1)

			plt.figure()
			plot_depth_img(spadtof_depths, min_scene_depth, max_scene_depth, 'spadtof', out_dirpath, scene_fname, file_ext='svg')
			spadtof_depths_mae = np.abs(spadtof_depths-gt_depths).flatten().mean()
			print("spadtof_depths_mae = {}".format(spadtof_depths_mae))
			plt.pause(0.1)

			plt.figure()
			plot_depth_img(dtof_depths, min_scene_depth, max_scene_depth, 'dtof', out_dirpath, scene_fname, file_ext='svg')
			# dtof_depths_mae = np.abs(dtof_depths-gt_depths).flatten().mean()
			# print("dtof_depths_mae = {}".format(dtof_depths_mae))
			plt.pause(0.1)




