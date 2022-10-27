'''
	Plots the recovered depth images from the linospad dataset without having to crop them
'''

#### Standard Library Imports
import enum
import sys
import os
sys.path.append('../')
sys.path.append('./')

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from tof_utils import bin2depth
from research_utils import plot_utils
from spad_dataset import Lindell2018LinoSpadDataset
import analyze_test_results_utils
from analyze_test_results_utils import init_model_metrics_dict, process_middlebury_test_results, get_hydra_io_dirpaths, compose_csph3d_model_names_list

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
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_linospad_results')

	## Choose the test set to plot with
	## Regular test set
	experiment_name = 'linospad_results'
	test_set_id = 'test_lindell2018_linospad_captured'

	## output dirpaths
	out_dirpath = os.path.join(io_dirpaths.results_figures_dirpath, experiment_name)
	os.makedirs(out_dirpath, exist_ok=True)

	## Create dataset object and load raw data for this scene
	dataset = Lindell2018LinoSpadDataset(datalist_fpath='datalists/{}.txt'.format(test_set_id))
	(nt, nr, nc, time_res_ps) = dataset.get_spad_data_sample_params(0)
	time_res = time_res_ps*1e-12
	tau = time_res*nt
	# frame_num = 0

	## scene we want to plot
	# scene_id = 'elephant'
	# scene_id = 'checkerboard'
	scene_id = 'stairs_ball'
	scene_id = 'lamp'
	scene_id = 'stuff'

	## Create dataset object and load raw data for this scene
	raw_data_sample = dataset.get_item_by_scene_name(scene_id)
	plt.clf()
	plt.imshow(raw_data_sample['intensity_img'].cpu().numpy())
	plot_utils.remove_ticks()
	plot_utils.save_currfig(dirpath = out_dirpath, filename = scene_id, file_ext='svg')
	plt.title('Intensity Image from Co-located Camera')

	## add all model dirpaths
	model_dirpaths = []
	
	## Generate model_names for the models we want to plots depths for
	model_names = []
	# no_compression_baseline = 'DDFN_C64B10/loss-kldiv_tv-1e-5'
	no_compression_baseline = 'DDFN_C64B10/norm-none/loss-kldiv_tv-1e-05'
	argmax_compression_baseline = 'DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-5'
	model_names.append(no_compression_baseline)
	model_names.append(argmax_compression_baseline)

	## CSPH3D models: Temporal vs. Spatio-Temporal Compression
	encoding_type_all = ['csph1d', 'csph1d', 'csph1d', 'csph1d', 'separable', 'separable']
	tdim_init_all = ['CoarseHist', 'TruncFourier', 'HybridGrayFourier', 'Rand', 'Rand', 'Rand']
	optCt_all = [False,False, False, True, True, True]
	optC_all = [False,False, False, True, True, True]
	spatial_down_factor_all = [1,1, 1, 1, 4, 4]
	num_tdim_blocks_all = [1, 1, 1, 1, 1, 4]
	compression_ratio_all = [32, 64, 128]
	# compression_ratio_all = [32,]
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

	(vmin, vmax) = (0.5, 4)
	if(scene_id == 'elephant'): (vmin,vmax)=(0.3, 2.3)
	if(scene_id == 'lamp'): (vmin,vmax)=(0.3, 2.3)
	if(scene_id == 'stuff'): (vmin, vmax)=(0.5, 2.2)
	if(scene_id == 'checkerboard'): (vmin, vmax)=(0.6, 1.1)
	if(scene_id == 'stairs_ball'): (vmin, vmax)=(0.8, 3.0)

	## load the depths (in units of bins)
	rec_depths_all = []
	for i, model_dir in enumerate(model_dirpaths):
		model_dir = model_dirpaths[i]
		fpath = os.path.join(model_dir, '{}/{}.npz'.format(test_set_id, scene_id))
		print(fpath)
		
		data = np.load(fpath)
		rec_bins = data['dep_re'].squeeze()*nt
		rec_depths = bin2depth(rec_bins, num_bins=nt, tau=tau)
		rec_depths_all.append(rec_depths)
	# compute argmax depth
	rec_depths_argmax = bin2depth(raw_data_sample['est_bins_argmax'].cpu().numpy()*nt, num_bins=nt, tau=tau).squeeze()
	rec_depths_all.append(rec_depths_argmax)
	model_names.append('argmax')

	## Plot all depth images
	for i, rec_depth in enumerate(rec_depths_all):
		plt.clf()
		plot_depth_img(rec_depths_all[i], vmin, vmax, model_names[i], out_dirpath, scene_id)
		plt.pause(0.1)



