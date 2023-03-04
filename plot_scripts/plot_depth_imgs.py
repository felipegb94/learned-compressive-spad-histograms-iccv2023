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
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_depth_imgs')

	## Choose the test set to plot with
	## Regular test set
	experiment_name_base = 'middlebury/depth_imgs'
	test_set_id = 'test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
	# ## Test set with larger depths than what was trained for
	# experiment_name_base = 'middlebury_largedepth/depth_imgs'
	# test_set_id = 'test_middlebury_largedepth_LargeDepthSimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'

	## output dirpaths
	out_dirpath = os.path.join(io_dirpaths.results_figures_dirpath, experiment_name_base)
	os.makedirs(out_dirpath, exist_ok=True)

	## Scene ids and signal and SBR parameters we want to plot for
	scene_ids = ['spad_Art','spad_Reindeer','spad_Moebius', 'spad_Laundry']
	sbr_params = ['10_1000','10_10','10_50','10_200', '50_500']
	# sbr_params = ['50_500','50_200']

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

	## Generate model_names for the models we want to plots depths for
	model_names = []
	# no_compression_baseline = 'DDFN_C64B10/loss-kldiv_tv-1e-5'
	no_compression_baseline = 'DDFN_C64B10/norm-none/loss-kldiv_tv-1e-05'
	argmax_compression_baseline = 'DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-5'
	model_names.append(no_compression_baseline)
	model_names.append(argmax_compression_baseline)
	

	# ## CSPH3D models: Temporal vs. Spatio-Temporal Compression
	# encoding_type_all = ['csph1d', 'csph1d', 'csph1d', 'separable', 'separable']
	# tdim_init_all = ['TruncFourier', 'HybridGrayFourier', 'Rand', 'Rand', 'Rand']
	# optCt_all = [False, False, True, True, True]
	# optC_all = [False, False, True, True, True]
	# spatial_down_factor_all = [1, 1, 1, 4, 4]
	# num_tdim_blocks_all = [1, 1, 1, 1, 4]
	# compression_ratio_all = [32, 64, 128]

	# ## CSPH3D Models for: Effect of Size of C
	# ## Parameters for: Does decreasing the number of parameters hurt performance?
	# encoding_type_all = ['full', 'separable', 'separable', 'separable']
	# tdim_init_all = ['Rand']*len(encoding_type_all)
	# optCt_all = [True]*len(encoding_type_all)
	# optC_all = [True]*len(encoding_type_all)
	# spatial_down_factor_all = [4]*len(encoding_type_all)
	# num_tdim_blocks_all = [1, 1, 4, 16]
	# compression_ratio_all = [32, 64, 128]
 
	# ## CSPH3D Models for: Effect of Size of C (supplement)
	# ## Parameters for: Does decreasing the number of parameters hurt performance?
	# encoding_type_all = ['full', 'full', 'separable', 'separable', 'separable', 'separable']
	# tdim_init_all = ['Rand']*len(encoding_type_all)
	# optCt_all = [True]*len(encoding_type_all)
	# optC_all = [True]*len(encoding_type_all)
	# spatial_down_factor_all = [4]*len(encoding_type_all)
	# num_tdim_blocks_all = [1, 4, 1, 4, 16, 64]
	# compression_ratio_all = [32, 64, 128]

	# ## Parameters for: Spatial kernel size effect?
	# num_tdim_blocks = 4
	# encoding_type_all = ['csph1d', 'separable', 'separable',  'separable']
	# spatial_down_factor_all = [1, 2, 4, 8]
	# tdim_init_all = ['Rand']*len(encoding_type_all)
	# optCt_all = [True]*len(encoding_type_all)
	# optC_all = [True]*len(encoding_type_all)
	# num_tdim_blocks_all = [num_tdim_blocks]*len(encoding_type_all)
	# compression_ratio_all = [32, 64, 128]

	## Parameters for: Fourier vs. Learned tdim
	experiment_id = 'learned_vs_fourier_tdim'
	encoding_type_all = ['separable', 'separable', 'separable', 'separable', 'csph1d', 'csph1d', 'csph1d', 'csph1d']
	spatial_down_factor_all = [4, 4, 2, 2, 1, 1, 1, 1]
	tdim_init_all = ['Rand', 'TruncFourier', 'Rand', 'TruncFourier', 'Rand', 'TruncFourier', 'TruncFourier', 'HybridGrayFourier'] 
	optCt_all = [True, False, True, False, True, False, False, False]
	optC_all = [True]*len(encoding_type_all)
	optC_all[-1] = False
	optC_all[-2] = False
	optC_all[-3] = False
	num_tdim_blocks_all = [4, 4, 4, 4, 4, 4, 1, 1]
	compression_ratio_all = [32, 64, 128]


	# ## CSPH3D models: Importance of learned coding
	# encoding_type_all = ['csph1d', 'csph1d', 'csph1d', 'csph1d', 'full', 'full']
	# tdim_init_all = ['TruncFourier', 'HybridGrayFourier', 'Rand', 'Rand', 'Rand', 'Rand']
	# optCt_all = [False, False, False, True, True, False]
	# optC_all = [False, False, False, True, True, False]
	# spatial_down_factor_all = [1, 1, 1, 1, 4, 4]
	# num_tdim_blocks_all = [1, 1, 1, 1, 1, 1]
	# compression_ratio_all = [32, 64, 128]

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

	## For each scene and sbr param load depth image
	## We need to do one by one so that rec_depths contain the correct scene + SBR params
	for scene_id in scene_ids:
		for sbr_param in sbr_params:
			scene_fname = '{}_{}'.format(scene_id, sbr_param)
			spad_dataset_sample = spad_dataset.get_item_by_scene_name(scene_fname)
			# plot intensity
			intensity_img = spad_dataset_sample['intensity'].squeeze().cpu().numpy()
			plt.clf()
			plt.imshow(intensity_img, cmap='gray')
			plot_utils.remove_ticks()
			plot_utils.save_currfig(dirpath=os.path.join(out_dirpath, scene_fname), filename='intensity_img', file_ext='svg')
			plt.pause(0.05)
			# init dict 
			model_metrics_all = init_model_metrics_dict(model_names, model_dirpaths)
			# only process a single scene
			model_metrics_all, rec_depths_all = process_middlebury_test_results([scene_id], [sbr_param], model_metrics_all, spad_dataset, return_rec_depths=True)
			# get gt depth to calculate the colorbar range
			gt_depths = rec_depths_all['gt']['depths']
			min_scene_depth = gt_depths.flatten().min() - 0.5*gt_depths.flatten().std()
			max_scene_depth = gt_depths.flatten().max() + 0.5*gt_depths.flatten().std()
			# for each model plot depths
			for model_id in rec_depths_all.keys():
				curr_model = rec_depths_all[model_id]
				model_depths = curr_model['depths']
				plt.clf()
				plot_depth_img(model_depths, min_scene_depth, max_scene_depth, model_id, out_dirpath, scene_fname)
				# plt.imshow(model_depths, vmin=min_scene_depth, vmax=max_scene_depth)
				# plot_utils.remove_ticks()
				# out_fname = model_id.replace('/','_')
				# plot_utils.save_currfig(dirpath=os.path.join(out_dirpath, scene_fname), filename=out_fname, file_ext='svg')
				# plt.pause(0.05)
				# # save with colorbar
				# plt.colorbar()
				# plot_utils.save_currfig(dirpath=os.path.join(out_dirpath, scene_fname + '/withcbar'), filename=out_fname, file_ext='svg')
				# plt.pause(0.05)
				# plt.title(model_id)

			# #####################################################################
			# # plot point cloud
			# ######################################################################
			# import open3d as o3d
			# scale_xy = 1; scale_z = 2
			# min_scene_depth_index = int(min_scene_depth / delta_depth)
			# max_scene_depth_index = int(max_scene_depth / delta_depth)
			# scene_depth_range_index = max_scene_depth_index - min_scene_depth_index
			# min_z_index = int(min_scene_depth_index)
			# max_z_index = int(max_scene_depth_index)
			# z_index_range = max_z_index - min_z_index

			# spad_hist_img = np.moveaxis(spad_dataset_sample['spad'].squeeze().cpu().numpy(), 0, -1)
			# spad_hist_img_zoomed = spad_hist_img[:,:,min_z_index:max_z_index] 
			# plt.figure()
			# plt.imshow(spad_hist_img_zoomed.sum(axis=-1))
			# plt.pause(0.05)
			# X, Y, Z = np.meshgrid(np.arange(nc)/(nc), np.arange(nr)/(nr), np.arange(z_index_range)/(z_index_range))
			# (x, y, z) = (X.flatten(), Y.flatten(), Z.flatten())
			# idx = (spad_hist_img_zoomed > 0)
			# idx = idx.flatten()
			# photon_counts = spad_hist_img_zoomed.flatten()
			# photon_counts = np.log(1+photon_counts)
			# norm_photon_counts = 0.1 + photon_counts / (photon_counts.max() + 1e-6)

			# xyz = np.concatenate((x[idx,np.newaxis],y[idx,np.newaxis],z[idx,np.newaxis]), axis=-1)
			# xyz_colors = np.concatenate((norm_photon_counts[idx,np.newaxis], norm_photon_counts[idx,np.newaxis], norm_photon_counts[idx,np.newaxis]), axis=-1)
			# # xyz_colors = 0.8*np.ones_like(xyz) 

			# pcd = o3d.geometry.PointCloud()
			# pcd.points = o3d.utility.Vector3dVector(xyz)
			# pcd.colors = o3d.utility.Vector3dVector(xyz_colors)

			# os.makedirs('./tmp', exist_ok=True)
			# o3d.io.write_point_cloud("./tmp/tmp_pc.ply", pcd)

			# # Load saved point cloud and visualize it
			# pcd_load = o3d.io.read_point_cloud("./tmp/tmp_pc.ply")


			# points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
			# 		[0, 1, 1], [1, 1, 1]]
			# lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
			# 		[0, 4], [1, 5], [2, 6], [3, 7]]
			# colors = [[1, 0, 0] for i in range(len(lines))]
			# line_set = o3d.geometry.LineSet()
			# line_set.points = o3d.utility.Vector3dVector(points)
			# line_set.lines = o3d.utility.Vector2iVector(lines)
			# line_set.colors = o3d.utility.Vector3dVector(colors)

			# vis = o3d.visualization.Visualizer()
			# vis.create_window()
			# vis.add_geometry(pcd_load)
			# vis.add_geometry(line_set)
			# opt = vis.get_render_option()
			# opt.background_color = np.asarray([0, 0, 0])
			# view_control = vis.get_view_control()
			# view_control.set_front(np.array([0.38,-0.20,-0.98]).reshape((3,1)))
			# view_control.set_up(np.array([-0.062,-0.977,0.199]).reshape((3,1)))
			# view_control.set_lookat(np.array([0.494,0.493,0.491]).reshape((3,1)))
			# view_control.set_zoom(1.0)
			# vis.run()
			# vis.destroy_window()

			# o3d.visualization.draw_geometries([pcd_load]
			# 		, width=1280, height=960
			# 		, left=10, top=10
			# 		, zoom=1.1
			# 		, front=np.array([0.38,-0.20,-0.98]).reshape((3,1))
			# 		, up=np.array([-0.062,-0.977,0.199]).reshape((3,1))
			# 		, lookat=np.array([0.494,0.493,0.491]).reshape((3,1))
			# 		)






