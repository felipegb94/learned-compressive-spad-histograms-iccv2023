#### Standard Library Imports
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import hydra
from omegaconf import OmegaConf
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from tof_utils import *
from research_utils import io_ops, plot_utils, np_utils
from spad_dataset import SpadDataset

middlebury_test_set_info = {
	'scene_ids': ['spad_Art', 'spad_Reindeer', 'spad_Books', 'spad_Moebius', 'spad_Bowling1', 'spad_Dolls', 'spad_Laundry', 'spad_Plastic']
	, 'sbr_params_low_flux': ['2_2','2_10','2_50','5_2','5_10','5_50','10_2','10_10','10_50']
	, 'sbr_params_high_flux': ['10_200', '10_500', '10_1000', '50_50', '50_200', '50_500', '50_1000']    
	, 'sbr_params_high_signal': ['200_500','200_2000','200_5000','200_10000','200_20000']
    , 'sbr_params_low_sbr': ['1_100','2_100','3_100','2_50','10_500','10_1000','50_5000','100_5000','100_10000','100_20000']    
}

def denorm_bins(bins, num_bins):
	return bins*(num_bins)

def compute_mse(gt, est):
	mse = np.mean((gt - est)**2)
	return mse

def compute_mae(gt, est):
	mae = np.mean(np.abs(gt - est))
	return mae

def compute_rmse(gt, est):
	rmse = np.sqrt(compute_mse(gt,est))
	return rmse

def compute_eps_tol_err(gt, est, eps_tol_thresh):
	'''
		Compute epsilon tolerance error. Percentage of points with errors lower than a threshold
	'''
	abs_errs = np.abs(gt-est)
	n = abs_errs.size
	eps_tolerance = float(np.sum(abs_errs < eps_tol_thresh)) / n
	assert(np.all(eps_tolerance <= 1.)), "eps tolerance should be <= 1"
	return eps_tolerance

def compute_error_metrics(gt, est, eps_tol_thresh=0.01):
	metrics = {}
	metrics['abs_errs'] = np.abs(gt-est)
	metrics['perc_errs'] = np.round(np_utils.calc_mean_percentile_errors(metrics['abs_errs'])[0], decimals=2)
	metrics['mae'] =  compute_mae(gt,est)
	metrics['mse'] =  compute_mse(gt,est)
	metrics['rmse'] =  compute_rmse(gt,est)
	metrics['1mm_tol_err'] =  compute_eps_tol_err(gt,est,eps_tol_thresh=1./1000.)
	metrics['5mm_tol_err'] =  compute_eps_tol_err(gt,est,eps_tol_thresh=5./1000.)
	metrics['10mm_tol_err'] =  compute_eps_tol_err(gt,est,eps_tol_thresh=10./1000.)
	metrics['20mm_tol_err'] =  compute_eps_tol_err(gt,est,eps_tol_thresh=20./1000.)
	return metrics
	# return (compute_rmse(gt,est), compute_mse(gt,est), compute_mae(gt,est), abs_errs, np.round(perc_errs[0], decimals=2))

def get_model_depths(model_result_dirpath, scene_fname, num_bins, tau):
	model_result_fpath = os.path.join(model_result_dirpath, scene_fname+'.npz')
	model_result = np.load(model_result_fpath)
	model_bins = denorm_bins(model_result['dep_re'], num_bins=num_bins).squeeze()
	model_depths = bin2depth(model_bins, num_bins=num_bins, tau=tau)
	return model_depths

def calc_mean_rmse(metric_dict):
	return np.mean(metric_dict['rmse'])

def calc_mean_mse(metric_dict):
	return np.mean(metric_dict['mse'])*100

def calc_mean_mae(metric_dict):
	return np.mean(metric_dict['mae'])*100

def calc_compression_from_ID(id, nt=1024):
	import re
	split_id = id.split('_')
	assert(len(split_id) > 3), "invalid id"
	k = int(split_id[1].split('k')[1])
	block_dims_str = re.findall(r'\d+', split_id[2])
	block_dims = [int(b) for b in block_dims_str]
	br = block_dims[0]
	bc = block_dims[1]
	bt = block_dims[2]
	compression_ratio = (br*bc*bt) / k
	return compression_ratio

def csph3d_compression2k(compression_ratio, block_nr, block_nc, block_nt):
	block_size = int(block_nr*block_nc*block_nt)
	assert((block_size % compression_ratio) == 0), "block_size should be divisible by compression_ratio"
	k = int(block_size / compression_ratio)
	return k

def csph3d_k2compression(k, block_nr, block_nc, block_nt):
	block_size = int(block_nr*block_nc*block_nt)
	assert((block_size % compression_ratio) == 0), "block_size should be divisible by compression_ratio"
	compression_ratio = int(block_size / k)
	return compression_ratio

def get_model_dirpaths(model_names):
	'''
		Given the model name get the dirpath containing all the results for that model. Usually, each model that was trained has a unique ID that is appended to the model_name to generate the dirpath, so in order to not have to keep track of these IDs we simply store them inside a dict whenever we test that model.
	'''
	## Get pretrained models dirpaths
	pretrained_models_all = io_ops.load_json('pretrained_models_rel_dirpaths.json')
	model_dirpaths = []
	for model_name in model_names:
		model_dirpaths.append(pretrained_models_all[model_name]['rel_dirpath'])

def append_model_metrics(model_metrics, test_set_id, scene_fname, gt_depths, num_bins, tau, model_depths=None):
	# get model depths if they are not provided
	if(model_depths is None):
		## Make sure dirpath with test set results exists
		model_dirpath = model_metrics['dirpath']
		model_result_dirpath = os.path.join(model_dirpath, test_set_id) 
		assert(os.path.exists(model_result_dirpath)), "{} path does not exist.".format(model_result_dirpath)
		## Get depths for model depths for current scene
		(model_depths) = get_model_depths(model_result_dirpath=model_result_dirpath, scene_fname=scene_fname, num_bins=num_bins, tau=tau)
	## init metrics data structure if they don't exist
	if(not ('rmse' in model_metrics.keys())): model_metrics['rmse'] = []
	if(not ('mae' in model_metrics.keys())): model_metrics['mae'] = []
	if(not ('mse' in model_metrics.keys())): model_metrics['mse'] = []
	if(not ('1mm_tol_err' in model_metrics.keys())): model_metrics['1mm_tol_err'] = []
	if(not ('5mm_tol_err' in model_metrics.keys())): model_metrics['5mm_tol_err'] = []
	if(not ('10mm_tol_err' in model_metrics.keys())): model_metrics['10mm_tol_err'] = []
	if(not ('20mm_tol_err' in model_metrics.keys())): model_metrics['20mm_tol_err'] = []
	## Compute error metrics with respect to ground truth
	scene_metrics = compute_error_metrics(gt_depths, model_depths)
	# (scene_rmse, scene_mse, scene_mae, scene_abs_errs, scene_perc_errs) = compute_error_metrics(gt_depths, model_depths)
	model_metrics['rmse'].append(scene_metrics['rmse'])
	model_metrics['mse'].append(scene_metrics['mse'])
	model_metrics['mae'].append(scene_metrics['mae'])
	model_metrics['1mm_tol_err'].append(scene_metrics['1mm_tol_err'])
	model_metrics['5mm_tol_err'].append(scene_metrics['5mm_tol_err'])
	model_metrics['10mm_tol_err'].append(scene_metrics['10mm_tol_err'])
	model_metrics['20mm_tol_err'].append(scene_metrics['20mm_tol_err'])
	# scene_metrics = {}
	# scene_metrics['rmse'] = scene_metrics['rmse']
	# scene_metrics['mse'] = scene_metrics['mse']
	# scene_metrics['mae'] = scene_metrics['mae']
	return model_metrics, model_depths, scene_metrics

def get_hydra_io_dirpaths(job_name='tmp'):
	'''
		Loads the io_dirpaths.conf without using hydra.main
		It is quite a round about way to load .conf and use the variable interpolation. Quite annoying. It could be one reason to not use hydra in the future and just use OmegaConf
		Solution from here: https://github.com/facebookresearch/hydra/issues/1786#issuecomment-913110496
	'''
	## load all dirpaths without creating a job 
	## Initialize hydra and resolve variables
	hydra.core.global_hydra.GlobalHydra.instance().clear() ## needed when running and re-runnig on ipython or jupyer
	hydra.initialize(config_path="./conf", job_name=job_name)
	cfg = hydra.compose(config_name="io_dirpaths",return_hydra_config=True)
	hydra.core.hydra_config.HydraConfig().cfg = cfg ## This line is required to allow omegaconf to interpolate
	OmegaConf.resolve(cfg)
	return cfg.io_dirpaths

def init_model_metrics_dict(model_names, model_dirpaths):
	##
	model_metrics_all = {}
	## initialize for gt and raw data baselines
	model_metrics_all['gt'] = {}
	model_metrics_all['gt']['dirpath'] = None
	model_metrics_all['lmf'] = {}
	model_metrics_all['lmf']['dirpath'] = None
	model_metrics_all['argmax'] = {}
	model_metrics_all['argmax']['dirpath'] = None

	## for each model name create one entry 
	for i, model_name in enumerate(model_names):
		model_metrics_all[model_name] = {}
		model_metrics_all[model_name]['dirpath'] = model_dirpaths[i]
	
	return model_metrics_all

def parse_sbr_params(sbr_params):
	mean_signal_photons = int(sbr_params.split('_')[0])
	mean_bkg_photons = int(sbr_params.split('_')[1])
	mean_sbr = float(mean_signal_photons) / float(mean_bkg_photons)
	return (mean_sbr, mean_signal_photons, mean_bkg_photons)

def parse_sbr_params_list(sbr_params_all):
	mean_signal_photons_all = []
	mean_bkg_photons_all = []
	mean_sbr_all = []
	for sbr_params in sbr_params_all:
		(mean_sbr, mean_signal_photons, mean_bkg_photons) = parse_sbr_params(sbr_params)
		mean_signal_photons_all.append(mean_signal_photons)
		mean_bkg_photons_all.append(mean_bkg_photons)
		mean_sbr_all.append(mean_sbr)
	return (mean_sbr_all, mean_signal_photons_all, mean_bkg_photons_all)

def process_middlebury_test_results(scene_ids, sbr_params, model_metrics_all, spad_dataset, out_dirpath=None, save_depth_images=False, return_rec_depths=False):
	## For each scene id and sbr param, load model rec_depths and compute error metrics. If save_depth images is true, save them, and if return_rec_depths is True store the rec depths and return them
	if(not ('sbr_params' in model_metrics_all.keys())): 
		model_metrics_all['sbr_params'] = {}
		model_metrics_all['sbr_params']['mean_sbr'] = []
		model_metrics_all['sbr_params']['mean_signal_photons'] = []
		model_metrics_all['sbr_params']['mean_bkg_photons'] = []
	for i in range(len(scene_ids)):
		for j in range(len(sbr_params)):
			curr_scene_id = scene_ids[i] 
			curr_sbr_params = sbr_params[j] 
			scene_fname = '{}_{}'.format(curr_scene_id, curr_sbr_params)
			# print("Processing: {}".format(scene_fname))

			(mean_sbr, mean_signal_photons, mean_bkg_photons) = parse_sbr_params(curr_sbr_params)
			model_metrics_all['sbr_params']['mean_sbr'].append(mean_sbr)
			model_metrics_all['sbr_params']['mean_signal_photons'].append(mean_signal_photons)
			model_metrics_all['sbr_params']['mean_bkg_photons'].append(mean_bkg_photons)

			## get scene from spad_dataset
			scene_data = spad_dataset.get_item_by_scene_name(scene_fname)

			## Load params
			(num_bins, nr, nc) = scene_data['rates'].squeeze().shape
			tres = spad_dataset.tres_ps*1e-12
			intensity = scene_data['intensity'].squeeze().cpu().numpy()
			SBR = scene_data['SBR']
			mean_background_photons = scene_data['mean_background_photons']
			mean_signal_photons = scene_data['mean_signal_photons']
			tau = num_bins*tres

			## load gt and baseline depths
			gt_norm_bins = scene_data['bins'].squeeze().cpu().numpy()
			lmf_norm_bins = scene_data['est_bins_lmf'].squeeze().cpu().numpy() 
			argmax_norm_bins = scene_data['est_bins_argmax'].squeeze().cpu().numpy() 
			gt_bins = gt_norm_bins*num_bins
			lmf_bins = lmf_norm_bins*num_bins
			argmax_bins = argmax_norm_bins*num_bins

			## Get depths for MATLAB data
			gt_depths = bin2depth(gt_bins, num_bins=num_bins, tau=tau)
			lmf_depths = bin2depth(lmf_bins, num_bins=num_bins, tau=tau)
			argmax_depths = bin2depth(argmax_bins, num_bins=num_bins, tau=tau)

			## Compute and store metrics for gt and basic baselines that are stored in .mat file instead of .npz files
			(gt_metrics, gt_depths, gt_scene_metrics) = append_model_metrics(model_metrics_all['gt'], spad_dataset.test_set_id, scene_fname, gt_depths, num_bins, tau, model_depths=gt_depths)
			(lmf_metrics, lmf_depths, lmf_scene_metrics) = append_model_metrics(model_metrics_all['lmf'], spad_dataset.test_set_id, scene_fname, gt_depths, num_bins, tau, model_depths=lmf_depths)
			(argmax_metrics, argmax_depths, argmax_scene_metrics) = append_model_metrics(model_metrics_all['argmax'], spad_dataset.test_set_id, scene_fname, gt_depths, num_bins, tau, model_depths=argmax_depths)

			## For visualization purposes
			min_depth = gt_depths.flatten().min() - 0.5*gt_depths.flatten().std() 
			max_depth = gt_depths.flatten().max() + 0.5*gt_depths.flatten().std()
			min_err = 0
			max_err = 0.15

			## Compute and store metrics for all models
			rec_depths = {}
			if(return_rec_depths):
				rec_depths['gt'] = {}
				rec_depths['gt']['depths'] = gt_depths
				rec_depths['gt']['metrics'] = gt_scene_metrics				
				rec_depths['lmf'] = {}
				rec_depths['lmf']['depths'] = lmf_depths
				rec_depths['lmf']['metrics'] = lmf_scene_metrics				
				rec_depths['argmax'] = {}
				rec_depths['argmax']['depths'] = argmax_depths
				rec_depths['argmax']['metrics'] = argmax_scene_metrics				

			for model_id in model_metrics_all.keys():
				curr_model_metrics = model_metrics_all[model_id]
				# skip if there is no dirpath --> i.e., gt, lmf, and argmax
				if(model_id == 'sbr_params'): continue
				if(curr_model_metrics['dirpath'] is None): continue
				(curr_model_metrics, model_depths, scene_metrics) = append_model_metrics(curr_model_metrics, spad_dataset.test_set_id, scene_fname, gt_depths, num_bins, tau)
				# if we want to plot the depths for the current scene keep track of the recovered depths
				if(return_rec_depths):
					rec_depths[model_id] = {}
					rec_depths[model_id]['depths'] = model_depths
					rec_depths[model_id]['metrics'] = scene_metrics
				# save depth images
				if(save_depth_images):
					if(out_dirpath is None):
						print("WARNING: Can't save depth images if no out_dirpath is given")
					else:
						if(os.path.exists(out_dirpath)):
							plt.clf()
							plt.imshow(model_depths, vmin=min_depth, vmax=max_depth)
							plt.title(model_id)
							plot_utils.remove_ticks()
							plot_utils.save_currfig_png(dirpath=os.path.join(out_dirpath, scene_fname), filename=model_id)
							plt.pause(0.05)
						else:
							print("WARNING: Can't save depth images if the out_dirpath does not exist")
	# ## Convert all lists into numpy arrray
	# for model_name in model_metrics_all.keys():
	# 	for metric_name in model_metrics_all[model_name].keys():
	# 		if(metric_name == 'dirpath'): continue
	# 		else: model_metrics_all[model_name][metric_name] = np.array(model_metrics_all[model_name][metric_name])

	return (model_metrics_all, rec_depths)

if __name__=='__main__':

	## get io dirpaths from hydra 
	## Initialize hydra and resolve variables
	io_dirpaths = get_hydra_io_dirpaths(job_name='analyze_test_results')

	# experiment_name = ''
	experiment_name = 'test_results/d2d2D_B12_tv_comparisson'
	experiment_name = 'test_results/phasor2depth_comparisson_v1'
	experiment_name = 'test_results/db3D_d2d_comparissons'
	experiment_name = 'test_results/db3D_csph1Dk16_comparissons'
	experiment_name = 'test_results/csphseparable_vs_csph1D2D_comparissons'
	experiment_name = 'test_results/64xcompression_csphseparable_vs_csph1D2D_comparissons'
	experiment_name = 'test_results/128xcompression_csphseparable_comparissons'
	experiment_name = 'test_results/64xcompression_csphseparable_comparissons'
	experiment_name = 'test_results/temporal_down_ablation'
	experiment_name = 'test_results/csph3D_results'
	experiment_name = 'test_results/csph3D_full_vs_separable'
	experiment_name = 'test_results/csph3D_good_norm'

	out_dirpath = os.path.join(io_dirpaths.results_weekly_dirpath, experiment_name)
	os.makedirs(out_dirpath, exist_ok=True)

	plot_results = False
	plot_compression_vs_perf = False
	save_depth_images = False

	## Scene ID and Params
	test_set_id = 'test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
	scene_ids = middlebury_test_set_info['scene_ids']
	sbr_params = middlebury_test_set_info['sbr_params_low_flux']

	scene_ids = ['spad_Art']
	# # # scene_ids = ['spad_Books']
	# # scene_ids = ['spad_Books']
	scene_ids = ['spad_Reindeer']
	# scene_ids = ['spad_Plastic']
	# sbr_params = ['2_10']
	# sbr_params = ['10_1000']
	sbr_params = ['50_50']
	sbr_params = ['10_2']

	compression_lvl = '32x' 

	if(save_depth_images and ((len(scene_ids) > 1) or (len(sbr_params) > 1))):
		print("WARNING: save_depth_images with many scene ids and sbr params will generate a lot of images. type continue to keep going or exit and edit parameters")
		breakpoint()

	## Create spad dataloader object to get ground truth data for each scene
	spad_dataset = SpadDataset(datalist_fpath=os.path.join(io_dirpaths.datalists_dirpath, test_set_id+'.txt'), disable_rand_crop=True)

	## Load pre-trained models
	pretrained_models_all = io_ops.load_json('pretrained_models_rel_dirpaths.json')

	## Ground truth and basic baselines (no need to specifi dirpath for these ones since they are loaded from the gt data file)
	model_metrics_all = {}
	# metrics_all = {}
	gt_data_dirpath = 'data_gener/TestData/middlebury/processed/SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
	model_metrics_all['gt'] = {}
	model_metrics_all['gt']['dirpath'] = None
	model_metrics_all['lmf'] = {}
	model_metrics_all['lmf']['dirpath'] = None
	model_metrics_all['argmax'] = {}
	model_metrics_all['argmax']['dirpath'] = None

	## Non CSPH Baselines
	model_metrics_all['db3D_tv'] = {}
	model_metrics_all['db3D_tv']['dirpath'] = pretrained_models_all['DDFN_C64B10/loss-kldiv_tv-1e-5']['rel_dirpath']
	# model_metrics_all['db3D_nl_tv'] = {}
	# model_metrics_all['db3D_nl_tv']['dirpath'] = pretrained_models_all['DDFN_C64B10_NL_original/loss-kldiv_tv-1e-5']['rel_dirpath']
	# model_metrics_all['db3D_nl_d2d_tv'] = {}
	# model_metrics_all['db3D_nl_d2d_tv']['dirpath'] = pretrained_models_all['DDFN_C64B10_NL_Depth2Depth/loss-kldiv_tv-1e-5']['rel_dirpath']
	# model_metrics_all['db3D_d2d'] = {}
	# model_metrics_all['db3D_d2d']['dirpath'] = pretrained_models_all['DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-5']['rel_dirpath']
	# model_metrics_all['db3D_d2d_tv0'] = {}
	# model_metrics_all['db3D_d2d_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_Depth2Depth/loss-kldiv_tv-0.0']['rel_dirpath']

	## CSPH1D Baseline Models
	# model_metrics_all['db3D_csph1Dk8_gray_tv0'] = {}
	# model_metrics_all['db3D_csph1Dk8_gray_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH1D/k8_HybridFourierGray/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph1Dk8_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csph1Dk8_grayfour_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH1D/k8_HybridGrayFourier/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph1Dk16_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csph1Dk16_grayfour_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH1D/k16_HybridGrayFourier/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph1Dk16_truncfour_tv0'] = {}
	# model_metrics_all['db3D_csph1Dk16_truncfour_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH1D/k16_TruncFourier/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph1Dk16_coarsehist_tv0'] = {}
	# model_metrics_all['db3D_csph1Dk16_coarsehist_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH1D/k16_CoarseHist/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph1Dk32_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csph1Dk32_grayfour_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH1D/k32_HybridGrayFourier/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph1Dk32_truncfour_tv0'] = {}
	# model_metrics_all['db3D_csph1Dk32_truncfour_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH1D/k32_TruncFourier/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph1Dk64_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csph1Dk64_grayfour_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH1D/k64_HybridGrayFourier/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph1Dk64_truncfour_tv0'] = {}
	# model_metrics_all['db3D_csph1Dk64_truncfour_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH1D/k64_TruncFourier/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph1Dk64_coarsehist_tv0'] = {}
	# model_metrics_all['db3D_csph1Dk64_coarsehist_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH1D/k64_CoarseHist/loss-kldiv_tv-0.0']['rel_dirpath']

	## CSPH3D Models with Unfiltered Backprojection Ups 
	# model_metrics_all['db3D_csph3Dk512_separable4x4x1024_opt_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk512_separable4x4x1024_opt_grayfour_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k512_down4_Mt1_HybridGrayFourier-optCt=True-optC=True_separable/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk256_separable4x4x1024_opt_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk256_separable4x4x1024_opt_grayfour_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k256_down4_Mt1_HybridGrayFourier-optCt=True-optC=True_separable/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk128_separable4x4x1024_opt_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk128_separable4x4x1024_opt_grayfour_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k128_down4_Mt1_HybridGrayFourier-optCt=True-optC=True_separable/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk64_separable4x4x1024_opt_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk64_separable4x4x1024_opt_grayfour_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k64_down4_Mt1_HybridGrayFourier-optCt=True-optC=True_separable/loss-kldiv_tv-0.0']['rel_dirpath']

	# model_metrics_all['db3D_csph3Dk512_separable4x4x1024_opt_rand_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk512_separable4x4x1024_opt_rand_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k512_down4_Mt1_Rand-optCt=True-optC=True_separable/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk256_separable4x4x1024_opt_rand_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk256_separable4x4x1024_opt_rand_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k256_down4_Mt1_Rand-optCt=True-optC=True_separable/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk128_separable4x4x1024_opt_rand_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk128_separable4x4x1024_opt_rand_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k128_down4_Mt1_Rand-optCt=True-optC=True_separable/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk64_separable4x4x1024_opt_rand_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk64_separable4x4x1024_opt_rand_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k64_down4_Mt1_Rand-optCt=True-optC=True_separable/loss-kldiv_tv-0.0']['rel_dirpath']

	# model_metrics_all['db3D_csph3Dk256_full4x4x1024_opt_rand_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk256_full4x4x1024_opt_rand_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k256_down4_Mt1_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk128_full4x4x1024_opt_rand_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk128_full4x4x1024_opt_rand_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k128_down4_Mt1_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk64_full4x4x1024_opt_rand_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk64_full4x4x1024_opt_rand_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k64_down4_Mt1_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk8_full4x4x128_opt_rand_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk8_full4x4x128_opt_rand_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k8_down4_Mt8_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk16_full4x4x128_opt_rand_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk16_full4x4x128_opt_rand_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k16_down4_Mt8_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk16_full4x4x64_opt_rand_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk16_full4x4x64_opt_rand_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k16_down4_Mt16_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk32_full4x4x64_opt_rand_tv0'] = {}
	# model_metrics_all['db3D_csph3Dk32_full4x4x64_opt_rand_tv0']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv1/k32_down4_Mt16_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0']['rel_dirpath']

	# ## CSPH3D models with unfilt. backproj && normalization
	# model_metrics_all['db3D_csph3Dk128_full4x4x1024_opt_rand_tv0_norm-L2'] = {}
	# model_metrics_all['db3D_csph3Dk128_full4x4x1024_opt_rand_tv0_norm-L2']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv2/k128_down4_Mt1_Rand-optCt=True-optC=True_full_norm-L2/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk128_full4x4x1024_opt_rand_tv0_norm-Linf'] = {}
	# model_metrics_all['db3D_csph3Dk128_full4x4x1024_opt_rand_tv0_norm-Linf']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv2/k128_down4_Mt1_Rand-optCt=True-optC=True_full_norm-Linf/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk512_full4x4x1024_opt_rand_tv0_norm-none'] = {}
	# model_metrics_all['db3D_csph3Dk512_full4x4x1024_opt_rand_tv0_norm-none']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk512_full4x4x1024_opt_rand_tv0_norm-L2'] = {}
	# model_metrics_all['db3D_csph3Dk512_full4x4x1024_opt_rand_tv0_norm-L2']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-L2/loss-kldiv_tv-0.0']['rel_dirpath']
	# model_metrics_all['db3D_csph3Dk512_full4x4x1024_opt_rand_tv0_norm-Linf'] = {}
	# model_metrics_all['db3D_csph3Dk512_full4x4x1024_opt_rand_tv0_norm-Linf']['dirpath'] = pretrained_models_all['DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-Linf/loss-kldiv_tv-0.0']['rel_dirpath']

	# ## [OLD] CSPH models with learned ups --> Are in backup drive
	# model_metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/temporal_down_ablation/DDFN_C64B10_CSPH/k64_down4_Mt1_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-24_094328/'
	# model_metrics_all['db3D_csphk32_separable4x4x512_opt_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csphk32_separable4x4x512_opt_grayfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/temporal_down_ablation/DDFN_C64B10_CSPH/k32_down4_Mt2_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-26_172638/'
	# model_metrics_all['db3D_csphk16_separable4x4x256_opt_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csphk16_separable4x4x256_opt_grayfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/temporal_down_ablation/DDFN_C64B10_CSPH/k16_down4_Mt4_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-29_190259/'
	# model_metrics_all['db3D_csphk8_separable4x4x128_opt_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csphk8_separable4x4x128_opt_grayfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/temporal_down_ablation/DDFN_C64B10_CSPH/k8_down4_Mt8_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-07-02_023357/'
	# model_metrics_all['db3D_csphk64_separable2x2x1024_opt_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csphk64_separable2x2x1024_opt_grayfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH/k64_down2_Mt1_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-18_183013/'
	# model_metrics_all['db3D_csphk512_separable4x4x1024_opt_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csphk512_separable4x4x1024_opt_grayfour_tv0']['dirpath'] = 'outputs_fmu/nyuv2_64x64x1024_80ps/compression_vs_perf/DDFN_C64B10_CSPH/k512_down4_Mt1_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-07-16_113012/'
	# model_metrics_all['db3D_csphk512_separable4x4x1024_opt_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csphk512_separable4x4x1024_opt_grayfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/compression_vs_perf/DDFN_C64B10_CSPH/k512_down4_Mt1_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-07-16_113012/'
	# model_metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0'] = {}
	# model_metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH/k256_down4_Mt1_TruncFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-22_130057/'
	# model_metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0'] = {}
	# model_metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH/k128_down4_Mt1_TruncFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-18_203447/'
	# model_metrics_all['db3D_csphk64_separable4x4x512_opt_truncfour_tv0'] = {}
	# model_metrics_all['db3D_csphk64_separable4x4x512_opt_truncfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH/k64_down4_Mt2_TruncFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-20_111348/'
	# model_metrics_all['db3D_csphk32_separable2x2x1024_opt_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csphk32_separable2x2x1024_opt_grayfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH/k32_down2_Mt1_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-21_230853/'

	## OLD CSPH models with ZNCC + Bilinear ups
	# model_metrics_all['db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0'] = {}
	# model_metrics_all['db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1DGlobal2DLocal4xDown/k128_down4_TruncFourier/loss-kldiv_tv-0.0/2022-06-06_131240/'
	# model_metrics_all['db3D_csph1D2Dk128down2upzncc_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csph1D2Dk128down2upzncc_grayfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D2D/k128_down2_HybridGrayFourier/loss-kldiv_tv-0.0/2022-05-25_182959_zncc/'
	# model_metrics_all['db3D_csph1D2Dk128down2_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csph1D2Dk128down2_grayfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D2D/k128_down2_HybridGrayFourier/loss-kldiv_tv-0.0/2022-06-05_181812/'
	# model_metrics_all['db3D_csph1D2Dk64down2_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csph1D2Dk64down2_grayfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D2D/k64_down2_HybridGrayFourier/loss-kldiv_tv-0.0/2022-05-24_220441/'
	# model_metrics_all['db3D_csph1D2Dk32down2_grayfour_tv0'] = {}
	# model_metrics_all['db3D_csph1D2Dk32down2_grayfour_tv0']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D2D/k32_down2_HybridGrayFourier/loss-kldiv_tv-0.0/2022-06-03_140623/'

	# ## OLD 2D CNN Depth2Depth
	# model_metrics_all['db2D_d2d2hist01Inputs_B12'] = {}
	# model_metrics_all['db2D_d2d2hist01Inputs_B12']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth2Hist_01Inputs/B-12_MS-8/2022-05-02_214727/'
	# model_metrics_all['db2D_p2d_B16_7freqs_tv1m5'] = {}
	# model_metrics_all['db2D_p2d_B16_7freqs_tv1m5']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/test/DDFN2D_Phasor2Depth7Freqs/B-16_MS-8/loss-L1_tv-1e-05/2022-05-05_122615_7freqs/'
	# model_metrics_all['db2D_p2d_B16_allfreqs_tv1m5'] = {}
	# model_metrics_all['db2D_p2d_B16_allfreqs_tv1m5']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/test/DDFN2D_Phasor2Depth/B-16_MS-8/loss-L1_tv-1e-05/2022-05-18_042156/'
	# model_metrics_all['db2D_d2d01Inputs_B12_tv1m3'] = {}
	# model_metrics_all['db2D_d2d01Inputs_B12_tv1m3']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/loss-L1_tv-0.001/2022-05-03_183044/'
	# model_metrics_all['db2D_d2d01Inputs_B12_tv1m4'] = {}
	# model_metrics_all['db2D_d2d01Inputs_B12_tv1m4']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/loss-L1_tv-0.0001/2022-05-03_183002/'
	# model_metrics_all['db2D_d2d01Inputs_B12_tv3m5'] = {}
	# model_metrics_all['db2D_d2d01Inputs_B12_tv3m5']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/loss-L1_tv-3e-05/2022-05-03_185303/'
	# model_metrics_all['db2D_d2d01Inputs_B12_tv1m5'] = {}
	# model_metrics_all['db2D_d2d01Inputs_B12_tv1m5']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/debug/2022-04-27_104532/'
	# model_metrics_all['db2D_d2d01Inputs_B12_tv1m10'] = {}
	# model_metrics_all['db2D_d2d01Inputs_B12_tv1m10']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/loss-L1_tv-1e-10/2022-05-03_183128/'
	# model_metrics_all['db2D_d2d01Inputs_B16_tv1m5'] = {}
	# model_metrics_all['db2D_d2d01Inputs_B16_tv1m5']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-16_MS-8/2022-05-01_172344/'
	# model_metrics_all['db2D_d2d01Inputs_B22_tv1m5'] = {}
	# model_metrics_all['db2D_d2d01Inputs_B22_tv1m5']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-22_MS-8/loss-L1_tv-1e-05/2022-05-20_103921'
	# model_metrics_all['db2D_d2d01Inputs_B24_tv1m5'] = {}
	# model_metrics_all['db2D_d2d01Inputs_B24_tv1m5']['dirpath'] = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-24_MS-8/2022-04-28_163253/'

	for i in range(len(scene_ids)):
		for j in range(len(sbr_params)):
			curr_scene_id = scene_ids[i] 
			curr_sbr_params = sbr_params[j] 
			scene_fname = '{}_{}'.format(curr_scene_id, curr_sbr_params)
			print("Processing: {}".format(scene_fname))

			## get scene from spad_dataset
			scene_data = spad_dataset.get_item_by_scene_name(scene_fname)

			## Load params
			(num_bins, nr, nc) = scene_data['rates'].squeeze().shape
			tres = spad_dataset.tres_ps*1e-12
			intensity = scene_data['intensity'].squeeze().cpu().numpy()
			SBR = scene_data['SBR']
			mean_background_photons = scene_data['mean_background_photons']
			mean_signal_photons = scene_data['mean_signal_photons']
			tau = num_bins*tres

			## load gt and baseline depths
			gt_norm_bins = scene_data['bins'].squeeze().cpu().numpy()
			lmf_norm_bins = scene_data['est_bins_lmf'].squeeze().cpu().numpy() 
			argmax_norm_bins = scene_data['est_bins_argmax'].squeeze().cpu().numpy() 
			gt_bins = gt_norm_bins*num_bins
			lmf_bins = lmf_norm_bins*num_bins
			argmax_bins = argmax_norm_bins*num_bins

			## Get depths for MATLAB data
			gt_depths = bin2depth(gt_bins, num_bins=num_bins, tau=tau)
			lmf_depths = bin2depth(lmf_bins, num_bins=num_bins, tau=tau)
			argmax_depths = bin2depth(argmax_bins, num_bins=num_bins, tau=tau)

			## Compute and store metrics for gt and basic baselines that are stored in .mat file instead of .npz files
			(gt_metrics, gt_depths, gt_scene_metrics) = append_model_metrics(model_metrics_all['gt'], test_set_id, scene_fname, gt_depths, num_bins, tau, model_depths=gt_depths)
			(lmf_metrics, lmf_depths, lmf_scene_metrics) = append_model_metrics(model_metrics_all['lmf'], test_set_id, scene_fname, gt_depths, num_bins, tau, model_depths=lmf_depths)
			(argmax_metrics, argmax_depths, argmax_scene_metrics) = append_model_metrics(model_metrics_all['argmax'], test_set_id, scene_fname, gt_depths, num_bins, tau, model_depths=argmax_depths)

			## For visualization purposes
			min_depth = gt_depths.flatten().min() - 0.5*gt_depths.flatten().std() 
			max_depth = gt_depths.flatten().max() + 0.5*gt_depths.flatten().std()
			min_err = 0
			max_err = 0.15

			## Compute and store metrics for all models
			rec_depths = {}
			for model_id in model_metrics_all.keys():
				curr_model_metrics = model_metrics_all[model_id]
				# skip if there is no dirpath
				if(curr_model_metrics['dirpath'] is None): continue
				(curr_model_metrics, model_depths, scene_metrics) = append_model_metrics(curr_model_metrics, test_set_id, scene_fname, gt_depths, num_bins, tau)
				# if we want to plot the depths for the current scene keep track of the recovered depths
				if(plot_results):
					rec_depths[model_id] = {}
					rec_depths[model_id]['depths'] = model_depths
					rec_depths[model_id]['metrics'] = scene_metrics

				if(save_depth_images):
					plt.clf()
					plt.imshow(model_depths, vmin=min_depth, vmax=max_depth)
					plt.title(model_id)
					plot_utils.remove_ticks()
					plot_utils.save_currfig_png(dirpath=os.path.join(out_dirpath, scene_fname), filename=model_id)
					plt.pause(0.1)

			if(plot_results):
				plt.clf()
				plt.suptitle("{} - SBR: {}, Signal: {} photons, Bkg: {} photons".format(scene_fname, SBR, mean_signal_photons, mean_background_photons), fontsize=20)
				plt.subplot(2,3,1)
				plt.imshow(rec_depths['db3D']['depths'], vmin=min_depth, vmax=max_depth); 
				plt.title('db3D \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D']['metrics']['mse'], rec_depths['db3D']['metrics']['mae']*100),fontsize=14)
				plt.colorbar()
				plt.subplot(2,3,2)
				plt.imshow(rec_depths['db3D_d2d']['depths'], vmin=min_depth, vmax=max_depth); 
				plt.title('db3D_d2d \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D_d2d']['metrics']['mse'], rec_depths['db3D_d2d']['metrics']['mae']*100),fontsize=14)
				plt.colorbar()
				plt.subplot(2,3,3)
				plt.imshow(rec_depths['db3D_csph1Dk64_coarsehist_tv0']['depths'], vmin=min_depth, vmax=max_depth); 
				plt.title('db3D_csph1Dk64_coarsehist \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D_csph1Dk64_coarsehist_tv0']['metrics']['mse'], rec_depths['db3D_csph1Dk64_coarsehist_tv0']['metrics']['mae']*100),fontsize=14)
				plt.colorbar()
				plt.subplot(2,3,4)
				if(compression_lvl == '32x'):
					plt.imshow(rec_depths['db3D_csph1Dk32_grayfour_tv0']['depths'], vmin=min_depth, vmax=max_depth); 
					plt.title('db3D_csph1Dk32_grayfour \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D_csph1Dk32_grayfour_tv0']['metrics']['mse'], rec_depths['db3D_csph1Dk32_grayfour_tv0']['metrics']['mae']*100),fontsize=14)
				elif(compression_lvl == '64x'):
					plt.imshow(rec_depths['db3D_csph1Dk16_grayfour_tv0']['depths'], vmin=min_depth, vmax=max_depth); 
					plt.title('db3D_csph1Dk16_grayfour \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D_csph1Dk16_grayfour_tv0']['metrics']['mse'], rec_depths['db3D_csph1Dk16_grayfour_tv0']['metrics']['mae']*100),fontsize=14)
				elif(compression_lvl == '128x'):
					plt.imshow(rec_depths['db3D_csph1Dk8_grayfour_tv0']['depths'], vmin=min_depth, vmax=max_depth); 
					plt.title('db3D_csph1Dk8_grayfour \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D_csph1Dk8_grayfour_tv0']['metrics']['mse'], rec_depths['db3D_csph1Dk8_grayfour_tv0']['metrics']['mae']*100),fontsize=14)
				# plt.imshow(db3D_csph3Dk32_full4x4x64_opt_rand_tv0_depths, vmin=min_depth, vmax=max_depth); 
				# plt.title('db3D_csph3Dk32_full4x4x64_opt_rand \n mse: {:.3f}m | mae: {:.2f}cm'.format(db3D_csph3Dk32_full4x4x64_opt_rand_tv0_mse, db3D_csph3Dk32_full4x4x64_opt_rand_tv0_mae*100),fontsize=14)
				plt.colorbar()
				plt.subplot(2,3,5)
				if(compression_lvl == '32x'):
					plt.imshow(rec_depths['db3D_csph3Dk32_full4x4x64_opt_rand_tv0']['depths'], vmin=min_depth, vmax=max_depth); 
					plt.title('db3D_csph3Dk32_full4x4x64_opt_rand \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D_csph3Dk32_full4x4x64_opt_rand_tv0']['metrics']['mse'], rec_depths['db3D_csph3Dk32_full4x4x64_opt_rand_tv0']['metrics']['mae']*100),fontsize=14)					
				elif(compression_lvl == '64x'):
					plt.imshow(rec_depths['db3D_csph3Dk16_full4x4x64_opt_rand_tv0']['depths'], vmin=min_depth, vmax=max_depth); 
					plt.title('db3D_csph3Dk16_full4x4x64_opt_rand \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D_csph3Dk16_full4x4x64_opt_rand_tv0']['metrics']['mse'], rec_depths['db3D_csph3Dk16_full4x4x64_opt_rand_tv0']['metrics']['mae']*100),fontsize=14)
				elif(compression_lvl == '128x'):
					plt.imshow(rec_depths['db3D_csph3Dk16_full4x4x128_opt_rand_tv0']['depths'], vmin=min_depth, vmax=max_depth); 
					plt.title('db3D_csph3Dk16_full4x4x128_opt_rand \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D_csph3Dk16_full4x4x128_opt_rand_tv0']['metrics']['mse'], rec_depths['db3D_csph3Dk16_full4x4x128_opt_rand_tv0']['metrics']['mae']*100),fontsize=14)
				elif(compression_lvl == '256x'):
					plt.imshow(rec_depths['db3D_csph3Dk8_full4x4x128_opt_rand_tv0']['depths'], vmin=min_depth, vmax=max_depth); 
					plt.title('db3D_csph3Dk8_full4x4x128_opt_rand \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D_csph3Dk8_full4x4x128_opt_rand_tv0']['metrics']['mse'], rec_depths['db3D_csph3Dk8_full4x4x128_opt_rand_tv0']['metrics']['mae']*100),fontsize=14)
				plt.colorbar()
				plt.subplot(2,3,6)
				if(compression_lvl == '32x'):
					plt.imshow(rec_depths['db3D_csphk512_separable4x4x1024_opt_grayfour_tv0']['depths'], vmin=min_depth, vmax=max_depth); 
					plt.title('db3D_csphk512_separable4x4x1024_opt_grayfour \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D_csphk512_separable4x4x1024_opt_grayfour_tv0']['metrics']['mse'], rec_depths['db3D_csphk512_separable4x4x1024_opt_grayfour_tv0']['metrics']['mae']*100),fontsize=14)
				elif(compression_lvl == '64x'):
					plt.imshow(rec_depths['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0']['depths'], vmin=min_depth, vmax=max_depth); 
					plt.title('db3D_csphk256_separable4x4x1024_opt_truncfour \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0']['metrics']['mse'], rec_depths['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0']['metrics']['mae']*100),fontsize=14)
				elif(compression_lvl == '128x'):
					plt.imshow(rec_depths['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0']['depths'], vmin=min_depth, vmax=max_depth); 
					plt.title('db3D_csphk128_separable4x4x1024_opt_truncfour \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0']['metrics']['mse'], rec_depths['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0']['metrics']['mae']*100),fontsize=14)
				elif(compression_lvl == '256x'):
					plt.imshow(rec_depths['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0']['depths'], vmin=min_depth, vmax=max_depth); 
					plt.title('db3D_csphk64_separable4x4x1024_opt_grayfour \n mse: {:.3f}m | mae: {:.2f}cm'.format(rec_depths['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0']['metrics']['mse'], rec_depths['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0']['metrics']['mae']*100),fontsize=14)
				plt.colorbar()
				if(compression_lvl is None):
					out_fname = 'depths_' + scene_fname
				else:
					out_fname = 'depths_' + scene_fname + '_' + compression_lvl
				plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname)


	print("Test Set RMSE | MAE (cm) | MSE (cm):")
	## Compute and store metrics for all models
	for model_id in model_metrics_all.keys():
		curr_model_metrics = model_metrics_all[model_id]
		mean_rmse = calc_mean_rmse(model_metrics_all[model_id])
		mean_mse = calc_mean_mse(model_metrics_all[model_id])
		mean_mae = calc_mean_mae(model_metrics_all[model_id])
		print("    {}: {:.3f} | {:.2f}cm | {:.2f}".format(model_id, mean_rmse, mean_mae, mean_mse))

	if(plot_compression_vs_perf):
		compression_vs_perf = {}
		compression_vs_perf['mae'] = {}
		compression_rates = [16, 32, 64, 128, 256]
		compression_vs_perf['db3D'] = 'No Compression'
		compression_vs_perf['mae']['db3D'] = [calc_mean_mae(model_metrics_all['db3D'])]*len(compression_rates)
		compression_vs_perf['db3D_d2d'] = 'Argmax Compression (Large Memory)'
		compression_vs_perf['mae']['db3D_d2d'] = [calc_mean_mae(model_metrics_all['db3D_d2d'])]*len(compression_rates)
		compression_vs_perf['db3D_csph3D_full4x4x1024_opt_rand_tv0'] = 'CSPH3D + Full4x4x1024 + Backproj Up + Rand Init'
		compression_vs_perf['mae']['db3D_csph3D_full4x4x1024_opt_rand_tv0'] = [ \
			np.nan
			, calc_mean_mae(model_metrics_all['db3D_csph3Dk512_full4x4x1024_opt_rand_tv0'])
			, calc_mean_mae(model_metrics_all['db3D_csph3Dk256_full4x4x1024_opt_rand_tv0'])
			, calc_mean_mae(model_metrics_all['db3D_csph3Dk128_full4x4x1024_opt_rand_tv0'])
			, calc_mean_mae(model_metrics_all['db3D_csph3Dk64_full4x4x1024_opt_rand_tv0'])
			]
		compression_vs_perf['db3D_csph3D_separable4x4x1024_opt_rand_tv0'] = 'CSPH3D + Separable4x4x1024 + Backproj Up + Rand Init'
		compression_vs_perf['mae']['db3D_csph3D_separable4x4x1024_opt_rand_tv0'] = [ \
			np.nan
			, calc_mean_mae(model_metrics_all['db3D_csph3Dk512_separable4x4x1024_opt_rand_tv0'])
			, calc_mean_mae(model_metrics_all['db3D_csph3Dk256_separable4x4x1024_opt_rand_tv0'])
			, calc_mean_mae(model_metrics_all['db3D_csph3Dk128_separable4x4x1024_opt_rand_tv0'])
			, calc_mean_mae(model_metrics_all['db3D_csph3Dk64_separable4x4x1024_opt_rand_tv0'])
			]
		compression_vs_perf['db3D_csph3D_separable4x4x1024_opt_grayfour_tv0'] = 'CSPH3D + Separable4x4x1024 + Backproj Up + GrayFour init'
		compression_vs_perf['mae']['db3D_csph3D_separable4x4x1024_opt_grayfour_tv0'] = [ \
			np.nan
			, calc_mean_mae(model_metrics_all['db3D_csph3Dk512_separable4x4x1024_opt_grayfour_tv0'])
			, calc_mean_mae(model_metrics_all['db3D_csph3Dk256_separable4x4x1024_opt_grayfour_tv0'])
			, calc_mean_mae(model_metrics_all['db3D_csph3Dk128_separable4x4x1024_opt_grayfour_tv0'])
			, calc_mean_mae(model_metrics_all['db3D_csph3Dk64_separable4x4x1024_opt_grayfour_tv0'])
			]
		# compression_vs_perf['db3D_csph3D_full4x4x128_opt_rand_tv0'] = 'CSPH3D + Full4x4x128 + Backproj Up'
		# compression_vs_perf['mae']['db3D_csph3D_full4x4x128_opt_rand_tv0'] = [ \
		# 	np.nan
		# 	, calc_mean_mae(model_metrics_all['db3D_csph3Dk32_full4x4x64_opt_rand_tv0'])
		# 	, calc_mean_mae(model_metrics_all['db3D_csph3Dk16_full4x4x64_opt_rand_tv0'])
		# 	, calc_mean_mae(model_metrics_all['db3D_csph3Dk16_full4x4x128_opt_rand_tv0'])
		# 	, calc_mean_mae(model_metrics_all['db3D_csph3Dk8_full4x4x128_opt_rand_tv0'])
		# 	]
		# compression_vs_perf['db3D_csph_separable4x4x1024_opt_four_tv0'] = 'CSPH3D + Separable4x4x1024 + Learned Up'
		# compression_vs_perf['mae']['db3D_csph_separable4x4x1024_opt_four_tv0'] = [ \
		# 	np.nan
		# 	, calc_mean_mae(model_metrics_all['db3D_csphk512_separable4x4x1024_opt_grayfour_tv0'])
		# 	, calc_mean_mae(model_metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0'])
		# 	, calc_mean_mae(model_metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0'])
		# 	, calc_mean_mae(model_metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0'])
		# 	]
		compression_vs_perf['db3D_csph1D_grayfour_tv0'] = 'CSPH1D + GrayFour1x1x1024 + ZNCC Up'
		compression_vs_perf['mae']['db3D_csph1D_grayfour_tv0'] = [ \
			calc_mean_mae(model_metrics_all['db3D_csph1Dk64_grayfour_tv0'])
			, calc_mean_mae(model_metrics_all['db3D_csph1Dk32_grayfour_tv0'])
			, calc_mean_mae(model_metrics_all['db3D_csph1Dk16_grayfour_tv0'])
			, calc_mean_mae(model_metrics_all['db3D_csph1Dk8_grayfour_tv0'])
			, np.nan
			]
		compression_vs_perf['db3D_csph1Dk64_coarsehist_tv0'] = 'Coarse Hist with 64 bins (16x compression)'
		compression_vs_perf['mae']['db3D_csph1Dk64_coarsehist_tv0'] = [ calc_mean_mae(model_metrics_all['db3D_csph1Dk64_coarsehist_tv0'])]*len(compression_rates)
		
		compression_vs_perf['mse'] = {}
		compression_vs_perf['mse']['db3D'] = [calc_mean_mse(model_metrics_all['db3D'])]*len(compression_rates)
		compression_vs_perf['mse']['db3D_d2d'] = [calc_mean_mse(model_metrics_all['db3D_d2d'])]*len(compression_rates)
		compression_vs_perf['mse']['db3D_csph3D_full4x4x1024_opt_rand_tv0'] = [ \
			np.nan
			, calc_mean_mse(model_metrics_all['db3D_csph3Dk512_full4x4x1024_opt_rand_tv0'])
			, calc_mean_mse(model_metrics_all['db3D_csph3Dk256_full4x4x1024_opt_rand_tv0'])
			, calc_mean_mse(model_metrics_all['db3D_csph3Dk128_full4x4x1024_opt_rand_tv0'])
			, calc_mean_mse(model_metrics_all['db3D_csph3Dk64_full4x4x1024_opt_rand_tv0'])
			]
		compression_vs_perf['mse']['db3D_csph3D_separable4x4x1024_opt_rand_tv0'] = [ \
			np.nan
			, calc_mean_mse(model_metrics_all['db3D_csph3Dk512_separable4x4x1024_opt_rand_tv0'])
			, calc_mean_mse(model_metrics_all['db3D_csph3Dk256_separable4x4x1024_opt_rand_tv0'])
			, calc_mean_mse(model_metrics_all['db3D_csph3Dk128_separable4x4x1024_opt_rand_tv0'])
			, calc_mean_mse(model_metrics_all['db3D_csph3Dk64_separable4x4x1024_opt_rand_tv0'])
			]
		compression_vs_perf['mse']['db3D_csph3D_separable4x4x1024_opt_grayfour_tv0'] = [ \
			np.nan
			, calc_mean_mse(model_metrics_all['db3D_csph3Dk512_separable4x4x1024_opt_grayfour_tv0'])
			, calc_mean_mse(model_metrics_all['db3D_csph3Dk256_separable4x4x1024_opt_grayfour_tv0'])
			, calc_mean_mse(model_metrics_all['db3D_csph3Dk128_separable4x4x1024_opt_grayfour_tv0'])
			, calc_mean_mse(model_metrics_all['db3D_csph3Dk64_separable4x4x1024_opt_grayfour_tv0'])
			]
		# compression_vs_perf['mse']['db3D_csph3D_full4x4x128_opt_rand_tv0'] = [ \
		# 	np.nan
		# 	, calc_mean_mse(model_metrics_all['db3D_csph3Dk32_full4x4x64_opt_rand_tv0'])
		# 	, calc_mean_mse(model_metrics_all['db3D_csph3Dk16_full4x4x64_opt_rand_tv0'])
		# 	, calc_mean_mse(model_metrics_all['db3D_csph3Dk16_full4x4x128_opt_rand_tv0'])
		# 	, calc_mean_mse(model_metrics_all['db3D_csph3Dk8_full4x4x128_opt_rand_tv0'])
		# 	]
		# compression_vs_perf['mse']['db3D_csph_separable4x4x1024_opt_four_tv0'] = [ \
		# 	np.nan
		# 	, calc_mean_mse(model_metrics_all['db3D_csphk512_separable4x4x1024_opt_grayfour_tv0'])
		# 	, calc_mean_mse(model_metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0'])
		# 	, calc_mean_mse(model_metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0'])
		# 	, calc_mean_mse(model_metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0'])
		# 	]
		compression_vs_perf['mse']['db3D_csph1D_grayfour_tv0'] = [ \
			calc_mean_mse(model_metrics_all['db3D_csph1Dk64_grayfour_tv0'])
			, calc_mean_mse(model_metrics_all['db3D_csph1Dk32_grayfour_tv0'])
			, calc_mean_mse(model_metrics_all['db3D_csph1Dk16_grayfour_tv0'])
			, calc_mean_mse(model_metrics_all['db3D_csph1Dk8_grayfour_tv0'])
			, np.nan
			]
		compression_vs_perf['mse']['db3D_csph1Dk64_coarsehist_tv0'] = [ calc_mean_mse(model_metrics_all['db3D_csph1Dk64_coarsehist_tv0'])]*len(compression_rates)
		  
		
		plt.clf()
		metric_name = 'mae'
		fname = 'compression_vs_'+metric_name
		if(len(sbr_params) == 1): fname = fname + '_' + sbr_params[0]
		for model_key in compression_vs_perf[metric_name]:
			model_perf = compression_vs_perf[metric_name][model_key]
			if((model_key == 'db3D') or  (model_key == 'db3D_d2d') or (model_key == 'db3D_csph1Dk64_coarsehist_tv0')):
				plt.plot(compression_rates, model_perf, '--', linewidth=2, label=compression_vs_perf[model_key])
			else:
				plt.plot(compression_rates, model_perf, '-o', linewidth=2, label=compression_vs_perf[model_key])
		plt.legend(loc='upper right')
		plt.title(fname, fontsize=14)
		plt.xlabel("Compression Level", fontsize=14)
		plt.ylabel(metric_name, fontsize=14)
		plot_utils.save_currfig_png(dirpath=out_dirpath, filename=fname)

		plt.clf()
		metric_name = 'mse'
		fname = 'compression_vs_'+metric_name
		if(len(sbr_params) == 1): fname = fname + '_' + sbr_params[0]
		for model_key in compression_vs_perf[metric_name]:
			model_perf = compression_vs_perf[metric_name][model_key]
			if((model_key == 'db3D') or  (model_key == 'db3D_d2d') or (model_key == 'db3D_csph1Dk64_coarsehist_tv0')):
				plt.plot(compression_rates, model_perf, '--', linewidth=2, label=compression_vs_perf[model_key])
			else:
				plt.plot(compression_rates, model_perf, '-o', linewidth=2, label=compression_vs_perf[model_key])
		plt.legend(loc='upper right')
		plt.title(fname, fontsize=14)
		plt.xlabel("Compression Level", fontsize=14)
		plt.ylabel(metric_name, fontsize=14)
		plot_utils.save_currfig_png(dirpath=out_dirpath, filename=fname)