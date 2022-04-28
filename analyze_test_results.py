#### Standard Library Imports
import os

#### Library imports
from hydra import compose, initialize
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from tof_utils import *
from research_utils import plot_utils

def denorm_bins(bins, num_bins):
    return bins*(num_bins)

def compute_rmse(gt, est):
    rmse = np.sqrt(np.mean((gt - est)**2))
    return rmse

def compute_mae(gt, est):
    mae = np.mean((gt - est)**2)
    return mae

def compute_error_metrics(gt, est):
    abs_errs = np.abs(gt-est)
    return (compute_rmse(gt,est), compute_mae(gt,est), abs_errs)


if __name__=='__main__':

    out_dirpath = './results/week_2022-04-25/test_results'

    ## Scene ID and Params
    scene_id = 'spad_Art'
    sbr_params = '2_50'
    scene_fname = '{}_{}'.format(scene_id, sbr_params)

    ## Set dirpaths
    compressive_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_NL_Compressive/debug/2022-04-23_132059/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    deepboosting_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_NL_original/debug/2022-04-20_185832/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    depth2depth_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_NL_Depth2Depth/debug/2022-04-22_134732/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db2D_d2d01Inputs_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/debug/2022-04-27_104532/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db2D_d2d_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth/B-12_MS-8/debug/2022-04-27_103314/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    gt_data_dirpath = 'data_gener/TestData/middlebury/processed/SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'

    compressive_model_result_fpath = os.path.join(compressive_model_result_dirpath, scene_fname+'.npz')
    deepboosting_model_result_fpath = os.path.join(deepboosting_model_result_dirpath, scene_fname+'.npz')
    depth2depth_model_result_fpath = os.path.join(depth2depth_model_result_dirpath, scene_fname+'.npz')
    db2D_d2d01Inputs_model_result_fpath = os.path.join(db2D_d2d01Inputs_model_result_dirpath, scene_fname+'.npz')
    db2D_d2d_model_result_fpath = os.path.join(db2D_d2d_model_result_dirpath, scene_fname+'.npz')
    gt_data_fpath = os.path.join(gt_data_dirpath, scene_fname+'.mat')

    ## Load data
    compressive_model_result = np.load(compressive_model_result_fpath)
    deepboosting_model_result = np.load(deepboosting_model_result_fpath)
    depth2depth_model_result = np.load(depth2depth_model_result_fpath)
    db2D_d2d01Inputs_model_result = np.load(db2D_d2d01Inputs_model_result_fpath)
    db2D_d2d_model_result = np.load(db2D_d2d_model_result_fpath)
    gt_data = io.loadmat(gt_data_fpath)

    ## Load params
    (nr, nc, num_bins) = gt_data['rates'].shape
    tres = gt_data['bin_size']
    intensity = gt_data['intensity']
    SBR = gt_data['SBR']
    mean_background_photons = gt_data['mean_background_photons']
    mean_signal_photons = gt_data['mean_signal_photons']
    tau = num_bins*tres
    
    ## Load depths in bins
    gt_bins = gt_data['range_bins']
    lmf_bins = gt_data['est_range_bins_lmf']
    argmax_bins = gt_data['est_range_bins_argmax']
    compressive_model_bins = denorm_bins(compressive_model_result['dep_re'], num_bins=num_bins).squeeze()
    deepboosting_model_bins = denorm_bins(deepboosting_model_result['dep_re'], num_bins=num_bins).squeeze()
    depth2depth_model_bins = denorm_bins(depth2depth_model_result['dep_re'], num_bins=num_bins).squeeze()
    db2D_d2d01Inputs_model_bins = denorm_bins(db2D_d2d01Inputs_model_result['dep_re'], num_bins=num_bins).squeeze()
    db2D_d2d_model_bins = denorm_bins(db2D_d2d_model_result['dep_re'], num_bins=num_bins).squeeze()

    ## Convert bins to depth
    gt_depths = bin2depth(gt_bins, num_bins=num_bins, tau=tau)
    lmf_depths = bin2depth(lmf_bins, num_bins=num_bins, tau=tau)
    argmax_depths = bin2depth(argmax_bins, num_bins=num_bins, tau=tau)
    compressive_model_depths = bin2depth(compressive_model_bins, num_bins=num_bins, tau=tau)
    deepboosting_model_depths = bin2depth(deepboosting_model_bins, num_bins=num_bins, tau=tau)
    depth2depth_model_depths = bin2depth(depth2depth_model_bins, num_bins=num_bins, tau=tau)
    db2D_d2d01Inputs_model_depths = bin2depth(db2D_d2d01Inputs_model_bins, num_bins=num_bins, tau=tau)
    db2D_d2d_model_depths = bin2depth(db2D_d2d_model_bins, num_bins=num_bins, tau=tau)

    ## Compute RMSE and MAE
    (gt_rmse, gt_mae, gt_abs_errs) = compute_error_metrics(gt_depths, gt_depths)
    (lmf_rmse, lmf_mae, lmf_abs_errs) = compute_error_metrics(gt_depths, lmf_depths)
    (argmax_rmse, argmax_mae, argmax_abs_errs) = compute_error_metrics(gt_depths, argmax_depths)
    (compressive_model_rmse, compressive_model_mae, compressive_model_abs_errs) = compute_error_metrics(gt_depths, compressive_model_depths)
    (deepboosting_model_rmse, deepboosting_model_mae, deepboosting_model_abs_errs) = compute_error_metrics(gt_depths, deepboosting_model_depths)
    (depth2depth_model_rmse, depth2depth_model_mae, depth2depth_model_abs_errs) = compute_error_metrics(gt_depths, depth2depth_model_depths)
    (db2D_d2d01Inputs_model_rmse, db2D_d2d01Inputs_model_mae, db2D_d2d01Inputs_model_abs_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_model_depths)
    (db2D_d2d_model_rmse, db2D_d2d_model_mae, db2D_d2d_model_abs_errs) = compute_error_metrics(gt_depths, db2D_d2d_model_depths)

    min_depth = gt_depths.flatten().min()
    max_depth = gt_depths.flatten().max()

    min_err = 0
    max_err = 0.08

    plt.clf()
    plt.suptitle("{} - SBR: {}, Signal: {} photons, Bkg: {} photons".format(scene_fname, SBR, mean_signal_photons, mean_background_photons), fontsize=20)
    plt.subplot(2,4,1)
    plt.imshow(gt_depths, vmin=min_depth, vmax=max_depth); 
    plt.title('gt_depths \n rmse: {:.4f}'.format(gt_rmse),fontsize=14)
    plt.colorbar()
    plt.subplot(2,4,2)
    plt.imshow(lmf_depths, vmin=min_depth, vmax=max_depth); 
    plt.title('lmf_depths \n rmse: {:.4f}'.format(lmf_rmse),fontsize=14)
    plt.colorbar()
    plt.subplot(2,4,3)
    plt.imshow(argmax_depths, vmin=min_depth, vmax=max_depth); 
    plt.title('argmax_depths \n rmse: {:.4f}'.format(argmax_rmse),fontsize=14)
    plt.colorbar()
    plt.subplot(2,4,4)
    plt.imshow(deepboosting_model_depths, vmin=min_depth, vmax=max_depth); 
    plt.title('deepboosting_model_depths \n rmse: {:.4f}'.format(deepboosting_model_rmse),fontsize=14)
    plt.colorbar()
    plt.subplot(2,4,5)
    plt.imshow(compressive_model_depths, vmin=min_depth, vmax=max_depth); 
    plt.title('compressive_model_depths \n rmse: {:.4f}'.format(compressive_model_rmse),fontsize=14)
    plt.colorbar()
    plt.subplot(2,4,6)
    plt.imshow(depth2depth_model_depths, vmin=min_depth, vmax=max_depth); 
    plt.title('depth2depth_model_depths \n rmse: {:.4f}'.format(depth2depth_model_rmse),fontsize=14)
    plt.colorbar()
    plt.subplot(2,4,7)
    plt.imshow(db2D_d2d_model_depths, vmin=min_depth, vmax=max_depth); 
    plt.title('db2D_d2d_model_depths \n rmse: {:.4f}'.format(db2D_d2d_model_rmse),fontsize=14)
    plt.colorbar()
    plt.subplot(2,4,8)
    plt.imshow(db2D_d2d01Inputs_model_depths, vmin=min_depth, vmax=max_depth); 
    plt.title('db2D_d2d01Inputs_model_depths \n rmse: {:.4f}'.format(db2D_d2d01Inputs_model_rmse),fontsize=14)
    plt.colorbar()
    out_fname = 'depths_' + scene_fname
    plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname)

    # plt.clf()
    # plt.suptitle("{} - SBR: {}, Signal: {} photons, Bkg: {} photons".format(scene_fname, SBR, mean_signal_photons, mean_background_photons), fontsize=20)
    # plt.subplot(2,3,1)
    # plt.imshow(gt_abs_errs, vmin=min_err, vmax=max_err); 
    # plt.title('gt_abs_errs \n rmse: {:.4f}'.format(gt_rmse),fontsize=14)
    # plt.colorbar()
    # plt.subplot(2,3,2)
    # plt.imshow(lmf_abs_errs, vmin=min_err, vmax=max_err); 
    # plt.title('lmf_abs_errs \n rmse: {:.4f}'.format(lmf_rmse),fontsize=14)
    # plt.colorbar()
    # plt.subplot(2,3,3)
    # plt.imshow(argmax_abs_errs, vmin=min_err, vmax=max_err); 
    # plt.title('argmax_abs_errs \n rmse: {:.4f}'.format(argmax_rmse),fontsize=14)
    # plt.colorbar()
    # plt.subplot(2,3,4)
    # plt.imshow(deepboosting_model_abs_errs, vmin=min_err, vmax=max_err); 
    # plt.title('deepboosting_model_abs_errs \n rmse: {:.4f}'.format(deepboosting_model_rmse),fontsize=14)
    # plt.colorbar()
    # plt.subplot(2,3,5)
    # plt.imshow(compressive_model_abs_errs, vmin=min_err, vmax=max_err); 
    # plt.title('compressive_model_abs_errs \n rmse: {:.4f}'.format(compressive_model_rmse),fontsize=14)
    # plt.colorbar()
    # plt.subplot(2,3,6)
    # plt.imshow(depth2depth_model_abs_errs, vmin=min_err, vmax=max_err); 
    # plt.title('depth2depth_model_abs_errs \n rmse: {:.4f}'.format(depth2depth_model_rmse),fontsize=14)
    # plt.colorbar()
    # out_fname = 'errors_' + scene_fname
    # plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname)
