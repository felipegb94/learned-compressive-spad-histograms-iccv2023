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
from research_utils import plot_utils, np_utils

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
    perc_errs = np_utils.calc_mean_percentile_errors(abs_errs)
    return (compute_rmse(gt,est), compute_mae(gt,est), abs_errs, np.round(perc_errs[0], decimals=2))

def get_model_depths(model_result_dirpath, scene_id, sbr_params, num_bins, tau):
    scene_fname = '{}_{}'.format(scene_id, sbr_params)
    model_result_fpath = os.path.join(model_result_dirpath, scene_fname+'.npz')
    model_result = np.load(model_result_fpath)
    model_bins = denorm_bins(model_result['dep_re'], num_bins=num_bins).squeeze()
    model_depths = bin2depth(model_bins, num_bins=num_bins, tau=tau)
    return model_depths


if __name__=='__main__':

    # experiment_name = ''
    experiment_name = 'd2d2D_B12_tv_comparisson'

    out_dirpath = os.path.join('./results/week_2022-05-02/test_results', experiment_name)
    os.makedirs(out_dirpath, exist_ok=True)

    plot_results = False

    ## Scene ID and Params
    scene_ids = ['spad_Art', 'spad_Reindeer', 'spad_Books', 'spad_Moebius', 'spad_Bowling1', 'spad_Dolls', 'spad_Laundry', 'spad_Plastic']
    # sbr_params = ['2_2','2_10','2_50','5_2','5_10','5_50','10_2','10_10','10_50']

    # scene_ids = ['spad_Art']
    # scene_ids = ['spad_Books']
    # scene_ids = ['spad_Books']
    # sbr_params = ['2_2','2_10','2_50','5_2','5_10','5_50','10_2','10_10','10_50']
    sbr_params = ['2_50']

    ## Set dirpaths
    compressive_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_NL_Compressive/debug/2022-04-23_132059/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db3D_nl_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_NL_original/debug/2022-04-20_185832/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db3D_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10/debug/2022-04-25_192521/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db3D_nl_d2d_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_NL_Depth2Depth/debug/2022-04-22_134732/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db3D_d2d_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_Depth2Depth/2022-05-02_085659/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db2D_d2d2hist01Inputs_B12_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth2Hist_01Inputs/B-12_MS-8/2022-05-02_214727/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db2D_d2d01Inputs_B12_tv1m3_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/loss-L1_tv-0.001/2022-05-03_183044/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db2D_d2d01Inputs_B12_tv1m4_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/loss-L1_tv-0.0001/2022-05-03_183002/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db2D_d2d01Inputs_B12_tv3m5_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/loss-L1_tv-3e-05/2022-05-03_185303/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db2D_d2d01Inputs_B12_tv1m5_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/debug/2022-04-27_104532/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db2D_d2d01Inputs_B12_tv1m10_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/loss-L1_tv-1e-10/2022-05-03_183128/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db2D_d2d01Inputs_B16_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-16_MS-8/2022-05-01_172344/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    db2D_d2d01Inputs_B24_model_result_dirpath = 'outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-24_MS-8/2022-04-28_163253/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    gt_data_dirpath = 'data_gener/TestData/middlebury/processed/SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'

    gt_rmse_all = []
    lmf_rmse_all = []
    argmax_rmse_all = []
    compressive_model_rmse_all = []
    db3D_nl_rmse_all = []
    db3D_rmse_all = []
    db3D_nl_d2d_rmse_all = []
    db3D_d2d_rmse_all = []
    db2D_d2d2hist01Inputs_B12_model_rmse_all = []
    db2D_d2d01Inputs_B12_tv1m3_model_rmse_all = []
    db2D_d2d01Inputs_B12_tv1m4_model_rmse_all = []
    db2D_d2d01Inputs_B12_tv3m5_model_rmse_all = []
    db2D_d2d01Inputs_B12_tv1m5_model_rmse_all = []
    db2D_d2d01Inputs_B12_tv1m10_model_rmse_all = []
    db2D_d2d01Inputs_B16_model_rmse_all = []
    db2D_d2d01Inputs_B24_model_rmse_all = []

    for i in range(len(scene_ids)):
        for j in range(len(sbr_params)):
            curr_scene_id = scene_ids[i] 
            curr_sbr_params = sbr_params[j] 

            scene_fname = '{}_{}'.format(curr_scene_id, curr_sbr_params)
            print("Processing: {}".format(scene_fname))

            gt_data_fpath = os.path.join(gt_data_dirpath, scene_fname+'.mat')
            gt_data = io.loadmat(gt_data_fpath)
            gt_bins = gt_data['range_bins']-1 # Subtract 1 because of matlab index notation
            lmf_bins = gt_data['est_range_bins_lmf']-1 # Subtract 1 because of matlab index notation
            argmax_bins = gt_data['est_range_bins_argmax']-1 # Subtract 1 because of matlab index notation
    
            ## Load params
            (nr, nc, num_bins) = gt_data['rates'].shape
            tres = gt_data['bin_size']
            intensity = gt_data['intensity']
            SBR = gt_data['SBR']
            mean_background_photons = gt_data['mean_background_photons']
            mean_signal_photons = gt_data['mean_signal_photons']
            tau = num_bins*tres

            ## Get depths for MATLAB data
            gt_depths = bin2depth(gt_bins, num_bins=num_bins, tau=tau)
            lmf_depths = bin2depth(lmf_bins, num_bins=num_bins, tau=tau)
            argmax_depths = bin2depth(argmax_bins, num_bins=num_bins, tau=tau)

            ## Get depths for model outputs
            (compressive_model_depths) = get_model_depths(model_result_dirpath=compressive_model_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_nl_depths) = get_model_depths(model_result_dirpath=db3D_nl_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_depths) = get_model_depths(model_result_dirpath=db3D_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_nl_d2d_depths) = get_model_depths(model_result_dirpath=db3D_nl_d2d_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_d2d_depths) = get_model_depths(model_result_dirpath=db3D_d2d_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d2hist01Inputs_B12_model_depths) = get_model_depths(model_result_dirpath=db2D_d2d2hist01Inputs_B12_model_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B12_tv1m3_model_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B12_tv1m3_model_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B12_tv1m4_model_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B12_tv1m4_model_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B12_tv3m5_model_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B12_tv3m5_model_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B12_tv1m5_model_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B12_tv1m5_model_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B12_tv1m10_model_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B12_tv1m10_model_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B16_model_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B16_model_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B24_model_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B24_model_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)

            ## Compute RMSE and MAE
            (gt_rmse, gt_mae, gt_abs_errs, gt_perc_errs) = compute_error_metrics(gt_depths, gt_depths)
            (lmf_rmse, lmf_mae, lmf_abs_errs, lmf_perc_errs) = compute_error_metrics(gt_depths, lmf_depths)
            (argmax_rmse, argmax_mae, argmax_abs_errs, argmax_perc_errs) = compute_error_metrics(gt_depths, argmax_depths)
            (compressive_model_rmse, compressive_model_mae, compressive_model_abs_errs, compressive_model_perc_errs) = compute_error_metrics(gt_depths, compressive_model_depths)
            (db3D_nl_rmse, db3D_nl_mae, db3D_nl_abs_errs, db3D_nl_perc_errs) = compute_error_metrics(gt_depths, db3D_nl_depths)
            (db3D_rmse, db3D_mae, db3D_abs_errs, db3D_perc_errs) = compute_error_metrics(gt_depths, db3D_depths)
            (db3D_nl_d2d_rmse, db3D_nl_d2d_mae, db3D_nl_d2d_abs_errs, db3D_nl_d2d_perc_errs) = compute_error_metrics(gt_depths, db3D_nl_d2d_depths)
            (db3D_d2d_rmse, db3D_d2d_mae, db3D_d2d_abs_errs, db3D_d2d_perc_errs) = compute_error_metrics(gt_depths, db3D_d2d_depths)
            (db2D_d2d2hist01Inputs_B12_model_rmse, db2D_d2d2hist01Inputs_B12_model_mae, db2D_d2d2hist01Inputs_B12_model_abs_errs, db2D_d2d2hist01Inputs_B12_model_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d2hist01Inputs_B12_model_depths)
            (db2D_d2d01Inputs_B12_tv1m3_model_rmse, db2D_d2d01Inputs_B12_tv1m3_model_mae, db2D_d2d01Inputs_B12_tv1m3_model_abs_errs, db2D_d2d01Inputs_B12_tv1m3_model_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B12_tv1m3_model_depths)
            (db2D_d2d01Inputs_B12_tv1m4_model_rmse, db2D_d2d01Inputs_B12_tv1m4_model_mae, db2D_d2d01Inputs_B12_tv1m4_model_abs_errs, db2D_d2d01Inputs_B12_tv1m4_model_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B12_tv1m4_model_depths)
            (db2D_d2d01Inputs_B12_tv3m5_model_rmse, db2D_d2d01Inputs_B12_tv3m5_model_mae, db2D_d2d01Inputs_B12_tv3m5_model_abs_errs, db2D_d2d01Inputs_B12_tv3m5_model_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B12_tv3m5_model_depths)
            (db2D_d2d01Inputs_B12_tv1m5_model_rmse, db2D_d2d01Inputs_B12_tv1m5_model_mae, db2D_d2d01Inputs_B12_tv1m5_model_abs_errs, db2D_d2d01Inputs_B12_tv1m5_model_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B12_tv1m5_model_depths)
            (db2D_d2d01Inputs_B12_tv1m10_model_rmse, db2D_d2d01Inputs_B12_tv1m10_model_mae, db2D_d2d01Inputs_B12_tv1m10_model_abs_errs, db2D_d2d01Inputs_B12_tv1m10_model_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B12_tv1m10_model_depths)
            (db2D_d2d01Inputs_B16_model_rmse, db2D_d2d01Inputs_B16_model_mae, db2D_d2d01Inputs_B16_model_abs_errs, db2D_d2d01Inputs_B16_model_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B16_model_depths)
            (db2D_d2d01Inputs_B24_model_rmse, db2D_d2d01Inputs_B24_model_mae, db2D_d2d01Inputs_B24_model_abs_errs, db2D_d2d01Inputs_B24_model_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B24_model_depths)


            gt_rmse_all.append(gt_rmse)
            lmf_rmse_all.append(lmf_rmse)
            argmax_rmse_all.append(argmax_rmse)
            compressive_model_rmse_all.append(compressive_model_rmse)
            db3D_nl_rmse_all.append(db3D_nl_rmse)
            db3D_rmse_all.append(db3D_rmse)
            db3D_nl_d2d_rmse_all.append(db3D_nl_d2d_rmse)
            db3D_d2d_rmse_all.append(db3D_d2d_rmse)
            db2D_d2d2hist01Inputs_B12_model_rmse_all.append(db2D_d2d2hist01Inputs_B12_model_rmse)
            db2D_d2d01Inputs_B12_tv1m3_model_rmse_all.append(db2D_d2d01Inputs_B12_tv1m3_model_rmse)
            db2D_d2d01Inputs_B12_tv1m4_model_rmse_all.append(db2D_d2d01Inputs_B12_tv1m4_model_rmse)
            db2D_d2d01Inputs_B12_tv3m5_model_rmse_all.append(db2D_d2d01Inputs_B12_tv3m5_model_rmse)
            db2D_d2d01Inputs_B12_tv1m5_model_rmse_all.append(db2D_d2d01Inputs_B12_tv1m5_model_rmse)
            db2D_d2d01Inputs_B12_tv1m10_model_rmse_all.append(db2D_d2d01Inputs_B12_tv1m10_model_rmse)
            db2D_d2d01Inputs_B16_model_rmse_all.append(db2D_d2d01Inputs_B16_model_rmse)
            db2D_d2d01Inputs_B24_model_rmse_all.append(db2D_d2d01Inputs_B24_model_rmse)

            # print("Percentile Errors - {}:".format(scene_fname))
            # print("    gt: {}".format(gt_perc_errs))
            # print("    lmf: {}".format(lmf_perc_errs))
            # print("    argmax: {}".format(argmax_perc_errs))
            # print("    compressive_model: {}".format(compressive_model_perc_errs))
            # print("    db3D_nl: {}".format(db3D_nl_perc_errs))
            # print("    db3D: {}".format(db3D_perc_errs))
            # print("    db3D_nl_d2d: {}".format(db3D_nl_d2d_perc_errs))
            # print("    db3D_d2d: {}".format(db3D_d2d_perc_errs))
            # print("    db2D_d2d2hist01Inputs_B12_model: {}".format(db2D_d2d2hist01Inputs_B12_model_perc_errs))
            # print("    db2D_d2d01Inputs_B12_tv1m5_model: {}".format(db2D_d2d01Inputs_B12_tv1m5_model_perc_errs))
            # print("    db2D_d2d01Inputs_B16_model: {}".format(db2D_d2d01Inputs_B16_model_perc_errs))
            # print("    db2D_d2d01Inputs_B24_model: {}".format(db2D_d2d01Inputs_B24_model_perc_errs))
            # print("    db2D_d2d_model: {}".format(db2D_d2d_model_perc_errs)) 
            
            min_depth = gt_depths.flatten().min()
            max_depth = gt_depths.flatten().max()
            min_err = 0
            max_err = 0.15

            if(plot_results):

                plt.clf()
                plt.suptitle("{} - SBR: {}, Signal: {} photons, Bkg: {} photons".format(scene_fname, SBR, mean_signal_photons, mean_background_photons), fontsize=20)
                plt.subplot(2,4,1)
                # plt.imshow(gt_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('gt_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(gt_rmse, gt_mae*100, gt_perc_errs),fontsize=14)
                plt.imshow(db3D_d2d_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('db3D_d2d_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_d2d_rmse, db3D_d2d_mae*100, db3D_d2d_perc_errs),fontsize=14)
                plt.colorbar()
                plt.subplot(2,4,2)
                plt.imshow(lmf_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('lmf_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(lmf_rmse, lmf_mae*100, lmf_perc_errs),fontsize=14)
                plt.colorbar()
                plt.subplot(2,4,3)
                plt.imshow(argmax_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('argmax_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(argmax_rmse, argmax_mae*100, argmax_perc_errs),fontsize=14)
                plt.colorbar()
                plt.subplot(2,4,4)
                plt.imshow(db3D_nl_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('db3D_nl_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_nl_rmse, db3D_nl_mae*100, db3D_nl_perc_errs),fontsize=14)
                plt.colorbar()
                plt.subplot(2,4,5)
                # plt.imshow(compressive_model_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('compressive_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(compressive_model_rmse, compressive_model_mae*100, compressive_model_perc_errs),fontsize=14)
                plt.imshow(db3D_nl_d2d_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('db3D_nl_d2d_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_nl_d2d_rmse, db3D_nl_d2d_mae*100, db3D_nl_d2d_perc_errs),fontsize=14)
                # plt.imshow(db2D_d2d2hist01Inputs_B12_model_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db2D_d2d2hist_B12_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d2hist01Inputs_B12_model_rmse, db2D_d2d2hist01Inputs_B12_model_mae*100, db2D_d2d2hist01Inputs_B12_model_perc_errs),fontsize=14)
                plt.imshow(db2D_d2d01Inputs_B12_tv1m3_model_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('db2D_B12_tv1m3_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B12_tv1m3_model_rmse, db2D_d2d01Inputs_B12_tv1m3_model_mae*100, db2D_d2d01Inputs_B12_tv1m3_model_perc_errs),fontsize=14)
                plt.colorbar()
                plt.subplot(2,4,6)
                plt.imshow(db2D_d2d01Inputs_B12_tv3m5_model_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('db2D_B12_tv3m5_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B12_tv3m5_model_rmse, db2D_d2d01Inputs_B12_tv3m5_model_mae*100, db2D_d2d01Inputs_B12_tv3m5_model_perc_errs),fontsize=14)
                # plt.imshow(db2D_d2d01Inputs_B12_tv1m4_model_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db2D_B12_tv1m4_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B12_tv1m4_model_rmse, db2D_d2d01Inputs_B12_tv1m4_model_mae*100, db2D_d2d01Inputs_B12_tv1m4_model_perc_errs),fontsize=14)
                plt.colorbar()
                plt.subplot(2,4,7)
                plt.imshow(db2D_d2d01Inputs_B12_tv1m5_model_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('db2D_B12_tv1m5_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B12_tv1m5_model_rmse, db2D_d2d01Inputs_B12_tv1m5_model_mae*100, db2D_d2d01Inputs_B12_tv1m5_model_perc_errs),fontsize=14)
                # plt.imshow(db2D_d2d01Inputs_B16_model_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db2D_B16_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B16_model_rmse, db2D_d2d01Inputs_B16_model_mae*100, db2D_d2d01Inputs_B16_model_perc_errs),fontsize=14)
                plt.colorbar()
                plt.subplot(2,4,8)
                plt.imshow(db2D_d2d01Inputs_B12_tv1m10_model_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('db2D_B12_tv1m10_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B12_tv1m10_model_rmse, db2D_d2d01Inputs_B12_tv1m10_model_mae*100, db2D_d2d01Inputs_B12_tv1m10_model_perc_errs),fontsize=14)
                # plt.imshow(db2D_d2d01Inputs_B24_model_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db2D_B24_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B24_model_rmse, db2D_d2d01Inputs_B24_model_mae*100, db2D_d2d01Inputs_B24_model_perc_errs),fontsize=14)
                plt.colorbar()
                out_fname = 'depths_' + scene_fname
                plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname)


                # plt.clf()
                # plt.suptitle("{} - SBR: {}, Signal: {} photons, Bkg: {} photons".format(scene_fname, SBR, mean_signal_photons, mean_background_photons), fontsize=20)
                # plt.subplot(2,4,1)
                # plt.imshow(gt_depths, vmin=min_err, vmax=max_err); 
                # plt.title('gt_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(gt_rmse, gt_mae*100, gt_perc_errs),fontsize=14)
                # plt.imshow(db3D_d2d_abs_errs, vmin=min_err, vmax=max_err); 
                # plt.title('db3D_d2d_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_d2d_rmse, db3D_d2d_mae*100, db3D_d2d_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,2)
                # plt.imshow(lmf_abs_errs, vmin=min_err, vmax=max_err); 
                # plt.title('lmf_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(lmf_rmse, lmf_mae*100, lmf_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,3)
                # plt.imshow(argmax_abs_errs, vmin=min_err, vmax=max_err); 
                # plt.title('argmax_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(argmax_rmse, argmax_mae*100, argmax_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,4)
                # plt.imshow(db3D_nl_abs_errs, vmin=min_err, vmax=max_err); 
                # plt.title('db3D_nl_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_nl_rmse, db3D_nl_mae*100, db3D_nl_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,5)
                # # plt.imshow(compressive_model_abs_errs, vmin=min_err, vmax=max_err); 
                # # plt.title('compressive_model_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(compressive_model_rmse, compressive_model_mae*100, compressive_model_perc_errs),fontsize=14)
                # # plt.imshow(db3D_nl_d2d_abs_errs, vmin=min_err, vmax=max_err); 
                # # plt.title('db3D_nl_d2d_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_nl_d2d_rmse, db3D_nl_d2d_mae*100, db3D_nl_d2d_perc_errs),fontsize=14)
                # plt.imshow(db2D_d2d2hist01Inputs_B12_model_abs_errs, vmin=min_err, vmax=max_err); 
                # plt.title('db2D_d2d2hist_B12_model_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d2hist01Inputs_B12_model_rmse, db2D_d2d2hist01Inputs_B12_model_mae*100, db2D_d2d2hist01Inputs_B12_model_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,6)
                # plt.imshow(db2D_d2d01Inputs_B12_tv1m5_model_abs_errs, vmin=min_err, vmax=max_err); 
                # plt.title('db2D_B12_model_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B12_tv1m5_model_rmse, db2D_d2d01Inputs_B12_tv1m5_model_mae*100, db2D_d2d01Inputs_B12_tv1m5_model_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,7)
                # plt.imshow(db2D_d2d01Inputs_B16_model_abs_errs, vmin=min_err, vmax=max_err); 
                # plt.title('db2D_B16_model_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B16_model_rmse, db2D_d2d01Inputs_B16_model_mae*100, db2D_d2d01Inputs_B16_model_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,8)
                # plt.imshow(db2D_d2d01Inputs_B24_model_abs_errs, vmin=min_err, vmax=max_err); 
                # plt.title('db2D_B24_model_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B24_model_rmse, db2D_d2d01Inputs_B24_model_mae*100, db2D_d2d01Inputs_B24_model_perc_errs),fontsize=14)
                # plt.colorbar()
                # out_fname = 'abs_errs_' + scene_fname
                # plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname)

    print("Test Set RMSE:")
    print("    gt: {}".format(np.mean(gt_rmse_all)))
    print("    lmf: {}".format(np.mean(lmf_rmse_all)))
    print("    argmax: {}".format(np.mean(argmax_rmse_all)))
    print("    compressive_model: {}".format(np.mean(compressive_model_rmse_all)))
    print("    db3D_nl: {}".format(np.mean(db3D_nl_rmse_all)))
    print("    db3D: {}".format(np.mean(db3D_rmse_all)))
    print("    db3D_nl_d2d: {}".format(np.mean(db3D_nl_d2d_rmse_all)))
    print("    db3D_d2d: {}".format(np.mean(db3D_d2d_rmse_all)))
    print("    db2D_d2d2hist01Inputs_B12_model: {}".format(np.mean(db2D_d2d2hist01Inputs_B12_model_rmse_all)))
    print("    db2D_d2d01Inputs_B12_tv1m3_model: {}".format(np.mean(db2D_d2d01Inputs_B12_tv1m3_model_rmse_all)))
    print("    db2D_d2d01Inputs_B12_tv1m4_model: {}".format(np.mean(db2D_d2d01Inputs_B12_tv1m4_model_rmse_all)))
    print("    db2D_d2d01Inputs_B12_tv3m5_model: {}".format(np.mean(db2D_d2d01Inputs_B12_tv3m5_model_rmse_all)))
    print("    db2D_d2d01Inputs_B12_tv1m5_model: {}".format(np.mean(db2D_d2d01Inputs_B12_tv1m5_model_rmse_all)))
    print("    db2D_d2d01Inputs_B12_tv1m10_model: {}".format(np.mean(db2D_d2d01Inputs_B12_tv1m10_model_rmse_all)))
    print("    db2D_d2d01Inputs_B16_model: {}".format(np.mean(db2D_d2d01Inputs_B16_model_rmse_all)))
    print("    db2D_d2d01Inputs_B24_model: {}".format(np.mean(db2D_d2d01Inputs_B24_model_rmse_all)))
