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

def compute_mse(gt, est):
    mse = np.mean((gt - est)**2)
    return mse

def compute_mae(gt, est):
    mae = np.mean(np.abs(gt - est))
    return mae

def compute_error_metrics(gt, est):
    abs_errs = np.abs(gt-est)
    perc_errs = np_utils.calc_mean_percentile_errors(abs_errs)
    return (compute_rmse(gt,est), compute_mse(gt,est), compute_mae(gt,est), abs_errs, np.round(perc_errs[0], decimals=2))

def get_model_depths(model_result_dirpath, scene_id, sbr_params, num_bins, tau):
    scene_fname = '{}_{}'.format(scene_id, sbr_params)
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

if __name__=='__main__':

    # experiment_name = ''
    experiment_name = 'd2d2D_B12_tv_comparisson'
    experiment_name = 'phasor2depth_comparisson_v1'
    experiment_name = 'db3D_d2d_comparissons'
    experiment_name = 'db3D_csph1Dk16_comparissons'
    experiment_name = 'csphseparable_vs_csph1D2D_comparissons'
    experiment_name = '64xcompression_csphseparable_vs_csph1D2D_comparissons'
    experiment_name = '128xcompression_csphseparable_comparissons'
    experiment_name = '64xcompression_csphseparable_comparissons'
    experiment_name = 'temporal_down_ablation'

    out_dirpath = os.path.join('./results/week_2022-07-10/test_results', experiment_name)
    os.makedirs(out_dirpath, exist_ok=True)

    plot_results = True

    ## Scene ID and Params
    scene_ids = ['spad_Art', 'spad_Reindeer', 'spad_Books', 'spad_Moebius', 'spad_Bowling1', 'spad_Dolls', 'spad_Laundry', 'spad_Plastic']
    sbr_params = ['2_2','2_10','2_50','5_2','5_10','5_50','10_2','10_10','10_50']

    scene_ids = ['spad_Art']
    # # # scene_ids = ['spad_Books']
    # # scene_ids = ['spad_Books']
    # scene_ids = ['spad_Reindeer']
    # scene_ids = ['spad_Plastic']
    sbr_params = ['10_50']

    test_set_id = 'test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'

    ## Set dirpaths
    compressive_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_NL_Compressive/debug/2022-04-23_132059/', test_set_id)
    db3D_nl_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_NL_original/debug/2022-04-20_185832/', test_set_id)
    db3D_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10/2022-04-25_192521/', test_set_id)
    db3D_nl_d2d_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_NL_Depth2Depth/debug/2022-04-22_134732/', test_set_id)
    db3D_d2d_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_Depth2Depth/2022-05-02_085659/', test_set_id)
    db3D_d2d_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_Depth2Depth/loss-kldiv_tv-0.0/2022-05-07_171658/', test_set_id)
    db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/temporal_down_ablation/DDFN_C64B10_CSPH/k64_down4_Mt1_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-24_094328/', test_set_id)
    db3D_csphk32_separable4x4x512_opt_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/temporal_down_ablation/DDFN_C64B10_CSPH/k32_down4_Mt2_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-26_172638/', test_set_id)
    db3D_csphk16_separable4x4x256_opt_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/temporal_down_ablation/DDFN_C64B10_CSPH/k16_down4_Mt4_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-29_190259/', test_set_id)
    db3D_csphk8_separable4x4x128_opt_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/temporal_down_ablation/DDFN_C64B10_CSPH/k8_down4_Mt8_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-07-02_023357/', test_set_id)
    db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH/k64_down2_Mt1_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-18_183013/', test_set_id)
    db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH/k256_down4_Mt1_TruncFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-22_130057/', test_set_id)
    db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH/k128_down4_Mt1_TruncFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-18_203447/', test_set_id)
    db3D_csphk64_separable4x4x512_opt_truncfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH/k64_down4_Mt2_TruncFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-20_111348/', test_set_id)
    db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH/k32_down2_Mt1_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0/2022-06-21_230853/', test_set_id)
    db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1DGlobal2DLocal4xDown/k128_down4_TruncFourier/loss-kldiv_tv-0.0/2022-06-06_131240/', test_set_id)
    db3D_csph1D2Dk128down2upzncc_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D2D/k128_down2_HybridGrayFourier/loss-kldiv_tv-0.0/2022-05-25_182959_zncc/', test_set_id)
    db3D_csph1D2Dk128down2_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D2D/k128_down2_HybridGrayFourier/loss-kldiv_tv-0.0/2022-06-05_181812/', test_set_id)
    db3D_csph1D2Dk64down2_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D2D/k64_down2_HybridGrayFourier/loss-kldiv_tv-0.0/2022-05-24_220441/', test_set_id)
    db3D_csph1D2Dk32down2_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D2D/k32_down2_HybridGrayFourier/loss-kldiv_tv-0.0/2022-06-03_140623/', test_set_id)
    db3D_csph1Dk8_gray_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D/k8_HybridFourierGray/loss-kldiv_tv-0.0/2022-05-26_084337/', test_set_id)
    db3D_csph1Dk8_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D/k8_HybridGrayFourier/loss-kldiv_tv-0.0/2022-05-27_181604/', test_set_id)
    db3D_csph1Dk16_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D/k16_HybridGrayFourier/loss-kldiv_tv-0.0/2022-05-20_194518/', test_set_id)
    # db3D_csph1Dk16_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_CSPH1D/k16_HybridGrayFourier/loss-kldiv_tv-0.0/2022-05-11_070443/', test_set_id)
    db3D_csph1Dk16_truncfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D/k16_TruncFourier/loss-kldiv_tv-0.0/2022-05-19_142207', test_set_id)
    # db3D_csph1Dk16_truncfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_CSPH1D/k16_TruncFourier/loss-kldiv_tv-0.0/2022-05-14_175011/', test_set_id)
    db3D_csph1Dk16_coarsehist_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_CSPH1D/k16_CoarseHist/loss-kldiv_tv-0.0/2022-05-15_103103/', test_set_id)
    db3D_csph1Dk32_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D/k32_HybridGrayFourier/loss-kldiv_tv-0.0/2022-05-20_194601/', test_set_id)
    db3D_csph1Dk32_truncfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test_csph/DDFN_C64B10_CSPH1D/k32_TruncFourier/loss-kldiv_tv-0.0/2022-05-26_075946/', test_set_id)
    db3D_csph1Dk64_grayfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_CSPH1D/k64_HybridGrayFourier/loss-kldiv_tv-0.0/2022-05-18_050753/', test_set_id)
    db3D_csph1Dk64_truncfour_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_CSPH1D/k64_TruncFourier/loss-kldiv_tv-0.0/2022-05-17_121426/', test_set_id)
    db3D_csph1Dk64_coarsehist_tv0_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_CSPH1D/k64_CoarseHist/loss-kldiv_tv-0.0/2022-05-17_224020/', test_set_id)
    db2D_d2d2hist01Inputs_B12_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth2Hist_01Inputs/B-12_MS-8/2022-05-02_214727/', test_set_id)
    db2D_p2d_B16_7freqs_tv1m5_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test/DDFN2D_Phasor2Depth7Freqs/B-16_MS-8/loss-L1_tv-1e-05/2022-05-05_122615_7freqs/', test_set_id)
    db2D_p2d_B16_allfreqs_tv1m5_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/test/DDFN2D_Phasor2Depth/B-16_MS-8/loss-L1_tv-1e-05/2022-05-18_042156/', test_set_id)
    db2D_d2d01Inputs_B12_tv1m3_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/loss-L1_tv-0.001/2022-05-03_183044/', test_set_id)
    db2D_d2d01Inputs_B12_tv1m4_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/loss-L1_tv-0.0001/2022-05-03_183002/', test_set_id)
    db2D_d2d01Inputs_B12_tv3m5_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/loss-L1_tv-3e-05/2022-05-03_185303/', test_set_id)
    db2D_d2d01Inputs_B12_tv1m5_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/debug/2022-04-27_104532/', test_set_id)
    db2D_d2d01Inputs_B12_tv1m10_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-12_MS-8/loss-L1_tv-1e-10/2022-05-03_183128/', test_set_id)
    db2D_d2d01Inputs_B16_tv1m5_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-16_MS-8/2022-05-01_172344/', test_set_id)
    db2D_d2d01Inputs_B22_tv1m5_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-22_MS-8/loss-L1_tv-1e-05/2022-05-20_103921', test_set_id)
    db2D_d2d01Inputs_B24_tv1m5_result_dirpath = os.path.join('outputs/nyuv2_64x64x1024_80ps/debug/DDFN2D_Depth2Depth_01Inputs/B-24_MS-8/2022-04-28_163253/', test_set_id)
    gt_data_dirpath = 'data_gener/TestData/middlebury/processed/SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'

    metrics_all = {}
    metrics_all['gt'] = {};metrics_all['gt']['rmse'] = [];metrics_all['gt']['mae'] = [];metrics_all['gt']['mse'] = [];
    metrics_all['lmf'] = {};metrics_all['lmf']['rmse'] = [];metrics_all['lmf']['mae'] = [];metrics_all['lmf']['mse'] = [];
    metrics_all['argmax'] = {};metrics_all['argmax']['rmse'] = [];metrics_all['argmax']['mae'] = [];metrics_all['argmax']['mse'] = [];
    metrics_all['compressive'] = {};metrics_all['compressive']['rmse'] = [];metrics_all['compressive']['mae'] = [];metrics_all['compressive']['mse'] = [];
    metrics_all['db3D_nl'] = {};metrics_all['db3D_nl']['rmse'] = [];metrics_all['db3D_nl']['mae'] = [];metrics_all['db3D_nl']['mse'] = [];
    metrics_all['db3D'] = {};metrics_all['db3D']['rmse'] = [];metrics_all['db3D']['mae'] = [];metrics_all['db3D']['mse'] = [];
    metrics_all['db3D_nl_d2d'] = {};metrics_all['db3D_nl_d2d']['rmse'] = [];metrics_all['db3D_nl_d2d']['mae'] = [];metrics_all['db3D_nl_d2d']['mse'] = [];
    metrics_all['db3D_d2d'] = {};metrics_all['db3D_d2d']['rmse'] = [];metrics_all['db3D_d2d']['mae'] = [];metrics_all['db3D_d2d']['mse'] = [];
    metrics_all['db3D_d2d_tv0'] = {};metrics_all['db3D_d2d_tv0']['rmse'] = [];metrics_all['db3D_d2d_tv0']['mae'] = [];metrics_all['db3D_d2d_tv0']['mse'] = [];
    metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0'] = {};metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0']['mae'] = [];metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csphk32_separable4x4x512_opt_grayfour_tv0'] = {};metrics_all['db3D_csphk32_separable4x4x512_opt_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csphk32_separable4x4x512_opt_grayfour_tv0']['mae'] = [];metrics_all['db3D_csphk32_separable4x4x512_opt_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csphk16_separable4x4x256_opt_grayfour_tv0'] = {};metrics_all['db3D_csphk16_separable4x4x256_opt_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csphk16_separable4x4x256_opt_grayfour_tv0']['mae'] = [];metrics_all['db3D_csphk16_separable4x4x256_opt_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csphk8_separable4x4x128_opt_grayfour_tv0'] = {};metrics_all['db3D_csphk8_separable4x4x128_opt_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csphk8_separable4x4x128_opt_grayfour_tv0']['mae'] = [];metrics_all['db3D_csphk8_separable4x4x128_opt_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csphk64_separable2x2x1024_opt_grayfour_tv0'] = {};metrics_all['db3D_csphk64_separable2x2x1024_opt_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csphk64_separable2x2x1024_opt_grayfour_tv0']['mae'] = [];metrics_all['db3D_csphk64_separable2x2x1024_opt_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0'] = {};metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0']['rmse'] = [];metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0']['mae'] = [];metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0']['mse'] = [];
    metrics_all['db3D_csphk32_separable2x2x1024_opt_grayfour_tv0'] = {};metrics_all['db3D_csphk32_separable2x2x1024_opt_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csphk32_separable2x2x1024_opt_grayfour_tv0']['mae'] = [];metrics_all['db3D_csphk32_separable2x2x1024_opt_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0'] = {};metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0']['rmse'] = [];metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0']['mae'] = [];metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0']['mse'] = [];
    metrics_all['db3D_csphk64_separable4x4x512_opt_truncfour_tv0'] = {};metrics_all['db3D_csphk64_separable4x4x512_opt_truncfour_tv0']['rmse'] = [];metrics_all['db3D_csphk64_separable4x4x512_opt_truncfour_tv0']['mae'] = [];metrics_all['db3D_csphk64_separable4x4x512_opt_truncfour_tv0']['mse'] = [];
    metrics_all['db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0'] = {};metrics_all['db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0']['rmse'] = [];metrics_all['db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0']['mae'] = [];metrics_all['db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0']['mse'] = [];
    metrics_all['db3D_csph1D2Dk32down2_grayfour_tv0'] = {};metrics_all['db3D_csph1D2Dk32down2_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csph1D2Dk32down2_grayfour_tv0']['mae'] = [];metrics_all['db3D_csph1D2Dk32down2_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csph1D2Dk64down2_grayfour_tv0'] = {};metrics_all['db3D_csph1D2Dk64down2_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csph1D2Dk64down2_grayfour_tv0']['mae'] = [];metrics_all['db3D_csph1D2Dk64down2_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csph1D2Dk128down2_grayfour_tv0'] = {};metrics_all['db3D_csph1D2Dk128down2_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csph1D2Dk128down2_grayfour_tv0']['mae'] = [];metrics_all['db3D_csph1D2Dk128down2_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csph1D2Dk128down2upzncc_grayfour_tv0'] = {};metrics_all['db3D_csph1D2Dk128down2upzncc_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csph1D2Dk128down2upzncc_grayfour_tv0']['mae'] = [];metrics_all['db3D_csph1D2Dk128down2upzncc_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csph1Dk8_gray_tv0'] = {};metrics_all['db3D_csph1Dk8_gray_tv0']['rmse'] = [];metrics_all['db3D_csph1Dk8_gray_tv0']['mae'] = [];metrics_all['db3D_csph1Dk8_gray_tv0']['mse'] = [];
    metrics_all['db3D_csph1Dk8_grayfour_tv0'] = {};metrics_all['db3D_csph1Dk8_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csph1Dk8_grayfour_tv0']['mae'] = [];metrics_all['db3D_csph1Dk8_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csph1Dk16_grayfour_tv0'] = {};metrics_all['db3D_csph1Dk16_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csph1Dk16_grayfour_tv0']['mae'] = [];metrics_all['db3D_csph1Dk16_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csph1Dk16_truncfour_tv0'] = {};metrics_all['db3D_csph1Dk16_truncfour_tv0']['rmse'] = [];metrics_all['db3D_csph1Dk16_truncfour_tv0']['mae'] = [];metrics_all['db3D_csph1Dk16_truncfour_tv0']['mse'] = [];
    metrics_all['db3D_csph1Dk16_coarsehist_tv0'] = {};metrics_all['db3D_csph1Dk16_coarsehist_tv0']['rmse'] = [];metrics_all['db3D_csph1Dk16_coarsehist_tv0']['mae'] = [];metrics_all['db3D_csph1Dk16_coarsehist_tv0']['mse'] = [];
    metrics_all['db3D_csph1Dk32_grayfour_tv0'] = {};metrics_all['db3D_csph1Dk32_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csph1Dk32_grayfour_tv0']['mae'] = [];metrics_all['db3D_csph1Dk32_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csph1Dk32_truncfour_tv0'] = {};metrics_all['db3D_csph1Dk32_truncfour_tv0']['rmse'] = [];metrics_all['db3D_csph1Dk32_truncfour_tv0']['mae'] = [];metrics_all['db3D_csph1Dk32_truncfour_tv0']['mse'] = [];
    metrics_all['db3D_csph1Dk64_grayfour_tv0'] = {};metrics_all['db3D_csph1Dk64_grayfour_tv0']['rmse'] = [];metrics_all['db3D_csph1Dk64_grayfour_tv0']['mae'] = [];metrics_all['db3D_csph1Dk64_grayfour_tv0']['mse'] = [];
    metrics_all['db3D_csph1Dk64_truncfour_tv0'] = {};metrics_all['db3D_csph1Dk64_truncfour_tv0']['rmse'] = [];metrics_all['db3D_csph1Dk64_truncfour_tv0']['mae'] = [];metrics_all['db3D_csph1Dk64_truncfour_tv0']['mse'] = [];
    metrics_all['db3D_csph1Dk64_coarsehist_tv0'] = {};metrics_all['db3D_csph1Dk64_coarsehist_tv0']['rmse'] = [];metrics_all['db3D_csph1Dk64_coarsehist_tv0']['mae'] = [];metrics_all['db3D_csph1Dk64_coarsehist_tv0']['mse'] = [];
    metrics_all['db2D_p2d_B16_7freqs_tv1m5'] = {};metrics_all['db2D_p2d_B16_7freqs_tv1m5']['rmse'] = [];metrics_all['db2D_p2d_B16_7freqs_tv1m5']['mae'] = [];metrics_all['db2D_p2d_B16_7freqs_tv1m5']['mse'] = [];
    metrics_all['db2D_p2d_B16_allfreqs_tv1m5'] = {};metrics_all['db2D_p2d_B16_allfreqs_tv1m5']['rmse'] = [];metrics_all['db2D_p2d_B16_allfreqs_tv1m5']['mae'] = [];metrics_all['db2D_p2d_B16_allfreqs_tv1m5']['mse'] = [];
    metrics_all['db2D_d2d2hist01Inputs_B12'] = {};metrics_all['db2D_d2d2hist01Inputs_B12']['rmse'] = [];metrics_all['db2D_d2d2hist01Inputs_B12']['mae'] = [];metrics_all['db2D_d2d2hist01Inputs_B12']['mse'] = [];
    metrics_all['db2D_d2d01Inputs_B12_tv1m3'] = {};metrics_all['db2D_d2d01Inputs_B12_tv1m3']['rmse'] = [];metrics_all['db2D_d2d01Inputs_B12_tv1m3']['mae'] = [];metrics_all['db2D_d2d01Inputs_B12_tv1m3']['mse'] = [];
    metrics_all['db2D_d2d01Inputs_B12_tv1m4'] = {};metrics_all['db2D_d2d01Inputs_B12_tv1m4']['rmse'] = [];metrics_all['db2D_d2d01Inputs_B12_tv1m4']['mae'] = [];metrics_all['db2D_d2d01Inputs_B12_tv1m4']['mse'] = [];
    metrics_all['db2D_d2d01Inputs_B12_tv3m5'] = {};metrics_all['db2D_d2d01Inputs_B12_tv3m5']['rmse'] = [];metrics_all['db2D_d2d01Inputs_B12_tv3m5']['mae'] = [];metrics_all['db2D_d2d01Inputs_B12_tv3m5']['mse'] = [];
    metrics_all['db2D_d2d01Inputs_B12_tv1m5'] = {};metrics_all['db2D_d2d01Inputs_B12_tv1m5']['rmse'] = [];metrics_all['db2D_d2d01Inputs_B12_tv1m5']['mae'] = [];metrics_all['db2D_d2d01Inputs_B12_tv1m5']['mse'] = [];
    metrics_all['db2D_d2d01Inputs_B12_tv1m10'] = {};metrics_all['db2D_d2d01Inputs_B12_tv1m10']['rmse'] = [];metrics_all['db2D_d2d01Inputs_B12_tv1m10']['mae'] = [];metrics_all['db2D_d2d01Inputs_B12_tv1m10']['mse'] = [];
    metrics_all['db2D_d2d01Inputs_B16_tv1m5'] = {};metrics_all['db2D_d2d01Inputs_B16_tv1m5']['rmse'] = [];metrics_all['db2D_d2d01Inputs_B16_tv1m5']['mae'] = [];metrics_all['db2D_d2d01Inputs_B16_tv1m5']['mse'] = [];
    metrics_all['db2D_d2d01Inputs_B22_tv1m5'] = {};metrics_all['db2D_d2d01Inputs_B22_tv1m5']['rmse'] = [];metrics_all['db2D_d2d01Inputs_B22_tv1m5']['mae'] = [];metrics_all['db2D_d2d01Inputs_B22_tv1m5']['mse'] = [];
    metrics_all['db2D_d2d01Inputs_B24_tv1m5'] = {};metrics_all['db2D_d2d01Inputs_B24_tv1m5']['rmse'] = [];metrics_all['db2D_d2d01Inputs_B24_tv1m5']['mae'] = [];metrics_all['db2D_d2d01Inputs_B24_tv1m5']['mse'] = [];

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
            (compressive_depths) = get_model_depths(model_result_dirpath=compressive_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_nl_depths) = get_model_depths(model_result_dirpath=db3D_nl_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_depths) = get_model_depths(model_result_dirpath=db3D_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_nl_d2d_depths) = get_model_depths(model_result_dirpath=db3D_nl_d2d_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_d2d_depths) = get_model_depths(model_result_dirpath=db3D_d2d_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_d2d_tv0_depths) = get_model_depths(model_result_dirpath=db3D_d2d_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csphk32_separable4x4x512_opt_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csphk32_separable4x4x512_opt_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csphk16_separable4x4x256_opt_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csphk16_separable4x4x256_opt_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csphk8_separable4x4x128_opt_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csphk8_separable4x4x128_opt_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csphk64_separable4x4x512_opt_truncfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csphk64_separable4x4x512_opt_truncfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1D2Dk32down2_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1D2Dk32down2_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1D2Dk64down2_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1D2Dk64down2_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1D2Dk128down2_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1D2Dk128down2_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1D2Dk128down2upzncc_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1D2Dk128down2upzncc_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1Dk8_gray_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1Dk8_gray_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1Dk8_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1Dk8_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1Dk16_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1Dk16_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1Dk16_truncfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1Dk16_truncfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1Dk16_coarsehist_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1Dk16_coarsehist_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1Dk32_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1Dk32_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1Dk32_truncfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1Dk32_truncfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1Dk64_grayfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1Dk64_grayfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1Dk64_truncfour_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1Dk64_truncfour_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db3D_csph1Dk64_coarsehist_tv0_depths) = get_model_depths(model_result_dirpath=db3D_csph1Dk64_coarsehist_tv0_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_p2d_B16_7freqs_tv1m5_depths) = get_model_depths(model_result_dirpath=db2D_p2d_B16_7freqs_tv1m5_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_p2d_B16_allfreqs_tv1m5_depths) = get_model_depths(model_result_dirpath=db2D_p2d_B16_allfreqs_tv1m5_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d2hist01Inputs_B12_depths) = get_model_depths(model_result_dirpath=db2D_d2d2hist01Inputs_B12_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B12_tv1m3_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B12_tv1m3_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B12_tv1m4_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B12_tv1m4_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B12_tv3m5_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B12_tv3m5_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B12_tv1m5_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B12_tv1m5_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B12_tv1m10_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B12_tv1m10_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B16_tv1m5_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B16_tv1m5_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B22_tv1m5_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B22_tv1m5_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)
            (db2D_d2d01Inputs_B24_tv1m5_depths) = get_model_depths(model_result_dirpath=db2D_d2d01Inputs_B24_tv1m5_result_dirpath, scene_id=curr_scene_id, sbr_params=curr_sbr_params, num_bins=num_bins, tau=tau)

            ## Compute RMSE and MAE
            (gt_rmse, gt_mse, gt_mae, gt_abs_errs, gt_perc_errs) = compute_error_metrics(gt_depths, gt_depths)
            (lmf_rmse, lmf_mse, lmf_mae, lmf_abs_errs, lmf_perc_errs) = compute_error_metrics(gt_depths, lmf_depths)
            (argmax_rmse, argmax_mse, argmax_mae, argmax_abs_errs, argmax_perc_errs) = compute_error_metrics(gt_depths, argmax_depths)
            (compressive_rmse, compressive_mse, compressive_mae, compressive_abs_errs, compressive_perc_errs) = compute_error_metrics(gt_depths, compressive_depths)
            (db3D_nl_rmse, db3D_nl_mse, db3D_nl_mae, db3D_nl_abs_errs, db3D_nl_perc_errs) = compute_error_metrics(gt_depths, db3D_nl_depths)
            (db3D_rmse, db3D_mse, db3D_mae, db3D_abs_errs, db3D_perc_errs) = compute_error_metrics(gt_depths, db3D_depths)
            (db3D_nl_d2d_rmse, db3D_nl_d2d_mse, db3D_nl_d2d_mae, db3D_nl_d2d_abs_errs, db3D_nl_d2d_perc_errs) = compute_error_metrics(gt_depths, db3D_nl_d2d_depths)
            (db3D_d2d_rmse, db3D_d2d_mse, db3D_d2d_mae, db3D_d2d_abs_errs, db3D_d2d_perc_errs) = compute_error_metrics(gt_depths, db3D_d2d_depths)
            (db3D_d2d_tv0_rmse, db3D_d2d_tv0_mse, db3D_d2d_tv0_mae, db3D_d2d_tv0_abs_errs, db3D_d2d_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_d2d_tv0_depths)
            (db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_rmse, db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_mse, db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_mae, db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_abs_errs, db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_depths)
            (db3D_csphk32_separable4x4x512_opt_grayfour_tv0_rmse, db3D_csphk32_separable4x4x512_opt_grayfour_tv0_mse, db3D_csphk32_separable4x4x512_opt_grayfour_tv0_mae, db3D_csphk32_separable4x4x512_opt_grayfour_tv0_abs_errs, db3D_csphk32_separable4x4x512_opt_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csphk32_separable4x4x512_opt_grayfour_tv0_depths)
            (db3D_csphk16_separable4x4x256_opt_grayfour_tv0_rmse, db3D_csphk16_separable4x4x256_opt_grayfour_tv0_mse, db3D_csphk16_separable4x4x256_opt_grayfour_tv0_mae, db3D_csphk16_separable4x4x256_opt_grayfour_tv0_abs_errs, db3D_csphk16_separable4x4x256_opt_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csphk16_separable4x4x256_opt_grayfour_tv0_depths)
            (db3D_csphk8_separable4x4x128_opt_grayfour_tv0_rmse, db3D_csphk8_separable4x4x128_opt_grayfour_tv0_mse, db3D_csphk8_separable4x4x128_opt_grayfour_tv0_mae, db3D_csphk8_separable4x4x128_opt_grayfour_tv0_abs_errs, db3D_csphk8_separable4x4x128_opt_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csphk8_separable4x4x128_opt_grayfour_tv0_depths)
            (db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_rmse, db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_mse, db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_mae, db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_abs_errs, db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_depths)
            (db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_rmse, db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_mse, db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_mae, db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_abs_errs, db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_depths)
            (db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_rmse, db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_mse, db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_mae, db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_abs_errs, db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_depths)
            (db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_rmse, db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_mse, db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_mae, db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_abs_errs, db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_depths)
            (db3D_csphk64_separable4x4x512_opt_truncfour_tv0_rmse, db3D_csphk64_separable4x4x512_opt_truncfour_tv0_mse, db3D_csphk64_separable4x4x512_opt_truncfour_tv0_mae, db3D_csphk64_separable4x4x512_opt_truncfour_tv0_abs_errs, db3D_csphk64_separable4x4x512_opt_truncfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csphk64_separable4x4x512_opt_truncfour_tv0_depths)
            (db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_rmse, db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_mse, db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_mae, db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_abs_errs, db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_depths)
            (db3D_csph1D2Dk32down2_grayfour_tv0_rmse, db3D_csph1D2Dk32down2_grayfour_tv0_mse, db3D_csph1D2Dk32down2_grayfour_tv0_mae, db3D_csph1D2Dk32down2_grayfour_tv0_abs_errs, db3D_csph1D2Dk32down2_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1D2Dk32down2_grayfour_tv0_depths)
            (db3D_csph1D2Dk64down2_grayfour_tv0_rmse, db3D_csph1D2Dk64down2_grayfour_tv0_mse, db3D_csph1D2Dk64down2_grayfour_tv0_mae, db3D_csph1D2Dk64down2_grayfour_tv0_abs_errs, db3D_csph1D2Dk64down2_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1D2Dk64down2_grayfour_tv0_depths)
            (db3D_csph1D2Dk128down2_grayfour_tv0_rmse, db3D_csph1D2Dk128down2_grayfour_tv0_mse, db3D_csph1D2Dk128down2_grayfour_tv0_mae, db3D_csph1D2Dk128down2_grayfour_tv0_abs_errs, db3D_csph1D2Dk128down2_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1D2Dk128down2_grayfour_tv0_depths)
            (db3D_csph1D2Dk128down2upzncc_grayfour_tv0_rmse, db3D_csph1D2Dk128down2upzncc_grayfour_tv0_mse, db3D_csph1D2Dk128down2upzncc_grayfour_tv0_mae, db3D_csph1D2Dk128down2upzncc_grayfour_tv0_abs_errs, db3D_csph1D2Dk128down2upzncc_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1D2Dk128down2upzncc_grayfour_tv0_depths)
            (db3D_csph1Dk8_gray_tv0_rmse, db3D_csph1Dk8_gray_tv0_mse, db3D_csph1Dk8_gray_tv0_mae, db3D_csph1Dk8_gray_tv0_abs_errs, db3D_csph1Dk8_gray_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1Dk8_gray_tv0_depths)
            (db3D_csph1Dk8_grayfour_tv0_rmse, db3D_csph1Dk8_grayfour_tv0_mse, db3D_csph1Dk8_grayfour_tv0_mae, db3D_csph1Dk8_grayfour_tv0_abs_errs, db3D_csph1Dk8_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1Dk8_grayfour_tv0_depths)
            (db3D_csph1Dk16_grayfour_tv0_rmse, db3D_csph1Dk16_grayfour_tv0_mse, db3D_csph1Dk16_grayfour_tv0_mae, db3D_csph1Dk16_grayfour_tv0_abs_errs, db3D_csph1Dk16_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1Dk16_grayfour_tv0_depths)
            (db3D_csph1Dk16_truncfour_tv0_rmse, db3D_csph1Dk16_truncfour_tv0_mse, db3D_csph1Dk16_truncfour_tv0_mae, db3D_csph1Dk16_truncfour_tv0_abs_errs, db3D_csph1Dk16_truncfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1Dk16_truncfour_tv0_depths)
            (db3D_csph1Dk16_coarsehist_tv0_rmse, db3D_csph1Dk16_coarsehist_tv0_mse, db3D_csph1Dk16_coarsehist_tv0_mae, db3D_csph1Dk16_coarsehist_tv0_abs_errs, db3D_csph1Dk16_coarsehist_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1Dk16_coarsehist_tv0_depths)
            (db3D_csph1Dk32_grayfour_tv0_rmse, db3D_csph1Dk32_grayfour_tv0_mse, db3D_csph1Dk32_grayfour_tv0_mae, db3D_csph1Dk32_grayfour_tv0_abs_errs, db3D_csph1Dk32_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1Dk32_grayfour_tv0_depths)
            (db3D_csph1Dk32_truncfour_tv0_rmse, db3D_csph1Dk32_truncfour_tv0_mse, db3D_csph1Dk32_truncfour_tv0_mae, db3D_csph1Dk32_truncfour_tv0_abs_errs, db3D_csph1Dk32_truncfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1Dk32_truncfour_tv0_depths)
            (db3D_csph1Dk64_grayfour_tv0_rmse, db3D_csph1Dk64_grayfour_tv0_mse, db3D_csph1Dk64_grayfour_tv0_mae, db3D_csph1Dk64_grayfour_tv0_abs_errs, db3D_csph1Dk64_grayfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1Dk64_grayfour_tv0_depths)
            (db3D_csph1Dk64_truncfour_tv0_rmse, db3D_csph1Dk64_truncfour_tv0_mse, db3D_csph1Dk64_truncfour_tv0_mae, db3D_csph1Dk64_truncfour_tv0_abs_errs, db3D_csph1Dk64_truncfour_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1Dk64_truncfour_tv0_depths)
            (db3D_csph1Dk64_coarsehist_tv0_rmse, db3D_csph1Dk64_coarsehist_tv0_mse, db3D_csph1Dk64_coarsehist_tv0_mae, db3D_csph1Dk64_coarsehist_tv0_abs_errs, db3D_csph1Dk64_coarsehist_tv0_perc_errs) = compute_error_metrics(gt_depths, db3D_csph1Dk64_coarsehist_tv0_depths)
            (db2D_p2d_B16_7freqs_tv1m5_rmse, db2D_p2d_B16_7freqs_tv1m5_mse, db2D_p2d_B16_7freqs_tv1m5_mae, db2D_p2d_B16_7freqs_tv1m5_abs_errs, db2D_p2d_B16_7freqs_tv1m5_perc_errs) = compute_error_metrics(gt_depths, db2D_p2d_B16_7freqs_tv1m5_depths)
            (db2D_p2d_B16_allfreqs_tv1m5_rmse, db2D_p2d_B16_allfreqs_tv1m5_mse, db2D_p2d_B16_allfreqs_tv1m5_mae, db2D_p2d_B16_allfreqs_tv1m5_abs_errs, db2D_p2d_B16_allfreqs_tv1m5_perc_errs) = compute_error_metrics(gt_depths, db2D_p2d_B16_allfreqs_tv1m5_depths)
            (db2D_d2d2hist01Inputs_B12_rmse, db2D_d2d2hist01Inputs_B12_mse, db2D_d2d2hist01Inputs_B12_mae, db2D_d2d2hist01Inputs_B12_abs_errs, db2D_d2d2hist01Inputs_B12_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d2hist01Inputs_B12_depths)
            (db2D_d2d01Inputs_B12_tv1m3_rmse, db2D_d2d01Inputs_B12_tv1m3_mse, db2D_d2d01Inputs_B12_tv1m3_mae, db2D_d2d01Inputs_B12_tv1m3_abs_errs, db2D_d2d01Inputs_B12_tv1m3_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B12_tv1m3_depths)
            (db2D_d2d01Inputs_B12_tv1m4_rmse, db2D_d2d01Inputs_B12_tv1m4_mse, db2D_d2d01Inputs_B12_tv1m4_mae, db2D_d2d01Inputs_B12_tv1m4_abs_errs, db2D_d2d01Inputs_B12_tv1m4_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B12_tv1m4_depths)
            (db2D_d2d01Inputs_B12_tv3m5_rmse, db2D_d2d01Inputs_B12_tv3m5_mse, db2D_d2d01Inputs_B12_tv3m5_mae, db2D_d2d01Inputs_B12_tv3m5_abs_errs, db2D_d2d01Inputs_B12_tv3m5_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B12_tv3m5_depths)
            (db2D_d2d01Inputs_B12_tv1m5_rmse, db2D_d2d01Inputs_B12_tv1m5_mse, db2D_d2d01Inputs_B12_tv1m5_mae, db2D_d2d01Inputs_B12_tv1m5_abs_errs, db2D_d2d01Inputs_B12_tv1m5_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B12_tv1m5_depths)
            (db2D_d2d01Inputs_B12_tv1m10_rmse, db2D_d2d01Inputs_B12_tv1m10_mse, db2D_d2d01Inputs_B12_tv1m10_mae, db2D_d2d01Inputs_B12_tv1m10_abs_errs, db2D_d2d01Inputs_B12_tv1m10_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B12_tv1m10_depths)
            (db2D_d2d01Inputs_B16_tv1m5_rmse, db2D_d2d01Inputs_B16_tv1m5_mse, db2D_d2d01Inputs_B16_tv1m5_mae, db2D_d2d01Inputs_B16_tv1m5_abs_errs, db2D_d2d01Inputs_B16_tv1m5_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B16_tv1m5_depths)
            (db2D_d2d01Inputs_B22_tv1m5_rmse, db2D_d2d01Inputs_B22_tv1m5_mse, db2D_d2d01Inputs_B22_tv1m5_mae, db2D_d2d01Inputs_B22_tv1m5_abs_errs, db2D_d2d01Inputs_B22_tv1m5_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B22_tv1m5_depths)
            (db2D_d2d01Inputs_B24_tv1m5_rmse, db2D_d2d01Inputs_B24_tv1m5_mse, db2D_d2d01Inputs_B24_tv1m5_mae, db2D_d2d01Inputs_B24_tv1m5_abs_errs, db2D_d2d01Inputs_B24_tv1m5_perc_errs) = compute_error_metrics(gt_depths, db2D_d2d01Inputs_B24_tv1m5_depths)


            metrics_all['gt']['rmse'].append(gt_rmse);metrics_all['gt']['mae'].append(gt_mae);metrics_all['gt']['mse'].append(gt_mse)
            metrics_all['lmf']['rmse'].append(lmf_rmse);metrics_all['lmf']['mae'].append(lmf_mae);metrics_all['lmf']['mse'].append(lmf_mse)
            metrics_all['argmax']['rmse'].append(argmax_rmse);metrics_all['argmax']['mae'].append(argmax_mae);metrics_all['argmax']['mse'].append(argmax_mse)
            metrics_all['compressive']['rmse'].append(compressive_rmse);metrics_all['compressive']['mae'].append(compressive_mae);metrics_all['compressive']['mse'].append(compressive_mse)
            metrics_all['db3D_nl']['rmse'].append(db3D_nl_rmse);metrics_all['db3D_nl']['mae'].append(db3D_nl_mae);metrics_all['db3D_nl']['mse'].append(db3D_nl_mse)
            metrics_all['db3D']['rmse'].append(db3D_rmse);metrics_all['db3D']['mae'].append(db3D_mae);metrics_all['db3D']['mse'].append(db3D_mse)
            metrics_all['db3D_nl_d2d']['rmse'].append(db3D_nl_d2d_rmse);metrics_all['db3D_nl_d2d']['mae'].append(db3D_nl_d2d_mae);metrics_all['db3D_nl_d2d']['mse'].append(db3D_nl_d2d_mse)
            metrics_all['db3D_d2d']['rmse'].append(db3D_d2d_rmse);metrics_all['db3D_d2d']['mae'].append(db3D_d2d_mae);metrics_all['db3D_d2d']['mse'].append(db3D_d2d_mse)
            metrics_all['db3D_d2d_tv0']['rmse'].append(db3D_d2d_tv0_rmse);metrics_all['db3D_d2d_tv0']['mae'].append(db3D_d2d_tv0_mae);metrics_all['db3D_d2d_tv0']['mse'].append(db3D_d2d_tv0_mse)
            metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0']['rmse'].append(db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_rmse);metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0']['mae'].append(db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_mae);metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0']['mse'].append(db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_mse)
            metrics_all['db3D_csphk32_separable4x4x512_opt_grayfour_tv0']['rmse'].append(db3D_csphk32_separable4x4x512_opt_grayfour_tv0_rmse);metrics_all['db3D_csphk32_separable4x4x512_opt_grayfour_tv0']['mae'].append(db3D_csphk32_separable4x4x512_opt_grayfour_tv0_mae);metrics_all['db3D_csphk32_separable4x4x512_opt_grayfour_tv0']['mse'].append(db3D_csphk32_separable4x4x512_opt_grayfour_tv0_mse)
            metrics_all['db3D_csphk16_separable4x4x256_opt_grayfour_tv0']['rmse'].append(db3D_csphk16_separable4x4x256_opt_grayfour_tv0_rmse);metrics_all['db3D_csphk16_separable4x4x256_opt_grayfour_tv0']['mae'].append(db3D_csphk16_separable4x4x256_opt_grayfour_tv0_mae);metrics_all['db3D_csphk16_separable4x4x256_opt_grayfour_tv0']['mse'].append(db3D_csphk16_separable4x4x256_opt_grayfour_tv0_mse)
            metrics_all['db3D_csphk8_separable4x4x128_opt_grayfour_tv0']['rmse'].append(db3D_csphk8_separable4x4x128_opt_grayfour_tv0_rmse);metrics_all['db3D_csphk8_separable4x4x128_opt_grayfour_tv0']['mae'].append(db3D_csphk8_separable4x4x128_opt_grayfour_tv0_mae);metrics_all['db3D_csphk8_separable4x4x128_opt_grayfour_tv0']['mse'].append(db3D_csphk8_separable4x4x128_opt_grayfour_tv0_mse)
            metrics_all['db3D_csphk64_separable2x2x1024_opt_grayfour_tv0']['rmse'].append(db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_rmse);metrics_all['db3D_csphk64_separable2x2x1024_opt_grayfour_tv0']['mae'].append(db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_mae);metrics_all['db3D_csphk64_separable2x2x1024_opt_grayfour_tv0']['mse'].append(db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_mse)
            metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0']['rmse'].append(db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_rmse);metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0']['mae'].append(db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_mae);metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0']['mse'].append(db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_mse)
            metrics_all['db3D_csphk32_separable2x2x1024_opt_grayfour_tv0']['rmse'].append(db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_rmse);metrics_all['db3D_csphk32_separable2x2x1024_opt_grayfour_tv0']['mae'].append(db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_mae);metrics_all['db3D_csphk32_separable2x2x1024_opt_grayfour_tv0']['mse'].append(db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_mse)
            metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0']['rmse'].append(db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_rmse);metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0']['mae'].append(db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_mae);metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0']['mse'].append(db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_mse)
            metrics_all['db3D_csphk64_separable4x4x512_opt_truncfour_tv0']['rmse'].append(db3D_csphk64_separable4x4x512_opt_truncfour_tv0_rmse);metrics_all['db3D_csphk64_separable4x4x512_opt_truncfour_tv0']['mae'].append(db3D_csphk64_separable4x4x512_opt_truncfour_tv0_mae);metrics_all['db3D_csphk64_separable4x4x512_opt_truncfour_tv0']['mse'].append(db3D_csphk64_separable4x4x512_opt_truncfour_tv0_mse)
            metrics_all['db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0']['rmse'].append(db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_rmse);metrics_all['db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0']['mae'].append(db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_mae);metrics_all['db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0']['mse'].append(db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_mse)
            metrics_all['db3D_csph1D2Dk32down2_grayfour_tv0']['rmse'].append(db3D_csph1D2Dk32down2_grayfour_tv0_rmse);metrics_all['db3D_csph1D2Dk32down2_grayfour_tv0']['mae'].append(db3D_csph1D2Dk32down2_grayfour_tv0_mae);metrics_all['db3D_csph1D2Dk32down2_grayfour_tv0']['mse'].append(db3D_csph1D2Dk32down2_grayfour_tv0_mse)
            metrics_all['db3D_csph1D2Dk64down2_grayfour_tv0']['rmse'].append(db3D_csph1D2Dk64down2_grayfour_tv0_rmse);metrics_all['db3D_csph1D2Dk64down2_grayfour_tv0']['mae'].append(db3D_csph1D2Dk64down2_grayfour_tv0_mae);metrics_all['db3D_csph1D2Dk64down2_grayfour_tv0']['mse'].append(db3D_csph1D2Dk64down2_grayfour_tv0_mse)
            metrics_all['db3D_csph1D2Dk128down2_grayfour_tv0']['rmse'].append(db3D_csph1D2Dk128down2_grayfour_tv0_rmse);metrics_all['db3D_csph1D2Dk128down2_grayfour_tv0']['mae'].append(db3D_csph1D2Dk128down2_grayfour_tv0_mae);metrics_all['db3D_csph1D2Dk128down2_grayfour_tv0']['mse'].append(db3D_csph1D2Dk128down2_grayfour_tv0_mse)
            metrics_all['db3D_csph1D2Dk128down2upzncc_grayfour_tv0']['rmse'].append(db3D_csph1D2Dk128down2upzncc_grayfour_tv0_rmse);metrics_all['db3D_csph1D2Dk128down2upzncc_grayfour_tv0']['mae'].append(db3D_csph1D2Dk128down2upzncc_grayfour_tv0_mae);metrics_all['db3D_csph1D2Dk128down2upzncc_grayfour_tv0']['mse'].append(db3D_csph1D2Dk128down2upzncc_grayfour_tv0_mse)
            metrics_all['db3D_csph1Dk8_gray_tv0']['rmse'].append(db3D_csph1Dk8_gray_tv0_rmse);metrics_all['db3D_csph1Dk8_gray_tv0']['mae'].append(db3D_csph1Dk8_gray_tv0_mae);metrics_all['db3D_csph1Dk8_gray_tv0']['mse'].append(db3D_csph1Dk8_gray_tv0_mse)
            metrics_all['db3D_csph1Dk8_grayfour_tv0']['rmse'].append(db3D_csph1Dk8_grayfour_tv0_rmse);metrics_all['db3D_csph1Dk8_grayfour_tv0']['mae'].append(db3D_csph1Dk8_grayfour_tv0_mae);metrics_all['db3D_csph1Dk8_grayfour_tv0']['mse'].append(db3D_csph1Dk8_grayfour_tv0_mse)
            metrics_all['db3D_csph1Dk16_grayfour_tv0']['rmse'].append(db3D_csph1Dk16_grayfour_tv0_rmse);metrics_all['db3D_csph1Dk16_grayfour_tv0']['mae'].append(db3D_csph1Dk16_grayfour_tv0_mae);metrics_all['db3D_csph1Dk16_grayfour_tv0']['mse'].append(db3D_csph1Dk16_grayfour_tv0_mse)
            metrics_all['db3D_csph1Dk16_truncfour_tv0']['rmse'].append(db3D_csph1Dk16_truncfour_tv0_rmse);metrics_all['db3D_csph1Dk16_truncfour_tv0']['mae'].append(db3D_csph1Dk16_truncfour_tv0_mae);metrics_all['db3D_csph1Dk16_truncfour_tv0']['mse'].append(db3D_csph1Dk16_truncfour_tv0_mse)
            metrics_all['db3D_csph1Dk16_coarsehist_tv0']['rmse'].append(db3D_csph1Dk16_coarsehist_tv0_rmse);metrics_all['db3D_csph1Dk16_coarsehist_tv0']['mae'].append(db3D_csph1Dk16_coarsehist_tv0_mae);metrics_all['db3D_csph1Dk16_coarsehist_tv0']['mse'].append(db3D_csph1Dk16_coarsehist_tv0_mse)
            metrics_all['db3D_csph1Dk32_grayfour_tv0']['rmse'].append(db3D_csph1Dk32_grayfour_tv0_rmse);metrics_all['db3D_csph1Dk32_grayfour_tv0']['mae'].append(db3D_csph1Dk32_grayfour_tv0_mae);metrics_all['db3D_csph1Dk32_grayfour_tv0']['mse'].append(db3D_csph1Dk32_grayfour_tv0_mse)
            metrics_all['db3D_csph1Dk32_truncfour_tv0']['rmse'].append(db3D_csph1Dk32_truncfour_tv0_rmse);metrics_all['db3D_csph1Dk32_truncfour_tv0']['mae'].append(db3D_csph1Dk32_truncfour_tv0_mae);metrics_all['db3D_csph1Dk32_truncfour_tv0']['mse'].append(db3D_csph1Dk32_truncfour_tv0_mse)
            metrics_all['db3D_csph1Dk64_grayfour_tv0']['rmse'].append(db3D_csph1Dk64_grayfour_tv0_rmse);metrics_all['db3D_csph1Dk64_grayfour_tv0']['mae'].append(db3D_csph1Dk64_grayfour_tv0_mae);metrics_all['db3D_csph1Dk64_grayfour_tv0']['mse'].append(db3D_csph1Dk64_grayfour_tv0_mse)
            metrics_all['db3D_csph1Dk64_truncfour_tv0']['rmse'].append(db3D_csph1Dk64_truncfour_tv0_rmse);metrics_all['db3D_csph1Dk64_truncfour_tv0']['mae'].append(db3D_csph1Dk64_truncfour_tv0_mae);metrics_all['db3D_csph1Dk64_truncfour_tv0']['mse'].append(db3D_csph1Dk64_truncfour_tv0_mse)
            metrics_all['db3D_csph1Dk64_coarsehist_tv0']['rmse'].append(db3D_csph1Dk64_coarsehist_tv0_rmse);metrics_all['db3D_csph1Dk64_coarsehist_tv0']['mae'].append(db3D_csph1Dk64_coarsehist_tv0_mae);metrics_all['db3D_csph1Dk64_coarsehist_tv0']['mse'].append(db3D_csph1Dk64_coarsehist_tv0_mse)
            metrics_all['db2D_p2d_B16_7freqs_tv1m5']['rmse'].append(db2D_p2d_B16_7freqs_tv1m5_rmse);metrics_all['db2D_p2d_B16_7freqs_tv1m5']['mae'].append(db2D_p2d_B16_7freqs_tv1m5_mae);metrics_all['db2D_p2d_B16_7freqs_tv1m5']['mse'].append(db2D_p2d_B16_7freqs_tv1m5_mse)
            metrics_all['db2D_p2d_B16_allfreqs_tv1m5']['rmse'].append(db2D_p2d_B16_allfreqs_tv1m5_rmse);metrics_all['db2D_p2d_B16_allfreqs_tv1m5']['mae'].append(db2D_p2d_B16_allfreqs_tv1m5_mae);metrics_all['db2D_p2d_B16_allfreqs_tv1m5']['mse'].append(db2D_p2d_B16_allfreqs_tv1m5_mse)
            metrics_all['db2D_d2d2hist01Inputs_B12']['rmse'].append(db2D_d2d2hist01Inputs_B12_rmse);metrics_all['db2D_d2d2hist01Inputs_B12']['mae'].append(db2D_d2d2hist01Inputs_B12_mae);metrics_all['db2D_d2d2hist01Inputs_B12']['mse'].append(db2D_d2d2hist01Inputs_B12_mse)
            metrics_all['db2D_d2d01Inputs_B12_tv1m3']['rmse'].append(db2D_d2d01Inputs_B12_tv1m3_rmse);metrics_all['db2D_d2d01Inputs_B12_tv1m3']['mae'].append(db2D_d2d01Inputs_B12_tv1m3_mae);metrics_all['db2D_d2d01Inputs_B12_tv1m3']['mse'].append(db2D_d2d01Inputs_B12_tv1m3_mse)
            metrics_all['db2D_d2d01Inputs_B12_tv1m4']['rmse'].append(db2D_d2d01Inputs_B12_tv1m4_rmse);metrics_all['db2D_d2d01Inputs_B12_tv1m4']['mae'].append(db2D_d2d01Inputs_B12_tv1m4_mae);metrics_all['db2D_d2d01Inputs_B12_tv1m4']['mse'].append(db2D_d2d01Inputs_B12_tv1m4_mse)
            metrics_all['db2D_d2d01Inputs_B12_tv3m5']['rmse'].append(db2D_d2d01Inputs_B12_tv3m5_rmse);metrics_all['db2D_d2d01Inputs_B12_tv3m5']['mae'].append(db2D_d2d01Inputs_B12_tv3m5_mae);metrics_all['db2D_d2d01Inputs_B12_tv3m5']['mse'].append(db2D_d2d01Inputs_B12_tv3m5_mse)
            metrics_all['db2D_d2d01Inputs_B12_tv1m5']['rmse'].append(db2D_d2d01Inputs_B12_tv1m5_rmse);metrics_all['db2D_d2d01Inputs_B12_tv1m5']['mae'].append(db2D_d2d01Inputs_B12_tv1m5_mae);metrics_all['db2D_d2d01Inputs_B12_tv1m5']['mse'].append(db2D_d2d01Inputs_B12_tv1m5_mse)
            metrics_all['db2D_d2d01Inputs_B12_tv1m10']['rmse'].append(db2D_d2d01Inputs_B12_tv1m10_rmse);metrics_all['db2D_d2d01Inputs_B12_tv1m10']['mae'].append(db2D_d2d01Inputs_B12_tv1m10_mae);metrics_all['db2D_d2d01Inputs_B12_tv1m10']['mse'].append(db2D_d2d01Inputs_B12_tv1m10_mse)
            metrics_all['db2D_d2d01Inputs_B16_tv1m5']['rmse'].append(db2D_d2d01Inputs_B16_tv1m5_rmse);metrics_all['db2D_d2d01Inputs_B16_tv1m5']['mae'].append(db2D_d2d01Inputs_B16_tv1m5_mae);metrics_all['db2D_d2d01Inputs_B16_tv1m5']['mse'].append(db2D_d2d01Inputs_B16_tv1m5_mse)
            metrics_all['db2D_d2d01Inputs_B22_tv1m5']['rmse'].append(db2D_d2d01Inputs_B22_tv1m5_rmse);metrics_all['db2D_d2d01Inputs_B22_tv1m5']['mae'].append(db2D_d2d01Inputs_B22_tv1m5_mae);metrics_all['db2D_d2d01Inputs_B22_tv1m5']['mse'].append(db2D_d2d01Inputs_B22_tv1m5_mse)
            metrics_all['db2D_d2d01Inputs_B24_tv1m5']['rmse'].append(db2D_d2d01Inputs_B24_tv1m5_rmse);metrics_all['db2D_d2d01Inputs_B24_tv1m5']['mae'].append(db2D_d2d01Inputs_B24_tv1m5_mae);metrics_all['db2D_d2d01Inputs_B24_tv1m5']['mse'].append(db2D_d2d01Inputs_B24_tv1m5_mse)

            min_depth = gt_depths.flatten().min()
            max_depth = gt_depths.flatten().max()
            min_err = 0
            max_err = 0.15

            if(plot_results):
                plt.clf()
                plt.suptitle("{} - SBR: {}, Signal: {} photons, Bkg: {} photons".format(scene_fname, SBR, mean_signal_photons, mean_background_photons), fontsize=20)
                plt.subplot(2,3,1)
                plt.imshow(db3D_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('db3D \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_mse, db3D_mae*100, db3D_perc_errs),fontsize=14)
                plt.colorbar()
                plt.subplot(2,3,2)
                plt.imshow(db3D_d2d_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('db3D_d2d \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_d2d_mse, db3D_d2d_mae*100, db3D_d2d_perc_errs),fontsize=14)
                plt.colorbar()
                plt.subplot(2,3,3)
                plt.imshow(db3D_csph1Dk8_grayfour_tv0_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('db3D_csph1Dk8_grayfour \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_csph1Dk8_grayfour_tv0_mse, db3D_csph1Dk8_grayfour_tv0_mae*100, db3D_csph1Dk8_grayfour_tv0_perc_errs),fontsize=14)
                # plt.imshow(db3D_csph1Dk16_grayfour_tv0_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_csph1Dk16_grayfour \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_csph1Dk16_grayfour_tv0_mse, db3D_csph1Dk16_grayfour_tv0_mae*100, db3D_csph1Dk16_grayfour_tv0_perc_errs),fontsize=14)
                plt.colorbar()
                plt.subplot(2,3,4)
                # plt.imshow(db3D_csph1D2Dk32down2_grayfour_tv0_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_csph1D2Dk32down2_grayfour \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_csph1D2Dk32down2_grayfour_tv0_mse, db3D_csph1D2Dk32down2_grayfour_tv0_mae*100, db3D_csph1D2Dk32down2_grayfour_tv0_perc_errs),fontsize=14)
                # plt.imshow(db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_csphk32_separable2x2x1024_opt_grayfour \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_mse, db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_mae*100, db3D_csphk32_separable2x2x1024_opt_grayfour_tv0_perc_errs),fontsize=14)
                # plt.imshow(db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_csphk256_separable4x4x1024_opt_truncfour \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_mse, db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_mae*100, db3D_csphk256_separable4x4x1024_opt_truncfour_tv0_perc_errs),fontsize=14)
                plt.imshow(db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('db3D_csphk64_separable4x4x1024_opt_grayfour \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_mse, db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_mae*100, db3D_csphk64_separable4x4x1024_opt_grayfour_tv0_perc_errs),fontsize=14)
                plt.colorbar()
                plt.subplot(2,3,5)
                # plt.imshow(db3D_csphk64_separable4x4x512_opt_truncfour_tv0_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_csphk64_separable4x4x512_opt_truncfour \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_csphk64_separable4x4x512_opt_truncfour_tv0_mse, db3D_csphk64_separable4x4x512_opt_truncfour_tv0_mae*100, db3D_csphk64_separable4x4x512_opt_truncfour_tv0_perc_errs),fontsize=14)
                # plt.imshow(db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_csph1DGlobal2DLocal4xDown_k128_truncfour \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_mse, db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_mae*100, db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0_perc_errs),fontsize=14)
                # plt.imshow(db3D_csph1D2Dk64down2_grayfour_tv0_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_csph1D2Dk64down2_grayfour \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_csph1D2Dk64down2_grayfour_tv0_mse, db3D_csph1D2Dk64down2_grayfour_tv0_mae*100, db3D_csph1D2Dk64down2_grayfour_tv0_perc_errs),fontsize=14)
                plt.imshow(db3D_csphk32_separable4x4x512_opt_grayfour_tv0_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('db3D_csphk32_separable4x4x512_opt_grayfour \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_csphk32_separable4x4x512_opt_grayfour_tv0_mse, db3D_csphk32_separable4x4x512_opt_grayfour_tv0_mae*100, db3D_csphk32_separable4x4x512_opt_grayfour_tv0_perc_errs),fontsize=14)
                plt.colorbar()
                plt.subplot(2,3,6)
                # plt.imshow(db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_csphk128_separable4x4x1024_opt_truncfour \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_mse, db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_mae*100, db3D_csphk128_separable4x4x1024_opt_truncfour_tv0_perc_errs),fontsize=14)
                # plt.imshow(db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_csphk64_separable2x2x1024_grayfour \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_mse, db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_mae*100, db3D_csphk64_separable2x2x1024_opt_grayfour_tv0_perc_errs),fontsize=14)
                plt.imshow(db3D_csphk8_separable4x4x128_opt_grayfour_tv0_depths, vmin=min_depth, vmax=max_depth); 
                plt.title('db3D_csphk8_separable4x4x128_opt_grayfour \n mse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_csphk8_separable4x4x128_opt_grayfour_tv0_mse, db3D_csphk8_separable4x4x128_opt_grayfour_tv0_mae*100, db3D_csphk8_separable4x4x128_opt_grayfour_tv0_perc_errs),fontsize=14)
                plt.colorbar()
                out_fname = 'depths_' + scene_fname
                plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname)

                # plt.clf()
                # plt.suptitle("{} - SBR: {}, Signal: {} photons, Bkg: {} photons".format(scene_fname, SBR, mean_signal_photons, mean_background_photons), fontsize=20)
                # plt.subplot(2,3,1)
                # plt.imshow(lmf_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('lmf_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(lmf_rmse, lmf_mae*100, lmf_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,3,2)
                # plt.imshow(argmax_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('argmax_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(argmax_rmse, argmax_mae*100, argmax_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,3,3)
                # plt.imshow(db3D_nl_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_nl_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_nl_rmse, db3D_nl_mae*100, db3D_nl_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,3,4)
                # plt.imshow(db3D_nl_d2d_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_nl_d2d_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_nl_d2d_rmse, db3D_nl_d2d_mae*100, db3D_nl_d2d_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,3,5)
                # plt.imshow(db3D_d2d_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_d2d_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_d2d_rmse, db3D_d2d_mae*100, db3D_d2d_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,3,6)
                # plt.imshow(db3D_d2d_tv0_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_d2d_tv0_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_d2d_tv0_rmse, db3D_d2d_tv0_mae*100, db3D_d2d_tv0_perc_errs),fontsize=14)
                # plt.colorbar()
                # out_fname = 'depths_' + scene_fname
                # plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname)

                # ######## Old 2x4 depths plots
                # plt.clf()
                # plt.suptitle("{} - SBR: {}, Signal: {} photons, Bkg: {} photons".format(scene_fname, SBR, mean_signal_photons, mean_background_photons), fontsize=20)
                # plt.subplot(2,4,1)
                # # plt.imshow(gt_depths, vmin=min_depth, vmax=max_depth); 
                # # plt.title('gt_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(gt_rmse, gt_mae*100, gt_perc_errs),fontsize=14)
                # plt.imshow(db3D_d2d_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_d2d_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_d2d_rmse, db3D_d2d_mae*100, db3D_d2d_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,2)
                # plt.imshow(lmf_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('lmf_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(lmf_rmse, lmf_mae*100, lmf_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,3)
                # plt.imshow(argmax_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('argmax_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(argmax_rmse, argmax_mae*100, argmax_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,4)
                # plt.imshow(db3D_nl_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_nl_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_nl_rmse, db3D_nl_mae*100, db3D_nl_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,5)
                # # plt.imshow(compressive_depths, vmin=min_depth, vmax=max_depth); 
                # # plt.title('compressive_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(compressive_rmse, compressive_mae*100, compressive_perc_errs),fontsize=14)
                # plt.imshow(db3D_nl_d2d_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db3D_nl_d2d_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_nl_d2d_rmse, db3D_nl_d2d_mae*100, db3D_nl_d2d_perc_errs),fontsize=14)
                # # plt.imshow(db2D_d2d2hist01Inputs_B12_depths, vmin=min_depth, vmax=max_depth); 
                # # plt.title('db2D_d2d2hist_B12_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d2hist01Inputs_B12_rmse, db2D_d2d2hist01Inputs_B12_mae*100, db2D_d2d2hist01Inputs_B12_perc_errs),fontsize=14)
                # plt.imshow(db2D_d2d01Inputs_B12_tv1m3_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db2D_B12_tv1m3_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B12_tv1m3_rmse, db2D_d2d01Inputs_B12_tv1m3_mae*100, db2D_d2d01Inputs_B12_tv1m3_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,6)
                # plt.imshow(db2D_d2d01Inputs_B12_tv3m5_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db2D_B12_tv3m5_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B12_tv3m5_rmse, db2D_d2d01Inputs_B12_tv3m5_mae*100, db2D_d2d01Inputs_B12_tv3m5_perc_errs),fontsize=14)
                # # plt.imshow(db2D_d2d01Inputs_B12_tv1m4_depths, vmin=min_depth, vmax=max_depth); 
                # # plt.title('db2D_B12_tv1m4_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B12_tv1m4_rmse, db2D_d2d01Inputs_B12_tv1m4_mae*100, db2D_d2d01Inputs_B12_tv1m4_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,7)
                # plt.imshow(db2D_d2d01Inputs_B12_tv1m5_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db2D_B12_tv1m5_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B12_tv1m5_rmse, db2D_d2d01Inputs_B12_tv1m5_mae*100, db2D_d2d01Inputs_B12_tv1m5_perc_errs),fontsize=14)
                # # plt.imshow(db2D_d2d01Inputs_B16_tv1m5_depths, vmin=min_depth, vmax=max_depth); 
                # # plt.title('db2D_B16_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B16_tv1m5_rmse, db2D_d2d01Inputs_B16_tv1m5_mae*100, db2D_d2d01Inputs_B16_tv1m5_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,8)
                # plt.imshow(db2D_d2d01Inputs_B12_tv1m10_depths, vmin=min_depth, vmax=max_depth); 
                # plt.title('db2D_B12_tv1m10_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B12_tv1m10_rmse, db2D_d2d01Inputs_B12_tv1m10_mae*100, db2D_d2d01Inputs_B12_tv1m10_perc_errs),fontsize=14)
                # # plt.imshow(db2D_d2d01Inputs_B24_tv1m5_depths, vmin=min_depth, vmax=max_depth); 
                # # plt.title('db2D_B24_model_depths \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B24_tv1m5_rmse, db2D_d2d01Inputs_B24_tv1m5_mae*100, db2D_d2d01Inputs_B24_tv1m5_perc_errs),fontsize=14)
                # plt.colorbar()
                # out_fname = 'depths_' + scene_fname
                # plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname)


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
                # # plt.imshow(compressive_abs_errs, vmin=min_err, vmax=max_err); 
                # # plt.title('compressive_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(compressive_rmse, compressive_mae*100, compressive_perc_errs),fontsize=14)
                # # plt.imshow(db3D_nl_d2d_abs_errs, vmin=min_err, vmax=max_err); 
                # # plt.title('db3D_nl_d2d_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db3D_nl_d2d_rmse, db3D_nl_d2d_mae*100, db3D_nl_d2d_perc_errs),fontsize=14)
                # plt.imshow(db2D_d2d2hist01Inputs_B12_abs_errs, vmin=min_err, vmax=max_err); 
                # plt.title('db2D_d2d2hist_B12_model_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d2hist01Inputs_B12_rmse, db2D_d2d2hist01Inputs_B12_mae*100, db2D_d2d2hist01Inputs_B12_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,6)
                # plt.imshow(db2D_d2d01Inputs_B12_tv1m5_abs_errs, vmin=min_err, vmax=max_err); 
                # plt.title('db2D_B12_model_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B12_tv1m5_rmse, db2D_d2d01Inputs_B12_tv1m5_mae*100, db2D_d2d01Inputs_B12_tv1m5_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,7)
                # plt.imshow(db2D_d2d01Inputs_B16_tv1m5_abs_errs, vmin=min_err, vmax=max_err); 
                # plt.title('db2D_B16_model_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B16_tv1m5_rmse, db2D_d2d01Inputs_B16_tv1m5_mae*100, db2D_d2d01Inputs_B16_tv1m5_perc_errs),fontsize=14)
                # plt.colorbar()
                # plt.subplot(2,4,8)
                # plt.imshow(db2D_d2d01Inputs_B24_tv1m5_abs_errs, vmin=min_err, vmax=max_err); 
                # plt.title('db2D_B24_model_abs_errs \n rmse: {:.3f}m | mae: {:.2f}cm \n perc. errs: {}'.format(db2D_d2d01Inputs_B24_tv1m5_rmse, db2D_d2d01Inputs_B24_tv1m5_mae*100, db2D_d2d01Inputs_B24_tv1m5_perc_errs),fontsize=14)
                # plt.colorbar()
                # out_fname = 'abs_errs_' + scene_fname
                # plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname)

    print("Test Set RMSE | MAE (cm) | MSE (cm):")
    print("    gt: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['gt']),calc_mean_mae(metrics_all['gt']),calc_mean_mse(metrics_all['gt'])))
    print("    lmf: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['lmf']),calc_mean_mae(metrics_all['lmf']),calc_mean_mse(metrics_all['lmf'])))
    print("    argmax: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['argmax']),calc_mean_mae(metrics_all['argmax']),calc_mean_mse(metrics_all['argmax'])))
    print("    compressive: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['compressive']),calc_mean_mae(metrics_all['compressive']),calc_mean_mse(metrics_all['compressive'])))
    print("    db3D: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D']),calc_mean_mae(metrics_all['db3D']),calc_mean_mse(metrics_all['db3D'])))
    print("    db3D_d2d: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_d2d']),calc_mean_mae(metrics_all['db3D_d2d']),calc_mean_mse(metrics_all['db3D_d2d'])))
    print("    db3D_d2d_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_d2d_tv0']),calc_mean_mae(metrics_all['db3D_d2d_tv0']),calc_mean_mse(metrics_all['db3D_d2d_tv0'])))
    print("    db3D_csphk64_separable4x4x1024_opt_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csphk64_separable4x4x1024_opt_grayfour_tv0'])))
    print("    db3D_csphk32_separable4x4x512_opt_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csphk32_separable4x4x512_opt_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csphk32_separable4x4x512_opt_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csphk32_separable4x4x512_opt_grayfour_tv0'])))
    print("    db3D_csphk16_separable4x4x256_opt_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csphk16_separable4x4x256_opt_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csphk16_separable4x4x256_opt_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csphk16_separable4x4x256_opt_grayfour_tv0'])))
    print("    db3D_csphk8_separable4x4x128_opt_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csphk8_separable4x4x128_opt_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csphk8_separable4x4x128_opt_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csphk8_separable4x4x128_opt_grayfour_tv0'])))
    print("    db3D_csphk128_separable4x4x1024_opt_truncfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0']),calc_mean_mae(metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0']),calc_mean_mse(metrics_all['db3D_csphk128_separable4x4x1024_opt_truncfour_tv0'])))
    print("    db3D_csphk64_separable4x4x512_opt_truncfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csphk64_separable4x4x512_opt_truncfour_tv0']),calc_mean_mae(metrics_all['db3D_csphk64_separable4x4x512_opt_truncfour_tv0']),calc_mean_mse(metrics_all['db3D_csphk64_separable4x4x512_opt_truncfour_tv0'])))
    print("    db3D_csphk32_separable2x2x1024_opt_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csphk32_separable2x2x1024_opt_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csphk32_separable2x2x1024_opt_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csphk32_separable2x2x1024_opt_grayfour_tv0'])))
    print("    db3D_csphk64_separable2x2x1024_opt_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csphk64_separable2x2x1024_opt_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csphk64_separable2x2x1024_opt_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csphk64_separable2x2x1024_opt_grayfour_tv0'])))
    print("    db3D_csphk256_separable4x4x1024_opt_truncfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0']),calc_mean_mae(metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0']),calc_mean_mse(metrics_all['db3D_csphk256_separable4x4x1024_opt_truncfour_tv0'])))
    print("    db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0']),calc_mean_mae(metrics_all['db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0']),calc_mean_mse(metrics_all['db3D_csph1DGlobal2DLocal4xDown_k128_truncfour_tv0'])))
    print("    db3D_csph1D2Dk32down2_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1D2Dk32down2_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csph1D2Dk32down2_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csph1D2Dk32down2_grayfour_tv0'])))
    print("    db3D_csph1D2Dk64down2_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1D2Dk64down2_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csph1D2Dk64down2_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csph1D2Dk64down2_grayfour_tv0'])))
    print("    db3D_csph1D2Dk128down2_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1D2Dk128down2_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csph1D2Dk128down2_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csph1D2Dk128down2_grayfour_tv0'])))
    print("    db3D_csph1D2Dk128down2upzncc_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1D2Dk128down2upzncc_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csph1D2Dk128down2upzncc_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csph1D2Dk128down2upzncc_grayfour_tv0'])))
    print("    db3D_csph1Dk8_gray_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1Dk8_gray_tv0']),calc_mean_mae(metrics_all['db3D_csph1Dk8_gray_tv0']),calc_mean_mse(metrics_all['db3D_csph1Dk8_gray_tv0'])))
    print("    db3D_csph1Dk8_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1Dk8_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csph1Dk8_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csph1Dk8_grayfour_tv0'])))
    print("    db3D_csph1Dk16_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1Dk16_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csph1Dk16_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csph1Dk16_grayfour_tv0'])))
    print("    db3D_csph1Dk16_truncfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1Dk16_truncfour_tv0']),calc_mean_mae(metrics_all['db3D_csph1Dk16_truncfour_tv0']),calc_mean_mse(metrics_all['db3D_csph1Dk16_truncfour_tv0'])))
    print("    db3D_csph1Dk16_coarsehist_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1Dk16_coarsehist_tv0']),calc_mean_mae(metrics_all['db3D_csph1Dk16_coarsehist_tv0']),calc_mean_mse(metrics_all['db3D_csph1Dk16_coarsehist_tv0'])))
    print("    db3D_csph1Dk32_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1Dk32_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csph1Dk32_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csph1Dk32_grayfour_tv0'])))
    print("    db3D_csph1Dk32_truncfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1Dk32_truncfour_tv0']),calc_mean_mae(metrics_all['db3D_csph1Dk32_truncfour_tv0']),calc_mean_mse(metrics_all['db3D_csph1Dk32_truncfour_tv0'])))
    print("    db3D_csph1Dk64_grayfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1Dk64_grayfour_tv0']),calc_mean_mae(metrics_all['db3D_csph1Dk64_grayfour_tv0']),calc_mean_mse(metrics_all['db3D_csph1Dk64_grayfour_tv0'])))
    print("    db3D_csph1Dk64_truncfour_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1Dk64_truncfour_tv0']),calc_mean_mae(metrics_all['db3D_csph1Dk64_truncfour_tv0']),calc_mean_mse(metrics_all['db3D_csph1Dk64_truncfour_tv0'])))
    print("    db3D_csph1Dk64_coarsehist_tv0: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db3D_csph1Dk64_coarsehist_tv0']),calc_mean_mae(metrics_all['db3D_csph1Dk64_coarsehist_tv0']),calc_mean_mse(metrics_all['db3D_csph1Dk64_coarsehist_tv0'])))
    print("    db2D_p2d_B16_7freqs_tv1m5: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db2D_p2d_B16_7freqs_tv1m5']),calc_mean_mae(metrics_all['db2D_p2d_B16_7freqs_tv1m5']),calc_mean_mse(metrics_all['db2D_p2d_B16_7freqs_tv1m5'])))
    print("    db2D_p2d_B16_allfreqs_tv1m5: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db2D_p2d_B16_allfreqs_tv1m5']),calc_mean_mae(metrics_all['db2D_p2d_B16_allfreqs_tv1m5']),calc_mean_mse(metrics_all['db2D_p2d_B16_allfreqs_tv1m5'])))
    print("    db2D_d2d2hist01Inputs_B12: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db2D_d2d2hist01Inputs_B12']),calc_mean_mae(metrics_all['db2D_d2d2hist01Inputs_B12']),calc_mean_mse(metrics_all['db2D_d2d2hist01Inputs_B12'])))
    print("    db2D_d2d01Inputs_B12_tv1m3: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db2D_d2d01Inputs_B12_tv1m3']),calc_mean_mae(metrics_all['db2D_d2d01Inputs_B12_tv1m3']),calc_mean_mse(metrics_all['db2D_d2d01Inputs_B12_tv1m3'])))
    print("    db2D_d2d01Inputs_B12_tv1m4: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db2D_d2d01Inputs_B12_tv1m4']),calc_mean_mae(metrics_all['db2D_d2d01Inputs_B12_tv1m4']),calc_mean_mse(metrics_all['db2D_d2d01Inputs_B12_tv1m4'])))
    print("    db2D_d2d01Inputs_B12_tv3m5: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db2D_d2d01Inputs_B12_tv3m5']),calc_mean_mae(metrics_all['db2D_d2d01Inputs_B12_tv3m5']),calc_mean_mse(metrics_all['db2D_d2d01Inputs_B12_tv3m5'])))
    print("    db2D_d2d01Inputs_B12_tv1m5: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db2D_d2d01Inputs_B12_tv1m5']),calc_mean_mae(metrics_all['db2D_d2d01Inputs_B12_tv1m5']),calc_mean_mse(metrics_all['db2D_d2d01Inputs_B12_tv1m5'])))
    print("    db2D_d2d01Inputs_B12_tv1m10: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db2D_d2d01Inputs_B12_tv1m10']),calc_mean_mae(metrics_all['db2D_d2d01Inputs_B12_tv1m10']),calc_mean_mse(metrics_all['db2D_d2d01Inputs_B12_tv1m10'])))
    print("    db2D_d2d01Inputs_B16_tv1m5: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db2D_d2d01Inputs_B16_tv1m5']),calc_mean_mae(metrics_all['db2D_d2d01Inputs_B16_tv1m5']),calc_mean_mse(metrics_all['db2D_d2d01Inputs_B16_tv1m5'])))
    print("    db2D_d2d01Inputs_B22_tv1m5: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db2D_d2d01Inputs_B22_tv1m5']),calc_mean_mae(metrics_all['db2D_d2d01Inputs_B22_tv1m5']),calc_mean_mse(metrics_all['db2D_d2d01Inputs_B22_tv1m5'])))
    print("    db2D_d2d01Inputs_B24_tv1m5: {:.3f} | {:.2f}cm | {:.2f} ".format(calc_mean_rmse(metrics_all['db2D_d2d01Inputs_B24_tv1m5']),calc_mean_mae(metrics_all['db2D_d2d01Inputs_B24_tv1m5']),calc_mean_mse(metrics_all['db2D_d2d01Inputs_B24_tv1m5'])))

