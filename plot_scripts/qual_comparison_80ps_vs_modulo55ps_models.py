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
from research_utils import plot_utils
import tof_utils


if __name__=='__main__':

    n_tbins = 1024
    scene_id = 'spad_Art'
    noise_id = '2_10'
    scene_fname = '{}_{}.npz'.format(scene_id, noise_id)
    out_dirpath = 'results/week_2023-02-13/qual_comparison_80ps_vs_modulo55ps'
    out_dirpath = os.path.join(out_dirpath, '{}_{}'.format(scene_id, noise_id))
    os.makedirs(out_dirpath, exist_ok=True)


    test_dataset = 'test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
    tau = n_tbins*98e-12

    if(scene_id == 'spad_Art'):
        (min_depth, max_depth) = (1.3, 2.3)
    else:
        (min_depth, max_depth) = (1., 3.0)


    # ############ Learned Separable 1024x4x4 ################ 
    # ## 80ps dataset learned separable 1024x4x4
    # # append 128x, 64x and 32x compression
    # model_80ps_dirpaths = {}
    # # model_80ps_dirpaths['separable_1024x4x4_k128'] = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models/DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-09-24_044702'
    # # model_80ps_dirpaths['separable_1024x4x4_k256'] = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models/DDFN_C64B10_CSPH3D/k256_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-09-25_174558'
    # # model_80ps_dirpaths['separable_1024x4x4_k512'] = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models/DDFN_C64B10_CSPH3D/k512_down8_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-10-04_183828'
    # model_80ps_dirpaths['separable_1024x4x4_k128'] = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models_rerun/DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-10-07_193425'
    # model_80ps_dirpaths['separable_1024x4x4_k256'] = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models_rerun/DDFN_C64B10_CSPH3D/k256_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-10-06_141241'
    # model_80ps_dirpaths['separable_1024x4x4_k512'] = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models_rerun/DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-10-06_141241'
    # ## modulo-55ps dataset learned separable 1024x4x4
    # # append 128x, 64x and 32x compression
    # model_55ps_dirpaths = {}
    # model_55ps_dirpaths['separable_1024x4x4_k128'] = 'outputs/modulo_nyuv2_64x64x1024_55ps/validate_new_modulo_dataset_20230213/DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-14_065901'
    # model_55ps_dirpaths['separable_1024x4x4_k256'] = 'outputs/modulo_nyuv2_64x64x1024_55ps/validate_new_modulo_dataset_20230213/DDFN_C64B10_CSPH3D/k256_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-14_065901'
    # model_55ps_dirpaths['separable_1024x4x4_k512'] = 'outputs/modulo_nyuv2_64x64x1024_55ps/validate_new_modulo_dataset_20230213/DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-14_065901'

    # ############ Learned Separable 256x4x4 ################ 
    # ## 80ps dataset learned separable 256x4x4
    # # append 128x, 64x and 32x compression
    # model_80ps_dirpaths = {}
    # model_80ps_dirpaths['separable_256x4x4_k32'] = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models/DDFN_C64B10_CSPH3D/k32_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-09-24_175849'
    # model_80ps_dirpaths['separable_256x4x4_k64'] = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models/DDFN_C64B10_CSPH3D/k64_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-09-24_175338'
    # model_80ps_dirpaths['separable_256x4x4_k128'] = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models/DDFN_C64B10_CSPH3D/k128_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-09-25_174558'
    # ## modulo-55ps dataset learned separable 256x4x4
    # # append 128x, 64x and 32x compression
    # model_55ps_dirpaths = {}
    # model_55ps_dirpaths['separable_256x4x4_k32'] = 'outputs/modulo_nyuv2_64x64x1024_55ps/validate_new_modulo_dataset_20230213/DDFN_C64B10_CSPH3D/k32_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-14_065801'
    # model_55ps_dirpaths['separable_256x4x4_k64'] = 'outputs/modulo_nyuv2_64x64x1024_55ps/validate_new_modulo_dataset_20230213/DDFN_C64B10_CSPH3D/k64_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-14_065801'
    # model_55ps_dirpaths['separable_256x4x4_k128'] = 'outputs/modulo_nyuv2_64x64x1024_55ps/validate_new_modulo_dataset_20230213/DDFN_C64B10_CSPH3D/k128_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-latest_2023-02-17_024851'

    ############ Learned Separable 1024x1x1 ################ 
    ## 80ps dataset learned separable 1024x1x1
    # append 128x, 64x and 32x compression
    model_80ps_dirpaths = {}
    model_80ps_dirpaths['separable_1024x1x1_k8'] = 'outputs/nyuv2_64x64x1024_80ps/csph3D_tdim_baselines/DDFN_C64B10_CSPH3D/k8_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-10-05_115838'
    model_80ps_dirpaths['separable_1024x1x1_k16'] = 'outputs/nyuv2_64x64x1024_80ps/csph3D_tdim_baselines/DDFN_C64B10_CSPH3D/k16_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-10-04_002644'
    model_80ps_dirpaths['separable_1024x1x1_k32'] = 'outputs/nyuv2_64x64x1024_80ps/csph3D_tdim_baselines/DDFN_C64B10_CSPH3D/k32_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-10-04_002551'
    ## modulo-55ps dataset learned separable 1024x1x1
    # append 128x, 64x and 32x compression
    model_55ps_dirpaths = {}
    model_55ps_dirpaths['separable_1024x1x1_k8'] = 'outputs/modulo_nyuv2_64x64x1024_55ps/validate_new_modulo_dataset_20230213/DDFN_C64B10_CSPH3D/k8_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-14_065601'
    model_55ps_dirpaths['separable_1024x1x1_k16'] = 'outputs/modulo_nyuv2_64x64x1024_55ps/validate_new_modulo_dataset_20230213/DDFN_C64B10_CSPH3D/k16_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-14_065716'
    model_55ps_dirpaths['separable_1024x1x1_k32'] = 'outputs/modulo_nyuv2_64x64x1024_55ps/validate_new_modulo_dataset_20230213/DDFN_C64B10_CSPH3D/k32_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-14_092750'



    for model_id in model_80ps_dirpaths.keys():
        model_id_80ps = model_id + '_80ps'
        fpath = os.path.join(model_80ps_dirpaths[model_id], test_dataset, scene_fname)
        norm_bins_img_80ps = np.load(fpath)['dep_re'].squeeze()
        depth_img_80ps = tof_utils.bin2depth(norm_bins_img_80ps*n_tbins, n_tbins, tau)
        plt.figure()
        plt.imshow(depth_img_80ps, vmin=min_depth, vmax=max_depth)
        plt.title(model_id_80ps, fontsize=12)
        plot_utils.remove_ticks()
        plot_utils.save_currfig_png(out_dirpath, model_id_80ps)

    for model_id in model_55ps_dirpaths.keys():
        model_id_55ps = model_id + '_55ps'
        fpath = os.path.join(model_55ps_dirpaths[model_id], test_dataset, scene_fname)
        norm_bins_img_55ps = np.load(fpath)['dep_re'].squeeze()
        depth_img_55ps = tof_utils.bin2depth(norm_bins_img_55ps*n_tbins, n_tbins, tau)
        plt.figure()
        plt.imshow(depth_img_55ps, vmin=min_depth, vmax=max_depth)
        plt.title(model_id_55ps, fontsize=12)
        plot_utils.remove_ticks()
        plot_utils.save_currfig_png(out_dirpath, model_id_55ps)





