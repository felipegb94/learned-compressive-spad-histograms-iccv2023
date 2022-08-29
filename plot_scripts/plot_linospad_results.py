'''
    Plots the recovered depth images from the linospad dataset without having to crop them
'''

#### Standard Library Imports
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


if __name__=='__main__':

    ## dirpuath where to store the images
    experiment_name = 'csph3d_good_norm'
    out_dirpath = os.path.join('./results/week_2022-08-22/linospad_results', experiment_name)
    os.makedirs(out_dirpath, exist_ok=True)

    ## Parameters
    nt = 1536
    time_res = 26e-12
    tau = time_res*nt
    frame_num = 0
    linospad_dataset_id = 'test_lindell2018_linospad_captured_min'

    ## scene we want to plot
    # scene_id = 'elephant'
    scene_id = 'kitchen'
    # scene_id = 'lamp'
    
    ## add all model dirpaths
    model_dirpaths = []
    
    ## Add Baseline Models
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10/2022-04-25_192521')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_Depth2Depth/loss-kldiv_tv-0.0/2022-05-07_171658')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/debug/DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-5/2022-05-02_085659')
    
    ## Add CSPH3D Models (LinfGlobal, TV=[ 1e-5, 3e-6, 1e-6]
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/csph3d_good_norm/DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-1e-05/run-complete_2022-08-23_231951')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/csph3d_good_norm/DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-1e-06/run-latest_2022-08-24_145125')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/csph3d_good_norm/DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-3e-06/run-complete_2022-08-26_040500')

    ## Add previous CSPH3Dv1 Separable Models (No Norm, TV=0.0]
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/test_csph3d/DDFN_C64B10_CSPH3Dv1/k512_down4_Mt1_HybridGrayFourier-optCt=True-optC=True_separable/loss-kldiv_tv-0.0/2022-08-10_013956')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/test_csph3d/DDFN_C64B10_CSPH3Dv1/k128_down4_Mt1_HybridGrayFourier-optCt=True-optC=True_separable/loss-kldiv_tv-0.0/2022-08-10_014115')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/test_csph3d/DDFN_C64B10_CSPH3Dv1/k128_down4_Mt1_Rand-optCt=True-optC=True_separable/loss-kldiv_tv-0.0/2022-08-11_145813')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/test_csph3d/DDFN_C64B10_CSPH3Dv1/k512_down4_Mt1_Rand-optCt=True-optC=True_separable/loss-kldiv_tv-0.0/2022-08-11_145814')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/csph3d_good_norm/DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0/run-complete_2022-08-10_162141')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/test_csph3d/DDFN_C64B10_CSPH3Dv1/k128_down4_Mt1_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0/2022-07-31_145411')

    ## Add CSPH3D Models (Different Normalization, TV=0.0)
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/csph3d_good_norm/DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0/run-complete_2022-08-10_162141')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/csph3d_good_norm/DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-0.0/run-latest_2022-08-24_144556')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/csph3d_good_norm/DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0/run-complete_2022-08-10_162141')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/csph3d_good_norm/DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-Linf/loss-kldiv_tv-0.0/run-latest_2022-08-18_011732')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/csph3d_good_norm/DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-L2/loss-kldiv_tv-0.0/run-latest_2022-08-18_013924')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/test_csph3d/DDFN_C64B10_CSPH3Dv1/k128_down4_Mt1_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0/2022-07-31_145411')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/csph3d_good_norm/DDFN_C64B10_CSPH3Dv2/k128_down4_Mt1_Rand-optCt=True-optC=True_full_norm-Linf/loss-kldiv_tv-0.0/run-complete_2022-08-18_011947')
    model_dirpaths.append('outputs/nyuv2_64x64x1024_80ps/csph3d_good_norm/DDFN_C64B10_CSPH3Dv2/k128_down4_Mt1_Rand-optCt=True-optC=True_full_norm-L2/loss-kldiv_tv-0.0/run-latest_2022-08-18_013924')
    # Generate model ids from dirpath
    model_ids = []
    for model_dir in model_dirpaths:
        model_id = '_'.join(model_dir.split('/')[3:-1])
        model_id = model_id.replace('DDFN_C64B10', 'db3d')
        model_id = model_id.replace('down4_Mt1', '1024x4x4')
        model_id = model_id.replace('_loss-kldiv', '')
        model_id = model_id.replace('optCt=', '\noptCt=')
        model_ids.append(model_id)
        print(model_id)

    (vmin, vmax) = (0.5, 4)
    if(scene_id == 'elephant'): (vmin,vmax)=(0.3, 2.5)
    if(scene_id == 'lamp'): (vmin,vmax)=(0.3, 2.5)
    if(scene_id == 'kitchen'): (vmin,vmax)=(1.3, 4.8)

    ## load the depths (in units of bins)
    rec_depths_all = []
    for i in range(len(model_dirpaths)):
        model_dir = model_dirpaths[i]
        fpath = os.path.join(model_dir, 'test_lindell2018_linospad_captured_min/{}.npz'.format(scene_id))
        data = np.load(fpath)
        rec_bins = data['dep_re'].squeeze()*nt
        rec_depths = bin2depth(rec_bins, num_bins=nt, tau=tau)
        rec_depths_all.append(rec_depths)

        plt.clf()
        plt.imshow(rec_depths, vmin=vmin, vmax=vmax)
        plt.title(model_ids[i])
        plot_utils.save_currfig_png(dirpath=os.path.join(out_dirpath, scene_id), filename=model_ids[i])
        plt.pause(0.5)





