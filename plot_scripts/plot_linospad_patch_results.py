'''
    This script loads the estimated patch depths for a model and stitches them into the final image
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
from research_utils import io_ops

def reshape_mat_sparse_spad_data(spad_sparse_data, nr, nc, frame_num=0):
    spad_data = np.asarray(scipy.sparse.csc_matrix.todense(spad_sparse_data[frame_num]))
    num_bins = spad_data.shape[0]
    spad_data = spad_data.reshape((num_bins, nr, nc)).swapaxes(-1,-2)
    return spad_data


if __name__=='__main__':

    ## Set patch dims we want to divide each image into
    (patch_nr, patch_nc) = (64, 64)
    (patch_nr, patch_nc) = (128, 128)
    num_patch_pixels = patch_nr*patch_nc

    ## Set dirpaths
    model_dirpath = 'outputs/nyuv2_64x64x1024_80ps/test_csph3d/DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0/2022-08-10_162141'
    dataset_name = 'test_lindell2018_linospad_captured_patches-{}-{}'.format(patch_nr, patch_nc)
    dataset_dirpath = os.path.join(model_dirpath, dataset_name)

    ## load scene
    scene_id = 'elephant'
    frame_num = 0

    ## raw data paths
    (n_tbins, nr, nc) = (1536, 256, 256) # original data dimensions
    assert((nr % patch_nr) == 0), "num rows should be divisible by patch num rows"
    assert((nc % patch_nc) == 0), "num rows should be divisible by patch num rows"
    raw_dataset_id = 'captured'
    raw_data_base_dirpath = './2018SIGGRAPH_lindell_test_data' 
    raw_data_dirpath = os.path.join(raw_data_base_dirpath, raw_dataset_id)

    ## load each patch and stitch it
    num_row_patches = int(nr / patch_nr)
    num_col_patches = int(nc / patch_nc)
    rec_norm_bin = np.zeros((nr,nc))
    for r in range(num_row_patches):
        start_row = r*patch_nr
        end_row = start_row + patch_nr
        for c in range(num_col_patches):
            start_col = c*patch_nc
            end_col = start_col + patch_nc
            patch_fname =  scene_id + '_frame-{}_patch-{}-{}'.format(frame_num, r, c) + '.npz'
            patch_fpath = os.path.join(dataset_dirpath, patch_fname)
            patch_rec_norm_bin = np.load(patch_fpath)['dep_re']
            rec_norm_bin[start_row:end_row, start_col:end_col] = patch_rec_norm_bin.squeeze() 


    plt.clf()
    plt.imshow(rec_norm_bin)
