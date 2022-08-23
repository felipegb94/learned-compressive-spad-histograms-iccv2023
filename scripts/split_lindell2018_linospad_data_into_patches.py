'''
    This script goes through the Lindell et al., 2018 linospad dataset and splits each image into patches of dimensison 1536x64x64 and saves each patch separately. It only grabs the first frame if there are multiple frames

    To run this script make sure to set the `data_base_dirpath` variable appropriately

'''

#### Standard Library Imports
import sys
import os
sys.path.append('../')
sys.path.append('./')

#### Library imports
import numpy as np
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
    (patch_n_rows, patch_n_cols) = (128, 128)
    num_patch_pixels = patch_n_rows*patch_n_cols
    frame_num = 0

    ## real data data paths
    (n_tbins, n_rows, n_cols) = (1536, 256, 256) # original data dimensions
    assert((n_rows % patch_n_rows) == 0), "num rows should be divisible by patch num rows"
    assert((n_cols % patch_n_cols) == 0), "num rows should be divisible by patch num rows"
    data_id = 'captured'
    data_base_dirpath = './2018SIGGRAPH_lindell_test_data' 
    data_dirpath = os.path.join(data_base_dirpath, data_id)


    ## create output dirpath
    out_data_dirpath = os.path.join(data_base_dirpath, data_id + '_patches-{}-{}'.format(patch_n_rows, patch_n_cols))
    os.makedirs(out_data_dirpath, exist_ok=True)

    ## Get all filenames in directory
    data_fnames = io_ops.get_filepaths_in_dir(data_dirpath, match_str_pattern='.mat', only_filenames=True, keep_ext=True)


    for i in range(len(data_fnames)):
        ## Load data
        data_fpath = os.path.join(data_dirpath, data_fnames[i])
        data = scipy.io.loadmat(data_fpath)
        spad_sparse_data = np.asarray(data['spad_processed_data'])[0]
        intensity_imgs = np.asarray(data['cam_processed_data'])[0]
        n_frames = len(spad_sparse_data)
        ## Reshape data appropriately.
        intensity_img = np.asarray(intensity_imgs[frame_num])
        spad_data = reshape_mat_sparse_spad_data(spad_sparse_data, nr=n_rows, nc=n_cols, frame_num=frame_num)

        ## Save each patch individually
        num_row_patches = int(n_rows / patch_n_rows)
        num_col_patches = int(n_cols / patch_n_cols)
        
        print("splitting {}".format(data_fnames[i]))

        for r in range(num_row_patches):
            start_row = r*patch_n_rows
            end_row = start_row + patch_n_rows
            for c in range(num_col_patches):
                start_col = c*patch_n_cols
                end_col = start_col + patch_n_cols
                spad_data_patch = spad_data[:, start_row:end_row, start_col:end_col]
                ## reshape the patch in the same way it was stored
                spad_data_patch = spad_data_patch.swapaxes(-1,-2).reshape((n_tbins, num_patch_pixels))
                spad_sparse_data_patch = scipy.sparse.csc_matrix(spad_data_patch)

                ## Save the sparse matrix
                out_fname = data_fnames[i].split('.mat')[0] + '_frame-{}_patch-{}-{}'.format(frame_num, r, c) + '.mat'
                out_fpath = os.path.join(out_data_dirpath, out_fname)
                print('    ' + out_fname)
                mdict = {'spad_processed_data': [spad_sparse_data_patch]} # do not include cam_processed_data to reduce replication of data
                scipy.io.savemat(out_fpath, mdict, format='5')

        ## Validate that the data that was saved matches the data we got it from (this is useful to make sure that we used the correct scipy.io.savemat settings)
        saved_spad_data = np.zeros_like(spad_data)
        for r in range(num_row_patches):
            start_row = r*patch_n_rows
            end_row = start_row + patch_n_rows
            for c in range(num_col_patches):
                start_col = c*patch_n_cols
                end_col = start_col + patch_n_cols
                patch_fname = data_fnames[i].split('.mat')[0] + '_frame-{}_patch-{}-{}'.format(frame_num, r, c) + '.mat'
                patch_fpath = os.path.join(out_data_dirpath, patch_fname)
                patch_data = scipy.io.loadmat(patch_fpath)
                spad_sparse_data_patch = np.asarray(patch_data['spad_processed_data'])[0] 
                spad_data_patch = reshape_mat_sparse_spad_data(spad_sparse_data_patch, nr=patch_n_rows, nc=patch_n_cols)
                saved_spad_data[:, start_row:end_row, start_col:end_col] = spad_data_patch

        abs_diff = np.abs(spad_data - saved_spad_data).sum()
        print("    abs_diff = {}".format(abs_diff))
        assert(abs_diff < 1e-5), "loaded data differs from the saved data"