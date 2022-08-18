#### Standard Library Imports
import sys
import os
sys.path.append('../')
sys.path.append('./')

#### Library imports
import numpy as np
import scipy
import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
import tof_utils

def print_arr_stats(arr, arr_name='arr'):
    print("{} stats:".format(arr_name))
    print("    - shape: {}".format(arr.shape))
    print("    - max: {:.3f}".format(arr.flatten().max()))
    print("    - min: {:.3f}".format(arr.flatten().min()))
    print("    - mean: {:.3f}".format(arr.flatten().mean()))
    print("    - std: {:.3f}".format(arr.flatten().std()))
    print("    - median: {:.3f}".format(np.median(arr.flatten())))

if __name__=='__main__':

    ## real data data paths
    data_dirpath = './2018SIGGRAPH_lindell_test_data/captured'
    ids = ['elephant','checkerboard','hallway','kitchen','lamp','roll','stairs_ball','stairs_walking','stuff'] 
    (nr, nc) = (256,256) # number of rows and cols
    tres = 26e-12 #secs

    # ## synthetic data dirpaths
    # data_dirpath = './2018SIGGRAPH_lindell_test_data/simulated/LR'
    # ids = ['LR_Art_10_10','LR_Books_10_10','LR_Bowling1_10_10','LR_Dolls_10_10','LR_Laundry_10_10'] 
    # (nr, nc) = (72, 88) # number of rows and cols
    # tres = 98e-12 #secs


    ## Load scene
    id = ids[3]
    data_fpath = os.path.join(data_dirpath, id+'.mat')
    data = scipy.io.loadmat(data_fpath)

    ## Get data
    if('simulated' in data_dirpath):
        spad_sparse_data = data['spad']
        intensity_imgs = data['intensity']
        ## Pre-process data
        intensity_img = intensity_imgs.astype(np.float32)
        spad_data = np.asarray(scipy.sparse.csc_matrix.todense(spad_sparse_data)).astype(np.float32)
        num_bins = spad_data.shape[-1]
        spad_data = spad_data.reshape((nc, nr, num_bins)).swapaxes(0,-1)
    else:
        spad_sparse_data = np.asarray(data['spad_processed_data'])[0]
        intensity_imgs = np.asarray(data['cam_processed_data'])[0]
        n_frames = len(spad_sparse_data)
        ## Pre-process data.
        frame_num = 0
        intensity_img = np.asarray(intensity_imgs[frame_num]).astype(np.float32)
        spad_data = np.asarray(scipy.sparse.csc_matrix.todense(spad_sparse_data[frame_num])).astype(np.float32)
        num_bins = spad_data.shape[0]
        spad_data = spad_data.reshape((num_bins, nr, nc)).swapaxes(-1,-2)

    (nt, nr, nc) = spad_data.shape
    # denoise data a little bit
    spad_data_dn = scipy.ndimage.gaussian_filter(spad_data, sigma=2, mode='wrap', truncate=10.0)

    # compute depths
    tau = nt*tres
    argmax_bin = np.argmax(spad_data_dn, axis=0)
    argmax_depths = tof_utils.bin2depth(argmax_bin, num_bins=nt, tau=tau)

    print_arr_stats(intensity_img, 'intensity_img')
    print_arr_stats(np.argmax(spad_data_dn, axis=0), 'spad argmax')
    print_arr_stats(spad_data, 'raw spad data')

    ## Visualize
    plt.clf()
    plt.subplot(3,2,1)
    plt.imshow(intensity_img)
    plt.title("Intensity Image")
    plt.subplot(3,2,2)
    plt.imshow(spad_data_dn.sum(axis=0))
    plt.title("Gauss Denoised Photon Counts")
    plt.subplot(3,3,4)
    plt.imshow(spad_data_dn.max(axis=0))
    plt.title("Gauss denoised Maximum")
    plt.subplot(3,3,5)
    plt.imshow(argmax_bin)
    plt.colorbar()
    plt.title("Gauss denoised Argmax")
    plt.subplot(3,3,6)
    plt.imshow(argmax_depths)
    plt.colorbar()
    plt.title("Gauss denoised Argmax Depths (m)")

    plt.subplot(3,1,3)
    plt.plot(spad_data[:,nr//2,nc//2], '-*', linewidth=2, label='raw hist ({},{})'.format(nr//2,nc//2))
    plt.plot(3*spad_data_dn[:,nr//2,nc//2], linewidth=2, label='denoised hist ({},{})'.format(nr//2,nc//2))
    plt.plot(spad_data[:,nr//4,nc//4], '-*', linewidth=2, label='raw hist ({},{})'.format(nr//4,nc//4))
    plt.plot(3*spad_data_dn[:,nr//4,nc//4], linewidth=2, label='denoised hist ({},{})'.format(nr//4,nc//4))
    plt.title("Example Histograms")
    plt.legend()
