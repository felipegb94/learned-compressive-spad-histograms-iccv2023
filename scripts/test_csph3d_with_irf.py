'''
@author: Felipe Gutierrez-Barragan
'''
import sys
import os
sys.path.append('../')
sys.path.append('./')

## Library Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import torch
from IPython.core import debugger
breakpoint = debugger.set_trace

## Local Imports
from csph_layers import CSPH3DLayer
from layers_parametric1D import IRF1DLayer
from research_utils import plot_utils


if __name__=='__main__':
    ## Fix seed to make sure we get the same random weights in the csph3d layer
    torch.manual_seed(0)

    ## Load IRF function
    irf_fpath = 'data_gener/TrainData/SimSPADDataset_nr-64_nc-64_nt-1024_tres-80ps_dark-1_psf-1/PSF_used_for_simulation_nr-64_nc-64.mat'
    irf = scipy.io.loadmat(irf_fpath)['psf'].mean(axis=-1)

    ## Create a CSPH3D object
    k = 512
    nt = 1024
    spatial_down_factor=4
    nt_blocks=1
    csph_out_norm='LinfGlobal'
    tblock_init = 'HybridGrayFourier'
    tblock_init = 'HybridGrayFourier'
    # tblock_init = 'Rand'
    csph3d_layer = CSPH3DLayer(k=k,num_bins=nt, h_irf=irf, nt_blocks=nt_blocks, spatial_down_factor=spatial_down_factor, encoding_type='full', csph_out_norm=csph_out_norm, tblock_init=tblock_init)
    # csph3d_layer = CSPH3DLayer(k=k,num_bins=nt, h_irf=None, nt_blocks=nt_blocks, spatial_down_factor=spatial_down_factor, encoding_type='full', csph_out_norm=csph_out_norm, tblock_init=tblock_init, account_irf=True)
    W3d = csph3d_layer.get_unfilt_backproj_W3d_full()
    W3d_decoding = csph3d_layer.get_unfilt_backproj_W3d_with_irf_full()

    csph3d_layer = CSPH3DLayer(k=k,num_bins=nt, h_irf=irf, nt_blocks=nt_blocks, spatial_down_factor=spatial_down_factor, encoding_type='separable', csph_out_norm=csph_out_norm, tblock_init=tblock_init)
    W3d = csph3d_layer.get_unfilt_backproj_W3d_separable()
    W3d_decoding = csph3d_layer.get_unfilt_backproj_W3d_with_irf_separable()

    ## plot temporal coding functions
    plt.clf()
    plt.subplot(3,1,1)
    plt.plot(csph3d_layer.irf_layer.irf_weights.squeeze())
    plt.title("Temporal Impulse Response Function")
    plt.subplot(3,1,2)
    plt.plot(W3d[50,0,:,0,0].detach().cpu().numpy().squeeze(), label='W3d[50,0,:,0,0]')
    plt.plot(W3d_decoding[50,0,:,0,0].detach().cpu().numpy().squeeze(), '--', label='W3d_decoding[50,0,:,0,0]')
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(W3d[200,0,:,3,3].detach().cpu().numpy().squeeze(), label='W3d[200,0,:,3,3]')
    plt.plot(W3d_decoding[200,0,:,3,3].detach().cpu().numpy().squeeze(), '--', label='W3d_decoding[200,0,:,3,3]')
    plt.legend()

