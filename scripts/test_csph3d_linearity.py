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
from research_utils import plot_utils


if __name__=='__main__':

	# Select device
	use_gpu = True
	if(torch.cuda.is_available() and use_gpu): device = torch.device("cuda:0")
	else: device = torch.device("cpu")
	
	# Set things to be reproducible
	deterministic = False
	torch.manual_seed(0)
	torch.use_deterministic_algorithms(deterministic)
	torch.backends.cudnn.deterministic = deterministic

	# dirpath to save results
	out_dirpath = 'results/week_2022-08-15/test_csph3d_linearity'

	# Set CSPH3D layer parameters input
	k=32
	tblock_init = 'TruncFourier'
	optimize_tdim_codes = True
	optimize_codes = True
	batch_size = 4
	nt_blocks = 1
	spatial_down_factor = 4	
	(nr, nc, nt) = (32, 32, 128)
	encoding_type = 'separable'
	inputs_type = 'randint'
	
	## Generate different inputs combinations
	if(inputs_type == 'randint'):
		inputs1 = torch.randint(low=0, high=5, size=(batch_size, 1, nt, nr, nc), dtype=torch.float32)
		inputs2 = torch.randint(low=0, high=10, size=(batch_size, 1, nt, nr, nc), dtype=torch.float32)
		inputs2[inputs2 < 7] = 0 
		inputs3 = torch.randint(low=0, high=22, size=(batch_size, 1, nt, nr, nc), dtype=torch.float32)
	elif(inputs_type == 'random'):
		inputs1 = 5*torch.rand(size=(batch_size, 1, nt, nr, nc), dtype=torch.float32)
		inputs2 = 10*torch.rand(size=(batch_size, 1, nt, nr, nc), dtype=torch.float32)
		inputs2[inputs2 < 7] = 0 
		inputs3 = 22*torch.rand(size=(batch_size, 1, nt, nr, nc), dtype=torch.float32)
	else:
		middlebury_data_dirpath = './data_gener/TestData/middlebury/processed/SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
		inputs1_fnames = ['spad_Art_10_10','spad_Books_10_10','spad_Dolls_10_10','spad_Laundry_10_10']
		inputs2_fnames = ['spad_Art_2_50','spad_Books_2_50','spad_Dolls_2_50','spad_Laundry_2_50']
		inputs3_fnames = ['spad_Art_10_50','spad_Books_10_50','spad_Dolls_10_50','spad_Laundry_10_50']
		inputs1 = torch.zeros((batch_size, 1, nt, nr, nc), dtype=torch.float32)
		inputs2 = torch.zeros((batch_size, 1, nt, nr, nc), dtype=torch.float32)
		inputs3 = torch.zeros((batch_size, 1, nt, nr, nc), dtype=torch.float32)
		for i in range(len(inputs1_fnames)):
			inputs1[i,0,:] = torch.from_numpy(np.asarray(scipy.sparse.csc_matrix.todense(scipy.io.loadmat(os.path.join(middlebury_data_dirpath, inputs1_fnames[i]))['spad'])).astype(np.float32).reshape(72,88,1024)).permute((2, 0, 1))[:, 0:nr, 0:nc]
			inputs2[i,0,:] = torch.from_numpy(np.asarray(scipy.sparse.csc_matrix.todense(scipy.io.loadmat(os.path.join(middlebury_data_dirpath, inputs2_fnames[i]))['spad'])).astype(np.float32).reshape(72,88,1024)).permute((2, 0, 1))[:, 0:nr, 0:nc]
			inputs3[i,0,:] = torch.from_numpy(np.asarray(scipy.sparse.csc_matrix.todense(scipy.io.loadmat(os.path.join(middlebury_data_dirpath, inputs3_fnames[i]))['spad'])).astype(np.float32).reshape(72,88,1024)).permute((2, 0, 1))[:, 0:nr, 0:nc]
	inputs1.to(device)
	inputs2.to(device)
	inputs3.to(device)
			

	## Create CSPH3D layer
	csph_layer = CSPH3DLayer(k=k, num_bins=nt, tblock_init=tblock_init, optimize_codes=True, nt_blocks=nt_blocks, spatial_down_factor=spatial_down_factor, encoding_type=encoding_type)
	csph_layer.to(device)

	## Test CSPH layer
	if(encoding_type == 'full'):
		out_csph1 = csph_layer.csph_layer_full(inputs1)
		out_csph2 = csph_layer.csph_layer_full(inputs2)
		out_csph3 = csph_layer.csph_layer_full(inputs3)
		out_csph1plus2 = csph_layer.csph_layer_full(inputs1 + inputs2)
		out_csph1plus3 = csph_layer.csph_layer_full(inputs1 + inputs3)
		out_csph2plus3 = csph_layer.csph_layer_full(inputs2 + inputs3)
		out_csph1plus2plus3 = csph_layer.csph_layer_full(inputs1 + inputs2) + out_csph3 
	elif(encoding_type=='separable'):
		out_csph1 = csph_layer.csph_layer_separable(inputs1)
		out_csph2 = csph_layer.csph_layer_separable(inputs2)
		out_csph3 = csph_layer.csph_layer_separable(inputs3)
		out_csph1plus2 = csph_layer.csph_layer_separable(inputs1 + inputs2)
		out_csph1plus3 = csph_layer.csph_layer_separable(inputs1 + inputs3)
		out_csph2plus3 = csph_layer.csph_layer_separable(inputs2 + inputs3)
		out_csph1plus2plus3 = csph_layer.csph_layer_separable(inputs1 + inputs2) + out_csph3 
	else:
		assert(False), "Invalid encoding_type"

	## Get the absolute differences between CSPH(x1+x2) - (CSPH(x1) + CSPH(x2))  
	csph12_minus_csph1_2 = (out_csph1plus2 - out_csph1 - out_csph2).flatten()
	csph23_minus_csph2_3 = (out_csph2plus3 - out_csph2 - out_csph3).flatten()
	csph123_minus_csph1_2_3 = (out_csph1plus2plus3 - out_csph1 - out_csph2 - out_csph3).flatten()

	## Get some incorrect absolute difference  
	csph12_minus_csph1_3 = (out_csph1plus2 - out_csph1 - out_csph3).flatten()
	csph23_minus_csph1_2 = (out_csph2plus3 - out_csph1 - out_csph2).flatten()
	csph123_minus_csph1_2 = (out_csph1plus2plus3 - out_csph1 - out_csph2).flatten()

	print("csph12_minus_csph1_2:")
	print("    abs Absmean: {}".format(csph12_minus_csph1_2.abs().mean()))
	print("    abs median: {}".format(torch.median(csph12_minus_csph1_2.abs())))
	print("    abs AbsStd: {}".format(csph12_minus_csph1_2.abs().std()))
	print("csph23_minus_csph2_3:")
	print("    abs Absmean: {}".format(csph23_minus_csph2_3.abs().mean()))
	print("    abs median: {}".format(torch.median(csph23_minus_csph2_3.abs())))
	print("    abs AbsStd: {}".format(csph23_minus_csph2_3.abs().std()))
	print("csph123_minus_csph1_2_3:")
	print("    abs Absmean: {}".format(csph123_minus_csph1_2_3.abs().mean()))
	print("    abs median: {}".format(torch.median(csph123_minus_csph1_2_3.abs())))
	print("    abs AbsStd: {}".format(csph123_minus_csph1_2_3.abs().std()))


	out_fname = 'tinit-{}_inputs-{}-{}-{}-{}_encoding-{}-{}-{}-{}-{}_deterministic-{}'.format(tblock_init, inputs_type, nt, nr, nc, encoding_type, k, csph_layer.tblock_len, spatial_down_factor, spatial_down_factor, deterministic)
	plt.clf()
	plt.suptitle("Inputs type: {}, tblock_init: {}, encoding_type: {}, deterministic cuddnn: {}, \n encoding kernel: ({},{},{},{}), input dims: {}".format(inputs_type, tblock_init, encoding_type, deterministic, k, csph_layer.tblock_len, spatial_down_factor, spatial_down_factor, inputs1.shape),fontsize=16)
	plt.subplot(3,4,1)
	plt.plot(inputs1[0, 0, :, nr//2,nc//2], label='inputs1[0, 0, :, nr//2,nc//2]')
	plt.plot(inputs2[1, 0, :, 10, 12], label='inputs2[1, 0, :, 10, 12]', alpha=0.7)
	plt.title('Example Input Histograms')
	plt.legend()
	plt.subplot(3,4,2)
	plt.hist(out_csph1.detach().cpu().numpy().flatten() + out_csph2.detach().cpu().numpy().flatten(), bins=300, label='CSPH3D(x1) + CSPH3D(x2)', alpha=0.9)	
	plt.hist(out_csph1plus2.detach().cpu().numpy().flatten(), bins=300,  label='CSPH3D(x1+x2)', alpha=0.7)
	plt.hist(out_csph1.detach().cpu().numpy().flatten(), bins=300,  label='CSPH3D(x1)', alpha=0.5)
	plt.legend()
	plt.title("Full CSPH Values")
	plt.subplot(3,4,3)
	plt.hist(csph12_minus_csph1_2.detach().cpu().numpy(), bins=300, label='CSPH3D(x1+x2)\n-CSPH3D(x1)-CSPH3D(x2)', alpha=0.9)	
	plt.legend()
	plt.xlim((-5e-4, 5e-4))
	plt.title("CORRECT Diff - AbsMean: {:.4f}, AbsStd: {:.4f}".format(csph12_minus_csph1_2.abs().mean(), csph12_minus_csph1_2.abs().std()))
	plt.subplot(3,4,4)
	plt.hist(csph12_minus_csph1_3.detach().cpu().numpy(), bins=300, label='CSPH3D(x1+x2)\n-CSPH3D(x1)-CSPH3D(x3)', alpha=0.9)	
	plt.legend()
	plt.title("WRONG Diff - AbsMean: {:.1f}, AbsStd: {:.1f}".format(csph12_minus_csph1_3.abs().mean(), csph12_minus_csph1_3.abs().std()))
	plt.subplot(3,4,5)
	plt.plot(inputs2[0, 0, :, nr//2,nc//2], label='inputs2[0, 0, :, nr//2,nc//2]')
	plt.plot(inputs3[1, 0, :, 10, 12], label='inputs3[1, 0, :, 10, 12]', alpha=0.7)
	plt.title('Example Input Histograms')
	plt.legend()
	plt.subplot(3,4,6)
	plt.hist(out_csph2.detach().cpu().numpy().flatten() + out_csph3.detach().cpu().numpy().flatten(), bins=300, label='CSPH3D(x1) + CSPH3D(x2)', alpha=0.9)	
	plt.hist(out_csph2plus3.detach().cpu().numpy().flatten(), bins=300,  label='CSPH3D(x2+x3)', alpha=0.7)
	plt.hist(out_csph2.detach().cpu().numpy().flatten(), bins=300,  label='CSPH3D(x2)', alpha=0.5)
	plt.legend()
	plt.title("Full CSPH Values")
	plt.subplot(3,4,7)
	plt.hist(csph23_minus_csph2_3.detach().cpu().numpy(), bins=300, label='CSPH3D(x2+x3)\n-CSPH3D(x2)-CSPH3D(x3)', alpha=0.9)	
	plt.legend()
	plt.xlim((-5e-4, 5e-4))
	plt.title("CORRECT Diff - AbsMean: {:.4f}, AbsStd: {:.4f}".format(csph23_minus_csph2_3.abs().mean(), csph23_minus_csph2_3.abs().std()))
	plt.subplot(3,4,8)
	plt.hist(csph23_minus_csph1_2.detach().cpu().numpy(), bins=300, label='CSPH3D(x2+x3)\n-CSPH3D(x1)-CSPH3D(x2)', alpha=0.9)	
	plt.legend()
	plt.title("WRONG Diff - AbsMean: {:.1f}, AbsStd: {:.1f}".format(csph23_minus_csph1_2.abs().mean(), csph23_minus_csph1_2.abs().std()))
	plt.subplot(3,4,9)
	plt.plot(inputs1[0, 0, :, nr//2,nc//2], label='inputs1[0, 0, :, nr//2,nc//2]')
	plt.plot(inputs2[0, 0, :, 5, 5], label='inputs2[0, 0, :, 5, 5]', alpha= 0.7)
	plt.plot(inputs3[3, 0, :, 3, 21], label='inputs3[3, 0, :, 3, 21]', alpha=0.7)
	plt.title('Example Input Histograms')
	plt.legend()
	plt.subplot(3,4,10)
	plt.hist(out_csph1.detach().cpu().numpy().flatten() + out_csph2.detach().cpu().numpy().flatten() + out_csph3.detach().cpu().numpy().flatten(), bins=300, label='CSPH3D(x1) + CSPH3D(x2) + CSPH3D(x3)', alpha=0.9)	
	plt.hist(out_csph1plus2plus3.detach().cpu().numpy().flatten(), bins=300,  label='CSPH3D(x1+x2+x3)', alpha=0.7)
	plt.hist(out_csph3.detach().cpu().numpy().flatten(), bins=300,  label='CSPH3D(x3)', alpha=0.5)
	plt.legend()
	plt.title("Full CSPH Values")
	plt.subplot(3,4,11)
	plt.hist(csph123_minus_csph1_2_3.detach().cpu().numpy(), bins=300, label='CSPH3D(x1+x2+x3)\n-CSPH3D(x1)-CSPH3D(x2)-CSPH3D(x3)', alpha=0.9)	
	plt.legend()
	plt.xlim((-5e-4, 5e-4))
	plt.title("CORRECT Diff - AbsMean: {:.4f}, AbsStd: {:.4f}".format(csph123_minus_csph1_2_3.abs().mean(), csph123_minus_csph1_2_3.abs().std()))
	plt.subplot(3,4,12)
	plt.hist(csph123_minus_csph1_2.detach().cpu().numpy(), bins=300, label='CSPH3D(x1+x2+x3)\n-CSPH3D(x1)-CSPH3D(x2)', alpha=0.9)	
	plt.legend()
	plt.title("WRONG Diff - AbsMean: {:.1f}, AbsStd: {:.1f}".format(csph123_minus_csph1_2.abs().mean(), csph123_minus_csph1_2.abs().std()))

	plot_utils.save_currfig_png(dirpath=out_dirpath, filename=out_fname)


