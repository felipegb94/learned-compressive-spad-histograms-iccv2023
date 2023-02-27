#### Standard Library Imports
import os

#### Library imports
import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt
import torch
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from tof_utils import *
from model_utils import load_model_from_ckpt


def compute_scale(beta, alpha, beta_q, alpha_q):
	return (float(beta) - float(alpha)) / (float(beta_q) - float(alpha_q))
	# return ((beta) - (alpha)) / ((beta_q) - (alpha_q))

def compute_zero_point(scale, alpha, alpha_q):
	## Cast everything to float first to make sure that we dont input torch.tensors to zero point
	return round(-1*((float(alpha)/float(scale)) - float(alpha_q)))
## 
def quantize_qint8(X, X_range=None):
	if(X_range is None):
		(X_min, X_max) = (X.min(), X.max())
	else:
		X_min = X_range[0]
		X_max = X_range[1]
	print("torch min: {}".format(X_min))
	print("torch max: {}".format(X_max))
	(min_q, max_q) = (-128, 127)
	scale = compute_scale(X_max, X_min, max_q, min_q)
	zero_point = compute_zero_point(scale, X_min, min_q)
	return torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point, dtype=torch.qint8)

def quantize_qint8_manual(X, X_range=None):
	if(X_range is None):
		(X_min, X_max) = (X.min(), X.max())
	else:
		X_min = X_range[0]
		X_max = X_range[1]
		assert(X_min <= X.min()), "minimum should be contained in range"
		assert(X_max >= X.max()), "maximum should be contained in range"
	print("manual min: {}".format(X_min))
	print("manual max: {}".format(X_max))
	(min_q, max_q) = (-128, 127)
	scale = compute_scale(X_max, X_min, max_q, min_q)
	zero_point = compute_zero_point(scale, X_min, min_q)
	qX = (X / scale) + zero_point
	if(isinstance(qX, np.ndarray)):
		qX_int8  = np.round(qX).astype(np.int8)
	else:
		qX_int8  = torch.round(qX).type(torch.int8)

	return  (qX_int8, scale, zero_point)

def dequantize_qint8(X_qint8, X_scale, X_zero_point):
	## Cast everything to float 32 first since X_zero_point may have different precision than X_qint8
	if(isinstance(X_qint8, np.ndarray)):
		return (X_qint8.astype(np.float32) - float(X_zero_point))*float(X_scale)
	else:
		return (X_qint8.type(torch.float32) - float(X_zero_point))*float(X_scale)
		

if __name__=='__main__':

	## Test scene to load
	test_data_fpath = 'data_gener/TestData/middlebury/processed/SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0/spad_Art_50_1000.mat'
	# test_data_fpath = 'data_gener/TestData/middlebury/processed/SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0/spad_Art_10_10.mat'
	# test_data_fpath = 'data_gener/TestData/middlebury/processed/SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0/spad_Reindeer_50_1000.mat'
	test_data = scipy.io.loadmat(test_data_fpath)
	spad_np = np.asarray(scipy.sparse.csc_matrix.todense(test_data['spad']))
	spad_np = spad_np.reshape((88, 72, 1024))
	spad_np = spad_np[np.newaxis, :]
	spad_np = np.transpose(spad_np, (0, 3, 2, 1))
	spad = torch.tensor(spad_np).unsqueeze(0)

	## 80 ps dataset 256x2x2 K=8 separable trunc fourier
	model_ckpt_fpath = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models_20230218/DDFN_C64B10_CSPH3D/k8_down2_Mt4_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-19_010108/checkpoints/epoch=29-step=101292-avgvalrmse=0.0208.ckpt'

	# ## 80 ps dataset 256x2x2 K=16 separable trunc fourier
	# model_ckpt_fpath = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models_20230218/DDFN_C64B10_CSPH3D/k16_down2_Mt4_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-19_010116/checkpoints/epoch=29-step=103887-avgvalrmse=0.0188.ckpt'

	# ## 80 ps dataset 1024x1x1 K=8  trunc fourier
	# model_ckpt_fpath = 'outputs/nyuv2_64x64x1024_80ps/csph3D_tdim_baselines/DDFN_C64B10_CSPH3D/k8_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-10-04_002521/checkpoints/epoch=29-step=103889-avgvalrmse=0.0193.ckpt'

	# ## 80 ps dataset 1024x1x1 K=16 Trunc Fourier
	# model_ckpt_fpath = 'outputs/nyuv2_64x64x1024_80ps/csph3D_tdim_baselines/DDFN_C64B10_CSPH3D/k16_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-09-22_104615/checkpoints/epoch=29-step=102158-avgvalrmse=0.0162.ckpt'

	## Load model
	model_ckpt_fname = os.path.basename(model_ckpt_fpath)
	model_dirpath = model_ckpt_fpath.split('/checkpoints/')[0]
	model_name = '/'.join(model_dirpath.split('/')[3:-1])


	model, _ = load_model_from_ckpt(model_name, model_ckpt_fname, logger=None, model_dirpath=model_dirpath)
	
	encoding_layer = model.csph3d_layer
	encoding_type = model.csph3d_layer.encoding_type
	K = encoding_layer.k
	if(encoding_type == 'separable'):
		Cmat_tdim = encoding_layer.Cmat_tdim
		Cmat_xydim = encoding_layer.Cmat_xydim
		Cmat_txydim = Cmat_xydim*Cmat_tdim
	elif(encoding_type == 'full'):
		Cmat_txydim = encoding_layer.Cmat_txydim
		Cmat_tdim = Cmat_txydim.mean(axis=(-1,-2), keepdims=True)
		Cmat_xydim = Cmat_txydim.mean(axis=(-3), keepdims=True)
	elif(encoding_type == 'csph1d'):
		Cmat_tdim = encoding_layer.Cmat_tdim
		Cmat_xydim = np.ones((K, 1, 1, 1, 1)).astype(Cmat_tdim.dtype)
		Cmat_txydim = Cmat_xydim*Cmat_tdim
	else:
		assert(False), "script is not setup for any other encoding type."


	X_range = (-1,1)
	(Cmat_tdim_int8, Cmat_tdim_scale, Cmat_tdim_zero_point) = quantize_qint8_manual(Cmat_tdim, X_range=X_range) 
	(Cmat_xydim_int8, Cmat_xydim_scale, Cmat_xydim_zero_point) = quantize_qint8_manual(Cmat_xydim, X_range=X_range) 
	(Cmat_txydim_int8, Cmat_txydim_scale, Cmat_txydim_zero_point) = quantize_qint8_manual(Cmat_txydim, X_range=X_range) 
	
	## Run regular encoding without quantization
	B = encoding_layer.csph_layer(spad.type(torch.float32))
	W = encoding_layer.get_unfilt_backproj_W3d()
	B_norm = encoding_layer.zncc_norm(B, dims=-4) # for 3D signals the channel dimension is -4 dimension 
	W_norm = encoding_layer.zncc_norm(W, dims=0) # for weights the channel dimension is the first channel
	X = encoding_layer.unfiltered_backproj_layer(y=B_norm, W=W_norm)

	## Run encoding with quantization
	# convert inputs to int8 (can simply cast to int8 since spad inputs are already integers btwn 0-255)
	spad_int8_dequantized = spad.type(torch.int8).type(torch.float32)
	Cmat_tdim_int8_dequantized = dequantize_qint8(Cmat_tdim_int8, Cmat_tdim_scale, Cmat_tdim_zero_point)
	Cmat_xydim_int8_dequantized = dequantize_qint8(Cmat_xydim_int8, Cmat_xydim_scale, Cmat_xydim_zero_point)
	# quantized csph layer
	import torch.nn.functional as F
	B1_quantized = F.conv3d(spad_int8_dequantized, Cmat_tdim_int8_dequantized, bias=None, stride=encoding_layer.encoding_kernel3d_t, padding=0, dilation = 1, groups = 1)
	B_quantized = F.conv3d(B1_quantized, Cmat_xydim_int8_dequantized, bias=None, stride=encoding_layer.encoding_kernel3d_xy, padding=0, dilation = 1, groups = encoding_layer.k)

	## Run encoding with quantization
	B_quantized2 = encoding_layer.csph_layer_separable_quantized(spad.type(torch.float32))


	## Dequantize the weights and view outputs
	(Cmat_tdim_fp32) = dequantize_qint8(Cmat_tdim_int8, Cmat_tdim_scale, Cmat_tdim_zero_point) 
	(Cmat_xydim_fp32) = dequantize_qint8(Cmat_xydim_int8, Cmat_xydim_scale, Cmat_xydim_zero_point) 
	(Cmat_txydim_fp32) = dequantize_qint8(Cmat_txydim_int8, Cmat_txydim_scale, Cmat_txydim_zero_point)

	## Make everything numpy arrays to visualize
	Cmat_tdim_np = Cmat_tdim.detach().cpu().numpy()
	Cmat_tdim_int8_np = Cmat_tdim_int8.detach().cpu().numpy()
	Cmat_tdim_fp32_np = Cmat_tdim_fp32.detach().cpu().numpy()
	Cmat_xydim_np = Cmat_xydim.detach().cpu().numpy()
	Cmat_xydim_int8_np = Cmat_xydim_int8.detach().cpu().numpy()
	Cmat_xydim_fp32_np = Cmat_xydim_fp32.detach().cpu().numpy()
	Cmat_txydim_np = Cmat_txydim.detach().cpu().numpy()
	Cmat_txydim_int8_np = Cmat_txydim_int8.detach().cpu().numpy()
	Cmat_txydim_fp32_np = Cmat_txydim_fp32.detach().cpu().numpy()

	plt.clf()
	plt.subplot(2,1,1)
	plt.title("Original FP32")
	plt.plot(Cmat_tdim_np.squeeze().transpose())
	plt.subplot(2,1,2)
	plt.title("Dequantized FP32")
	plt.plot(Cmat_tdim_fp32_np.squeeze().transpose())

