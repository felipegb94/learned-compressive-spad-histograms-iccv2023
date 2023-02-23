#### Standard Library Imports
import os

#### Library imports
import numpy as np
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
	
	## 80 ps dataset 256x4x4 separable
	model_ckpt_fpath = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models/DDFN_C64B10_CSPH3D/k32_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-09-24_175849/checkpoints/epoch=29-step=103889-avgvalrmse=0.0208.ckpt'


	## 80 ps dataset 256x2x2 K=8 separable trunc fourier
	model_ckpt_fpath = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models_20230218/DDFN_C64B10_CSPH3D/k8_down2_Mt4_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-19_010108/checkpoints/epoch=29-step=101292-avgvalrmse=0.0208.ckpt'

	# ## 80 ps dataset 256x2x2 K=16 separable trunc fourier
	# model_ckpt_fpath = 'outputs/nyuv2_64x64x1024_80ps/csph3d_models_20230218/DDFN_C64B10_CSPH3D/k16_down2_Mt4_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-19_010116/checkpoints/epoch=29-step=103887-avgvalrmse=0.0188.ckpt'


	# ## 80 ps dataset 256x1x1 K=4  trunc fourier
	# model_ckpt_fpath = 'outputs/nyuv2_64x64x1024_80ps/csph3D_tdim-256_20230218/DDFN_C64B10_CSPH3D/k4_down1_Mt4_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-19_011009/checkpoints/epoch=29-step=103022-avgvalrmse=0.0203.ckpt'

	# # ## 80 ps dataset 256x1x1 K=8 trunc fourier
	# model_ckpt_fpath = 'outputs/nyuv2_64x64x1024_80ps/csph3D_tdim-256_20230218/DDFN_C64B10_CSPH3D/k8_down1_Mt4_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-19_011012/checkpoints/epoch=29-step=102157-avgvalrmse=0.0173.ckpt'

	# ## 80 ps dataset 1024x1x1 K=8  trunc fourier
	# model_ckpt_fpath = 'outputs/nyuv2_64x64x1024_80ps/csph3D_tdim_baselines/DDFN_C64B10_CSPH3D/k8_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-10-04_002521/checkpoints/epoch=29-step=103889-avgvalrmse=0.0193.ckpt'

	# ## 80 ps dataset 1024x1x1 K=16 Trunc Fourier
	# model_ckpt_fpath = 'outputs/nyuv2_64x64x1024_80ps/csph3D_tdim_baselines/DDFN_C64B10_CSPH3D/k16_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2022-09-22_104615/checkpoints/epoch=29-step=102158-avgvalrmse=0.0162.ckpt'

	## Load model
	model_ckpt_fname = os.path.basename(model_ckpt_fpath)
	model_dirpath = model_ckpt_fpath.split('/checkpoints/')[0]
	model_name = '/'.join(model_dirpath.split('/')[3:-1])

	## dirpath to save the quantized weights
	out_dirpath = os.path.join(model_dirpath, "csph3d_layer_qint8")
	os.makedirs(out_dirpath, exist_ok=True)
	out_fname = model_ckpt_fname.split('.ckpt')[0]
	out_fpath = os.path.join(out_dirpath, out_fname)

	model, _ = load_model_from_ckpt(model_name, model_ckpt_fname, logger=None, model_dirpath=model_dirpath)
	
	encoding_layer = model.csph3d_layer
	encoding_type = model.csph3d_layer.encoding_type
	K = encoding_layer.k
	if(encoding_type == 'separable'):
		Cmat_tdim = encoding_layer.Cmat_tdim.detach().cpu().numpy()
		Cmat_xydim = encoding_layer.Cmat_xydim.detach().cpu().numpy()
		Cmat_txydim = Cmat_xydim*Cmat_tdim
	elif(encoding_type == 'full'):
		Cmat_txydim = encoding_layer.Cmat_txydim.detach().cpu().numpy()
		Cmat_tdim = Cmat_txydim.mean(axis=(-1,-2), keepdims=True)
		Cmat_xydim = Cmat_txydim.mean(axis=(-3), keepdims=True)
	elif(encoding_type == 'csph1d'):
		Cmat_tdim = encoding_layer.Cmat_tdim.detach().cpu().numpy()
		Cmat_xydim = np.ones((K, 1, 1, 1, 1)).astype(Cmat_tdim.dtype)
		Cmat_txydim = Cmat_xydim*Cmat_tdim
	else:
		assert(False), "script is not setup for any other encoding type."


	X_range = (-1,1)
	(Cmat_tdim_int8, Cmat_tdim_scale, Cmat_tdim_zero_point) = quantize_qint8_manual(Cmat_tdim, X_range=X_range) 
	(Cmat_xydim_int8, Cmat_xydim_scale, Cmat_xydim_zero_point) = quantize_qint8_manual(Cmat_xydim, X_range=X_range) 
	(Cmat_txydim_int8, Cmat_txydim_scale, Cmat_txydim_zero_point) = quantize_qint8_manual(Cmat_txydim, X_range=X_range) 
	
	(Cmat_tdim_fp32) = dequantize_qint8(Cmat_tdim_int8, Cmat_tdim_scale, Cmat_tdim_zero_point) 
	(Cmat_xydim_fp32) = dequantize_qint8(Cmat_xydim_int8, Cmat_xydim_scale, Cmat_xydim_zero_point) 
	(Cmat_txydim_fp32) = dequantize_qint8(Cmat_txydim_int8, Cmat_txydim_scale, Cmat_txydim_zero_point)

	## save the file
	#NOTE: scale and zero_point shpould match for all if X_range is fixed
	np.savez(out_fpath 
	  , range_float32=X_range
	  , range_int8=(-128,127)
	  , Cmat_tdim_int8=Cmat_tdim_int8.squeeze()
	  , Cmat_tdim=Cmat_tdim.squeeze()
	  , Cmat_xydim_int8=Cmat_xydim_int8.squeeze()
	  , Cmat_xydim=Cmat_xydim.squeeze()
	  , Cmat_txydim_int8=Cmat_txydim_int8.squeeze()
	  , Cmat_txydim=Cmat_txydim.squeeze()
	  , scale=Cmat_tdim_scale
	  , zero_point=Cmat_tdim_zero_point
	  )


	plt.clf()
	plt.subplot(2,1,1)
	plt.title("Original FP32")
	plt.plot(Cmat_tdim.squeeze().transpose())
	plt.subplot(2,1,2)
	plt.title("Dequantized FP32")
	plt.plot(Cmat_tdim_fp32.squeeze().transpose())


	## VALIDATE Quantization Code

	## Generate between X_min and X_max
	(min_val, max_val) = (-0.5, 0.5)
	X_fp32_1 = (min_val - max_val) * torch.rand(20) + max_val
	# X_fp32_1 = (min_val - max_val) * np.random.rand(10).astype(np.float32) + max_val
	# X_fp32_1_torch = torch.tensor(X_fp32_1)
	# X_fp32_1 = torch.tensor([-0.3, -0.2, 0.1, 0.4])
	# X_fp32_1 = torch.tensor([-0.4, -0.26, -0.2, 0.01, 0.05, 0.1, 0.15, 0.3])
	print("Output of torch quantize and dequantize:")
	print("    Initial fp32: {}".format(X_fp32_1))
	X_qint8_1 = quantize_qint8(X_fp32_1, X_range)
	print("    Torch Quantized int8: {}".format(X_qint8_1))
	print("    Torch DeQuantized fp32: {}".format(X_qint8_1.dequantize()))
	print("Output of manual quantize and dequantize:")
	print("    Initial fp32: {}".format(X_fp32_1))
	(X_qint8_2, X_qint8_2_scale, X_qint8_2_zero_point) = quantize_qint8_manual(X_fp32_1, X_range)
	print("    Manual Quantized int8: {}".format(X_qint8_2))
	X_fp32_2 = dequantize_qint8(X_qint8_2, X_qint8_2_scale, X_qint8_2_zero_point)
	print("    Manual DeQuantized fp32: {}".format(X_fp32_2))

	assert(np.all((X_qint8_1.dequantize() == X_fp32_2).numpy())), "manual quantization does not match pytorch"



