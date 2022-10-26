#### Standard Library Imports
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from research_utils import plot_utils, np_utils
from model_utils import load_model_from_ckpt

if __name__=='__main__':


	base_dirpath = 'outputs/nyuv2_64x64x1024_80ps'

	# 3D CSPH 1024x4x4 - Compression=32x --> k=512
	model_name="DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0"
	experiment_name="csph3d_models"
	model_dirpath="{}/{}/{}/run-complete_2022-08-30_125727".format(base_dirpath, experiment_name, model_name)
	ckpt_fname="epoch=29-step=103889-avgvalrmse=0.0172.ckpt"

	# model_name="DDFN_C64B10_CSPH3D/k128_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0"
	# experiment_name="csph3d_models"
	# model_dirpath="{}/{}/{}/run-complete_2022-09-25_174558".format(base_dirpath, experiment_name, model_name)
	# ckpt_fname="epoch=29-step=103889-avgvalrmse=0.0169.ckpt" #  | 120 images== | 128 images==0.01627

	model_name = "DDFN_C64B10_CSPH3D/k16_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0"
	experiment_name="csph3D_tdim_baselines"
	model_dirpath="{}/{}/{}/run-complete_2022-09-28_223938".format(base_dirpath, experiment_name, model_name)
	ckpt_fname = "epoch=29-step=103889-avgvalrmse=0.0179.ckpt"


	if(not ('.ckpt' in ckpt_fname)): ckpt_fname += '.ckpt' 
	model_ckpt_fpath = os.path.join(model_dirpath, 'checkpoints/', ckpt_fname)
	model, ckpt_fpath = load_model_from_ckpt(model_name, ckpt_fname, logger=None, model_dirpath=model_dirpath)

	encoding_layer = model.csph3d_layer
	if('separable' in model_name):
		Cmat_tdim = encoding_layer.Cmat_tdim.data.squeeze().detach().cpu().numpy()
		Cmat_xydim = encoding_layer.Cmat_xydim.data.squeeze().detach().cpu().numpy()
		Cmat_txydim = Cmat_tdim[...,np.newaxis,np.newaxis]*Cmat_xydim[:,np.newaxis,:] 
		Cmat_tdim_init = encoding_layer.Cmat_tdim_init.squeeze()
	elif('csph1d' in model_name):
		Cmat_tdim = encoding_layer.Cmat_tdim.data.squeeze().detach().cpu().numpy()
		Cmat_xydim = 1
		Cmat_txydim = Cmat_tdim[..., np.newaxis, np.newaxis]
		Cmat_tdim_init = encoding_layer.Cmat_tdim_init.squeeze()
	else:
		Cmat_txydim = encoding_layer.Cmat_txydim.data.squeeze().detach().cpu().numpy()
		Cmat_tdim = Cmat_txydim[:,:,0,0]
		if(len(encoding_layer.Cmat_tdim_init.shape) > 2):
			Cmat_txydim_init = encoding_layer.Cmat_tdim_init.squeeze()
			Cmat_tdim_init = Cmat_txydim_init[:,:,0,0].transpose()
		else:
			Cmat_tdim_init = encoding_layer.Cmat_tdim_init.squeeze()

	if(('Rand' in model_name)): Cmat_tdim_init = Cmat_tdim_init.transpose()

	## Save coding matrix as npy object
	out_cmat_dirpath = os.path.join(model_dirpath, "coding_mats")
	os.makedirs(out_cmat_dirpath, exist_ok=True)
	np.save(os.path.join(out_cmat_dirpath, "cmat_"+ckpt_fname), Cmat_txydim)


	if(Cmat_tdim.max() < 0.25): ylim=(-0.25,0.25)
	elif(Cmat_tdim.max() < 0.5): ylim=(-0.5,0.5)
	elif(Cmat_tdim.max() < 1.0): ylim=(-1.0,1.0)
	else: ylim=(-2.0,2.0)

	# Plot the first 4 time domain codes
	plt.clf()
	plt.subplot(2,1,1)
	plt.plot(Cmat_tdim.transpose()[:,1:20:4])
	plt.ylim(ylim)
	plt.title("Model: {}, \n ckpt: {}".format(model_name.split('/')[1], ckpt_fname))
	plt.subplot(2,1,2)
	plt.plot(Cmat_tdim_init[:,0:10:2])
	plt.ylim(ylim)
	plt.title("Init Codes")
	



