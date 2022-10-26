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


	base_dirpath = 'outputs/nyuv2_64x64x1024_80ps/test_csph'

	# db3D_csphk64_separable2x2x1024_grayfour_opt_tv0_model_name = 'DDFN_C64B10_CSPH/k64_down2_Mt1_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0'
	# db3D_csphk64_separable2x2x1024_grayfour_opt_tv0_result_dirpath = os.path.join(base_dirpath, db3D_csphk64_separable2x2x1024_grayfour_opt_tv0_model_name, '2022-06-18_183013/')
	# model_name = db3D_csphk64_separable2x2x1024_grayfour_opt_tv0_model_name
	# model_dirpath = db3D_csphk64_separable2x2x1024_grayfour_opt_tv0_result_dirpath
	# ckpt_fname = 'epoch=00-step=1731-avgvalrmse=0.0324.ckpt'
	# ckpt_fname = 'epoch=18-step=64065-avgvalrmse=0.0161.ckpt'

	# db3D_csphk128_separable4x4x1024_truncfour_opt_tv0_model_name = 'DDFN_C64B10_CSPH/k128_down4_Mt1_TruncFourier-opt-True_separable/loss-kldiv_tv-0.0'
	# db3D_csphk128_separable4x4x1024_truncfour_opt_tv0_result_dirpath = os.path.join(base_dirpath, db3D_csphk128_separable4x4x1024_truncfour_opt_tv0_model_name, '2022-06-18_203447/')
	# model_name = db3D_csphk128_separable4x4x1024_truncfour_opt_tv0_model_name
	# model_dirpath = db3D_csphk128_separable4x4x1024_truncfour_opt_tv0_result_dirpath
	# ckpt_fname = 'epoch=17-step=62333-avgvalrmse=0.0197'

	# db3D_csphk32_separable2x2x1024_grayfour_opt_tv0_model_name = 'DDFN_C64B10_CSPH/k32_down2_Mt1_HybridGrayFourier-opt-True_separable/loss-kldiv_tv-0.0'
	# db3D_csphk32_separable2x2x1024_grayfour_opt_tv0_result_dirpath = os.path.join(base_dirpath, db3D_csphk32_separable2x2x1024_grayfour_opt_tv0_model_name, '2022-06-21_230853/')
	# model_name = db3D_csphk32_separable2x2x1024_grayfour_opt_tv0_model_name
	# model_dirpath = db3D_csphk32_separable2x2x1024_grayfour_opt_tv0_result_dirpath
	# ckpt_fname = 'epoch=19-step=67528-avgvalrmse=0.0174'

	# db3D_csphk128_separable4x4x1024_truncfour_opt_tv0_model_name = 'DDFN_C64B10_CSPH/k128_down4_Mt1_TruncFourier-opt-True_separable/loss-kldiv_tv-0.0'
	# db3D_csphk128_separable4x4x1024_truncfour_opt_tv0_result_dirpath = os.path.join(base_dirpath, db3D_csphk128_separable4x4x1024_truncfour_opt_tv0_model_name, '2022-06-18_203447/')
	# model_name = db3D_csphk128_separable4x4x1024_truncfour_opt_tv0_model_name
	# model_dirpath = db3D_csphk128_separable4x4x1024_truncfour_opt_tv0_result_dirpath
	# ckpt_fname = 'epoch=17-step=62333-avgvalrmse=0.0197'

	# db3D_csphk256_separable4x4x1024_truncfour_opt_tv0_model_name = 'DDFN_C64B10_CSPH/k256_down4_Mt1_TruncFourier-opt-True_separable/loss-kldiv_tv-0.0'
	# db3D_csphk256_separable4x4x1024_truncfour_opt_tv0_result_dirpath = os.path.join(base_dirpath, db3D_csphk256_separable4x4x1024_truncfour_opt_tv0_model_name, '2022-06-22_130057/')
	# model_name = db3D_csphk256_separable4x4x1024_truncfour_opt_tv0_model_name
	# model_dirpath = db3D_csphk256_separable4x4x1024_truncfour_opt_tv0_result_dirpath
	# ckpt_fname = 'epoch=19-step=67528-avgvalrmse=0.0194'


	# db3D_csphk64_separable4x4x512_truncfour_opt_tv0_model_name = 'DDFN_C64B10_CSPH/k64_down4_Mt2_TruncFourier-opt-True_separable/loss-kldiv_tv-0.0'
	# db3D_csphk64_separable4x4x512_truncfour_opt_tv0_result_dirpath = os.path.join(base_dirpath, db3D_csphk64_separable4x4x512_truncfour_opt_tv0_model_name, '2022-06-20_111348')
	# model_name = db3D_csphk64_separable4x4x512_truncfour_opt_tv0_model_name
	# model_dirpath = db3D_csphk64_separable4x4x512_truncfour_opt_tv0_result_dirpath
	# ckpt_fname = 'epoch=00-step=1731-avgvalrmse=0.0363'
	# ckpt_fname = 'epoch=19-step=69259-avgvalrmse=0.0205'

	# db3D_csphk128_separable4x4x1024_grayfour_tv0_model_name = 'DDFN_C64B10_CSPH/k128_down4_Mt1_HybridGrayFourier-opt-False_separable/loss-kldiv_tv-0.0'
	# db3D_csphk128_separable4x4x1024_grayfour_tv0_result_dirpath = os.path.join(base_dirpath, db3D_csphk128_separable4x4x1024_grayfour_tv0_model_name, '2022-06-23_170831/')
	# ckpt_fname = 'epoch=00-step=692-avgvalrmse=0.0428'
	# model_name = db3D_csphk128_separable4x4x1024_grayfour_tv0_model_name
	# model_dirpath = db3D_csphk128_separable4x4x1024_grayfour_tv0_result_dirpath

	# db3D_csphk128_separable4x4x1024_rand_opt_tv0_model_name = 'DDFN_C64B10_CSPH/k128_down4_Mt1_Rand-opt-True_separable/loss-kldiv_tv-0.0'
	# db3D_csphk128_separable4x4x1024_rand_opt_tv0_result_dirpath = os.path.join(base_dirpath, db3D_csphk128_separable4x4x1024_rand_opt_tv0_model_name, '2022-06-23_174622/')
	# ckpt_fname = 'epoch=00-step=1038-avgvalrmse=0.0452'
	# model_name = db3D_csphk128_separable4x4x1024_rand_opt_tv0_model_name
	# model_dirpath = db3D_csphk128_separable4x4x1024_rand_opt_tv0_result_dirpath
	
	# db3D_csphk128_full4x4x1024_rand_opt_tv0_model_name = 'DDFN_C64B10_CSPH/k128_down4_Mt1_Rand-opt-True_full/loss-kldiv_tv-0.0'
	# db3D_csphk128_full4x4x1024_rand_opt_tv0_result_dirpath = os.path.join(base_dirpath, db3D_csphk128_full4x4x1024_rand_opt_tv0_model_name, '2022-06-23_184602/')
	# # ckpt_fname = 'epoch=00-step=2422-avgvalrmse=0.0365'
	# ckpt_fname = 'epoch=00-step=346-avgvalrmse=0.0619'
	# model_name = db3D_csphk128_full4x4x1024_rand_opt_tv0_model_name
	# model_dirpath = db3D_csphk128_full4x4x1024_rand_opt_tv0_result_dirpath


	# db3D_csphk128_full4x4x1024_grayfour_opt_tv0_model_name = 'DDFN_C64B10_CSPH/k128_down4_Mt1_HybridGrayFourier-opt-True_full/loss-kldiv_tv-0.0'
	# db3D_csphk128_full4x4x1024_grayfour_opt_tv0_result_dirpath = os.path.join(base_dirpath, db3D_csphk128_full4x4x1024_grayfour_opt_tv0_model_name, '2022-06-23_222403/')
	# ckpt_fname = 'epoch=00-step=346-avgvalrmse=0.2627'
	# model_name = db3D_csphk128_full4x4x1024_grayfour_opt_tv0_model_name
	# model_dirpath = db3D_csphk128_full4x4x1024_grayfour_opt_tv0_result_dirpath


	if(not ('.ckpt' in ckpt_fname)): ckpt_fname += '.ckpt' 
	model_ckpt_fpath = os.path.join(model_dirpath, 'checkpoints/', ckpt_fname)
	model, ckpt_fpath = load_model_from_ckpt(model_name, ckpt_fname, logger=None, model_dirpath=model_dirpath)


	encoding_layer = model.csph_encoding_layer
	if('separable' in  model_name):
		Cmat_tdim = encoding_layer.Cmat_tdim.weight.squeeze().detach().cpu().numpy()
		Cmat_xydim = encoding_layer.Cmat_xydim.weight.squeeze().detach().cpu().numpy()
		Cmat_tdim_init = encoding_layer.Cmat_tdim_init.squeeze()
	else:
		Cmat_txydim = encoding_layer.Cmat_txydim.weight.squeeze().detach().cpu().numpy()
		Cmat_tdim = Cmat_txydim[:,:,0,0]
		if(len(encoding_layer.Cmat_tdim_init.shape) > 2):
			Cmat_txydim_init = encoding_layer.Cmat_tdim_init.squeeze()
			Cmat_tdim_init = Cmat_txydim_init[:,:,0,0].transpose()
		else:
			Cmat_tdim_init = encoding_layer.Cmat_tdim_init.squeeze()

	if(('Rand' in model_name)): Cmat_tdim_init = Cmat_tdim_init.transpose()

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
	



