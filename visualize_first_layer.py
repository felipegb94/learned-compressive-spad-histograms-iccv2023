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

	db3D_csphk64_separable2x2x1024_grayfour_tv0_model_name = 'DDFN_C64B10_CSPH/k64_down2_Mt1_HybridGrayFourier-opt-False/loss-kldiv_tv-0.0'
	db3D_csphk128_separable4x4x1024_truncfour_tv0_model_name = 'DDFN_C64B10_CSPH/k128_down4_Mt1_TruncFourier-opt-False/loss-kldiv_tv-0.0'
	db3D_csphk64_separable4x4x512_truncfour_tv0_model_name = 'DDFN_C64B10_CSPH/k64_down4_Mt2_TruncFourier-opt-False/loss-kldiv_tv-0.0'

	db3D_csphk64_separable2x2x1024_grayfour_tv0_result_dirpath = os.path.join(base_dirpath, db3D_csphk64_separable2x2x1024_grayfour_tv0_model_name, '2022-06-18_183013/')
	db3D_csphk128_separable4x4x1024_truncfour_tv0_result_dirpath = os.path.join(base_dirpath, db3D_csphk128_separable4x4x1024_truncfour_tv0_model_name, '2022-06-18_203447/')
	db3D_csphk64_separable4x4x512_truncfour_tv0_result_dirpath = os.path.join(base_dirpath, db3D_csphk64_separable4x4x512_truncfour_tv0_model_name, '2022-06-20_111348')

	# model_name = db3D_csphk64_separable2x2x1024_grayfour_tv0_model_name
	# model_dirpath = db3D_csphk64_separable2x2x1024_grayfour_tv0_result_dirpath
	# ckpt_fname = 'epoch=00-step=1731-avgvalrmse=0.0324.ckpt'
	# ckpt_fname = 'epoch=18-step=64065-avgvalrmse=0.0161.ckpt'

	# model_name = db3D_csphk128_separable4x4x1024_truncfour_tv0_model_name
	# model_dirpath = db3D_csphk128_separable4x4x1024_truncfour_tv0_result_dirpath
	# ckpt_fname = 'epoch=00-step=1731-avgvalrmse=0.0324.ckpt'
	# ckpt_fname = 'epoch=18-step=64065-avgvalrmse=0.0161.ckpt'

	model_name = db3D_csphk64_separable4x4x512_truncfour_tv0_model_name
	model_dirpath = db3D_csphk64_separable4x4x512_truncfour_tv0_result_dirpath
	ckpt_fname = 'epoch=00-step=1731-avgvalrmse=0.0363'
	# ckpt_fname = 'epoch=19-step=69259-avgvalrmse=0.0205'

	if(not ('.ckpt' in ckpt_fname)): ckpt_fname += '.ckpt' 
	model_ckpt_fpath = os.path.join(model_dirpath, 'checkpoints/', ckpt_fname)
	model, ckpt_fpath = load_model_from_ckpt(model_name, ckpt_fname, logger=None, model_dirpath=model_dirpath)


	encoding_layer = model.csph_encoding_layer

	Cmat_tdim = encoding_layer.Cmat_tdim.weight.squeeze().detach().cpu().numpy()
	Cmat_xydim = encoding_layer.Cmat_xydim.weight.squeeze().detach().cpu().numpy()

	# Plot the first 4 time domain codes
	plt.clf()
	plt.plot(Cmat_tdim.transpose()[:,0:10:2])

	plt.title("Model: {}, ckpt: {}".format(model_name, ckpt_fname))

	



