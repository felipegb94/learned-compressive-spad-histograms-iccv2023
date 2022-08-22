'''
	This script compares the results of the same CSPH3D model that processes the same input in two different ways: 
	Original input size == (1536x256x256)
	Spatial crop (for memory issues) == (1536x176x176)

	1) Crop time dimension of input and process normally, i.e., crop to 1024x176x176

	2) Pad time dimension of input, process normally, and crop output, i.e., 
		2.1) Pad to 2048x176x176 (use circular padding since tdim is periodic)
		2.2) Process
		2.3) Crop output back to 1536x176x176

'''
#### Standard Library Imports
import sys
import os
sys.path.append('../')
sys.path.append('./')

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

#### Local imports
from research_utils import plot_utils

if __name__=='__main__':

	out_results_dirpath = 'results/week_2022-08-15/csph3d_with_diff_sized_inputs'

	## Set inputs
	scene_id = 'elephant'
	scene_id = 'kitchen'
	scene_id = 'lamp'
	cropped_inputs_result_dir = 'outputs/nyuv2_64x64x1024_80ps/test_csph3d/DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0/2022-08-10_162141/test_lindell2018_linospad_captured_min_cropped_inputs' 
	padded_inputs_result_dir = 'outputs/nyuv2_64x64x1024_80ps/test_csph3d/DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0/2022-08-10_162141/test_lindell2018_linospad_captured_min_padded_inputs'
	# padded_inputs_result_dir = 'outputs/nyuv2_64x64x1024_80ps/test_csph3d/DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0/2022-08-10_162141/test_lindell2018_linospad_captured_min'

	## get intensity image
	raw_spad_data_dir = '/home/felipe/repos/spatio-temporal-csph/2018SIGGRAPH_lindell_test_data/captured_min'
	raw_spad_data = scipy.io.loadmat(os.path.join(raw_spad_data_dir, scene_id+'.mat'))
	frame_num = 0 # frame number to use (some data files have multiple frames in them)
	intensity_imgs = np.asarray(raw_spad_data['cam_processed_data'])[0]
	intensity_img = np.asarray(intensity_imgs[frame_num]).astype(np.float32)


	## Load files
	cropped_inputs_result = np.load(os.path.join(cropped_inputs_result_dir, scene_id + '.npz'))['dep_re'].squeeze()
	padded_inputs_result = np.load(os.path.join(padded_inputs_result_dir, scene_id + '.npz'))['dep_re'].squeeze()

	## bring to same scale
	cropped_inputs_rec_bin = cropped_inputs_result*1024 
	padded_inputs_rec_bin = padded_inputs_result*1536

	## compute difference
	abs_diff = np.abs(cropped_inputs_rec_bin - padded_inputs_rec_bin[0:cropped_inputs_rec_bin.shape[0], 0:cropped_inputs_rec_bin.shape[1]])

	vmin = 0.05*1536
	vmax = 0.85*1536


	plt.clf()
	plt.subplot(2,2,1)
	plt.imshow(cropped_inputs_rec_bin, vmin=vmin, vmax=vmax); 
	plt.title('cropped_inputs_rec_bin')
	plt.colorbar()
	plt.subplot(2,2,2)
	plt.imshow(padded_inputs_rec_bin, vmin=vmin, vmax=vmax); 
	plt.title('padded_inputs_rec_bin')
	plt.colorbar()
	plt.subplot(2,2,3)
	plt.imshow(abs_diff, vmin=0, vmax=0.01*1536); 
	plt.title('absolute difference (bins)')
	plt.colorbar()
	plt.subplot(2,2,4)
	plt.imshow(intensity_img); 
	plt.title('Co-located intensity image')
	plt.colorbar()

	plot_utils.save_currfig_png(dirpath=out_results_dirpath, filename=scene_id)



