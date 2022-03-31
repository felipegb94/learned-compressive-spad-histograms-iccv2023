import numpy as np
import torch
import torch.nn as nn
from glob import glob
import pathlib
import scipy
import os
import scipy.io as scio
import time
# import h5py

## For debugging
from IPython.core import debugger
breakpoint = debugger.set_trace
import matplotlib.pyplot as plt


dtype = torch.cuda.FloatTensor
C = 3e8
Tp = 100e-9
def bin2depth(b):
	return b * Tp * C / 2

# test function for Middlebury dataset 
def test_sm(model, opt, outdir_m):

	rmse_all = []
	time_all = []

	with torch.no_grad():
		for name_test in glob(opt["testDataDir"] + "*.mat"):
			name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
			name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

			# print("Loading data {} and processing...".format(name_test_id))
			mat_data_file = scio.loadmat(name_test)
			## if depth is in keys this means the data was generations with SimulateTestMeasurements.m
			if("depth" in mat_data_file.keys()): 
				dep = mat_data_file["depth"]
			else:
				bin = mat_data_file["bin"] / 1023
				dep = bin2depth(bin)
			dep = np.asarray(dep).astype(np.float32)
			(h, w) = dep.shape

			# ## Old load from original PENonLocal
			# M_mea = mat_data_file["spad"]
			# M_mea = scipy.sparse.csc_matrix.todense(M_mea)
			# M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, h, w, -1])
			# M_mea = torch.from_numpy(np.transpose(M_mea, (0, 1, 4, 2, 3))).type(dtype)
			# M_mea_re, dep_re = model(M_mea)
			# dep_re = dep_re.data.cpu().numpy()[0, 0, :, :]
			# # dep_re = dep_re.transpose()

			## New load
			M_mea = mat_data_file["spad"]
			M_mea = scipy.sparse.csc_matrix.todense(M_mea)
			M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, w, h, -1])
			M_mea = torch.from_numpy(np.transpose(M_mea, (0, 1, 4, 2, 3))).type(dtype)
			M_mea_re, dep_re = model(M_mea)
			dep_re = dep_re.data.cpu().numpy()[0, 0, :, :].transpose()

			# t_s = time.time()
			# M_mea_re, dep_re = model(M_mea)
			# t_e = time.time()
			# time_all.append(t_e - t_s)


			dist = bin2depth(dep_re)
			rmse = np.sqrt(np.mean((dist - dep)**2))
			rmse_all.append(rmse)

			# scio.savemat(name_test_save, {"data":dist, "rmse":rmse})
			# print("The RMSE: {}".format(rmse))

			del M_mea_re
			torch.cuda.empty_cache()


			# plt.clf()
			# plt.subplot(1,2,1)
			# plt.imshow(dist, vmin=0, vmax=dist.max())
			# plt.title("Recovered Depth")
			# plt.colorbar()
			# plt.subplot(1,2,2)
			# plt.imshow(dep, vmin=0, vmax=dist.max())
			# plt.title("Ground Truth Depth")
			# plt.colorbar()
			# plt.pause(0.1)

	return np.mean(rmse_all), np.mean(time_all)


# test function for outdoor real-world dataset
def test_outrw(model, opt, outdir_m):
	rmse_all = [0,0]
	time_all = []
	base_pad = 16
	step = 32
	grab = 32
	dim = 64

	for name_test in glob(opt["testDataDir"] + "*.mat"):
		name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
		name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

		print("Loading data {} and processing...".format(name_test_id))
		M_mea_raw = np.asarray(scipy.io.loadmat(name_test)['y'])
		
		tp_s = 4770 
		t_inter =1024
		M_mea = M_mea_raw[:, :,tp_s:tp_s+t_inter]
		M_mea = M_mea.transpose((2, 0, 1))
		M_mea = torch.from_numpy(M_mea).unsqueeze(0).unsqueeze(0).type(dtype)

		out = np.zeros((M_mea.shape[1], M_mea.shape[2]))
		M_mea = torch.nn.functional.pad(M_mea, (base_pad, 0, base_pad, 0, 0, 0)) # pad only on edge

		t_s = time.clock()
		for i in tqdm(range(4)):
			for j in range(4):
				M_mea_input = M_mea[:, :, :, i*step:(i)*step+dim, j*step:(j)*step+dim]
				print("Size of input:{}".format(M_mea_input.shape))
				M_mea_re, dep_re = model(M_mea_input)
				M_mea_re = M_mea_re.data.cpu().numpy().squeeze()
				tile_out = np.argmax(M_mea_re, axis=0)  
				if i <= 14:
					out[i*step:(i+1)*step, j*step:(j+1)*step] = tile_out[step:step+step, step:step+step]
				else:
					out[i*step:(i+1)*step, j*step:(j+1)*step] = tile_out[16:16+step, 16:16+step]
		
		t_e = time.clock()
		time_all.append(t_e - t_s)

		dist = out.astype(np.float32) * 0.15

		scio.savemat(name_test_save, {"data":dist})
	
	return np.mean(rmse_all), np.mean(time_all)


# test function for indoor real-world data
def test_inrw(model, opt, outdir_m):

	rmse_all = [0,0]
	time_all = []
	base_pad = 16
	step = 16
	grab = 32
	dim = 64

	for name_test in glob(opt["testDataDir"] + "*.mat"):
		name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
		name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

		print("Loading data {} and processing...".format(name_test_id))
		M_mea = scio.loadmat(name_test)["spad_processed_data"]
		M_mea = scipy.sparse.csc_matrix.todense(M_mea)
		M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, 1536, 256, 256])
		M_mea = M_mea.transpose((0,1,2,4,3)).type(dtype)

		out = np.zeros((M_mea.shape[3], M_mea.shape[4]))
		M_mea = torch.nn.functional.pad(M_mea, (base_pad, 0, base_pad, 0, 0, 0))

		t_s = time.time()
		for i in tqdm(range(16)):
			for j in range(16):
				M_mea_input = M_mea[:, :, :, i*step:(i)*step+dim, j*step:(j)*step+dim]
				M_mea_re, dep_re = model(M_mea_input)
				M_mea_re = M_mea_re.data.cpu().numpy().squeeze()
				tile_out = np.argmax(M_mea_re, axis=0)
				out[i*step:(i+1)*step, j*step:(j+1)*step] = \
					tile_out[step//2:step//2+step, step//2:step//2+step]

		t_e = time.time()
		time_all.append(t_e - t_s)
		
		dist = out * 6 / 1536.

		scio.savemat(name_test_save, {"data":dist})

	return np.mean(rmse_all), np.mean(time_all)

   

def save_currfig( dirpath = '.', filename = 'curr_fig', file_ext = 'png', use_imsave=False  ):
	# Create directory to store figure if it does not exist
	os.makedirs(dirpath, exist_ok=True)
	# Pause to make sure plot is fully rendered and not warnings or errors are thown
	plt.pause(0.02)
	# If filename contains file extension then ignore the input file ext
	# Else add the input file etension
	if('.{}'.format(file_ext) in  filename): filepath = os.path.join(dirpath, filename)
	else: filepath = os.path.join(dirpath, filename) + '.{}'.format(file_ext)
	plt.savefig(filepath, 
				dpi=None, 
				# facecolor='w', 
				# edgecolor='w',
				# orientation='portrait', 
				# papertype=None, 
				transparent=True, 
				bbox_inches='tight', 
				# pad_inches=0.1,
				# metadata=None 
				format=file_ext
				)