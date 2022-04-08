# The train, validation list make function
import numpy as np
from glob import glob
import re
import os.path
import sys
## For debugging
from IPython.core import debugger
breakpoint = debugger.set_trace

# # specify dataset folder here that contains the output of SimulateTrainMeasurements.m
# dataset_folder = os.path.abspath('../../data_gener/TrainData/processed') + '/'
# intensity_dataset_folder = os.path.abspath('../../data_gener/TrainData/processed') + '/'

base_dataset_folder = '/home/felipe/repos/spatio-temporal-csph/data_gener/TrainData'
spad_dataset_folder = os.path.join(base_dataset_folder, 'SimSPADDataset_nr-64_nc-64_nt-1024_tres-80ps_dark-1_psf-1') + '/'
intensity_dataset_folder = os.path.join(base_dataset_folder, 'processed') + '/'

simulation_param_idx = 1    # 1 or 2 corresponding to that in SimulateTrainMeasurements.m

def remove_base_dirpath_to_fpaths(base_dirpath, fpaths):
	if(base_dirpath[-1] != '/'):
		base_dirpath = base_dirpath + '/'
	return [fpath.replace(base_dirpath, '') for fpath in fpaths]

def add_base_dirpath_to_fpaths(base_dirpath, fpaths):
	return [os.path.join(base_dirpath, fpath) for fpath in fpaths]

def intersect_files(train_files):
	'''
		previous: intersect files assumed spad+intensity are in same folder
		new: intersect files does not assume this
	'''
	intensity_train_files = []
	intensity_train_files = []
	for t in train_files:
		intensity_fpath_pattern = os.path.join(intensity_dataset_folder, t, 'intensity*.mat')
		intensity_train_files.append(glob(intensity_fpath_pattern))
	intensity_train_files = [file for sublist in intensity_train_files for file in sublist]
	intensity_train_file_ids = remove_base_dirpath_to_fpaths(intensity_dataset_folder, intensity_train_files)

	spad_train_files = []
	spad_train_file_ids = []
	if simulation_param_idx is not None:
		noise_param = [simulation_param_idx]
	else:
		print("Specify simulation_param_idx = 1 or 2")
		sys.exit("SIMULATION PARAMETER INDEX ERROR")
	for p in noise_param:
		spad_train_files.append([])
		spad_train_file_ids.append([])
		for t in train_files:
			spad_fpath_pattern = os.path.join(spad_dataset_folder, t, 'spad*p{}.mat'.format(p))
			spad_train_files[-1].append(glob(spad_fpath_pattern))
		spad_train_files[-1] = [file for sublist in spad_train_files[-1] for file in sublist]
		spad_train_files[-1] = [re.sub(r'(.*)/spad_(.*)_p.*.mat', r'\1/intensity_\2.mat',
								 file) for file in spad_train_files[-1]]
		spad_train_file_ids[-1] = remove_base_dirpath_to_fpaths(spad_dataset_folder, spad_train_files[-1])
	for idx, p in enumerate(noise_param):
		spad_train_files[idx] = set(spad_train_files[idx])
		spad_train_file_ids[idx] = set(spad_train_file_ids[idx])

	intensity_train_files = set(intensity_train_files) 
	intensity_train_file_ids = set(intensity_train_file_ids) 

	intensity_train_files = intensity_train_files.intersection(*tuple(spad_train_files))
	intensity_train_file_ids = intensity_train_file_ids.intersection(*tuple(spad_train_file_ids))
	
	intensity_train_files = add_base_dirpath_to_fpaths(spad_dataset_folder, intensity_train_file_ids)
	# spad_train_files = []
	# for idx, p in enumerate(noise_param):
	# 	spad_train_files.append([])
	# 	for intensity_file_id in intensity_train_file_ids:
	# 		spad_file_id = intensity_file_id.replace('intensity', 'spad').replace('.mat', '_p{}.mat'.format(p)) 
	# 		spad_train_files[-1].append(os.path.join(spad_dataset_folder, spad_file_id))
	return intensity_train_files

def main():

	with open('train.txt') as f:
		train_files = f.read().split()
	with open('val.txt') as f:
		val_files = f.read().split()

	print('Sorting training files')
	intensity_train_files = intersect_files(train_files)
	print('Sorting validation files')
	intensity_val_files = intersect_files(val_files)

	print('Writing training files')
	with open('train_intensity.txt', 'w') as f:
		for file in intensity_train_files:
			f.write(file + '\n')
	print('Writing validation files')
	with open('val_intensity.txt', 'w') as f:
		for file in intensity_val_files:
			f.write(file + '\n')

	print('Wrote {} train, {} validation files'.format(len(intensity_train_files),
													   len(intensity_val_files)))
	return

if __name__ == '__main__':
	main()





