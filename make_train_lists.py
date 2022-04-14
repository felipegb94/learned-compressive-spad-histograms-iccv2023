#### Standard Library Imports
import os
from glob import glob

#### Library imports
from omegaconf import DictConfig, OmegaConf
import hydra
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports

def generate_spad_data_fpaths(dirpaths):
	'''
		Go through each dirpath in the dirpaths list, and find all files with prefix spad_ and that are .mat files
	'''
	spad_data_fpaths = []
	for dirpath in dirpaths:
		pattern = os.path.join(dirpath, 'spad_*.mat')
		fpaths = glob(pattern)
		spad_data_fpaths.extend(fpaths)
	return spad_data_fpaths

def write_fpaths(out_fpath, fpaths):
	with open(out_fpath, 'w') as f:
		for file in fpaths:
			f.write(file + '\n')


@hydra.main(config_path="./conf", config_name="io_dirpaths")
def make_train_lists(cfg):
	## Load dir paths
	datalists_dirpath = cfg.io_dirpaths.datalists_dirpath
	spad_dataset_dirpath = cfg.io_dirpaths.nyuv2_spad_dataset_dirpath
	spad_dataset_dirpath = os.path.abspath(spad_dataset_dirpath)
	train_datalists_fname = cfg.io_dirpaths.nyuv2_train_list_fname
	val_datalists_fname = cfg.io_dirpaths.nyuv2_val_list_fname
	train_datalists_fpath = os.path.join(datalists_dirpath, train_datalists_fname)
	val_datalists_fpath = os.path.join(datalists_dirpath, val_datalists_fname)

	print(f"Datalists stored in: {datalists_dirpath}")
	print(f"spad_dataset_dirpath stored in: {spad_dataset_dirpath}")

	## Set the dataset name
	train_dataset_fname = os.path.splitext(train_datalists_fname)[0] + '_' + os.path.basename(spad_dataset_dirpath) + '.txt'
	val_dataset_fname = os.path.splitext(val_datalists_fname)[0] + '_' + os.path.basename(spad_dataset_dirpath) + '.txt'

	# Read the train split
	# with open(os.path.join(datalists_dirpath, 'nyuv2_train.txt')) as f:
	with open(train_datalists_fpath) as f:
		train_files_dirs = f.read().split()
	# Concatenate the full directory path
	train_files_dirs = [os.path.join(spad_dataset_dirpath, curr_dir) for curr_dir in train_files_dirs]
	# Get all filepaths for SPAD files inside the list of directories
	train_fpaths = generate_spad_data_fpaths(train_files_dirs)
	print('Writing training files')
	write_fpaths(os.path.join(datalists_dirpath, train_dataset_fname), train_fpaths)

	# Read the val split
	with open(val_datalists_fpath) as f:
		val_files_dirs = f.read().split()
	# Concatenate the full directory path
	val_files_dirs = [os.path.join(spad_dataset_dirpath, curr_dir) for curr_dir in val_files_dirs]
	# Get all filepaths for SPAD files inside the list of directories
	val_fpaths = generate_spad_data_fpaths(val_files_dirs)
	print('Writing val files')
	write_fpaths(os.path.join(datalists_dirpath, val_dataset_fname), val_fpaths)

	print('Wrote {} train, {} validation files'.format(len(train_fpaths), len(val_fpaths)))

if __name__=='__main__':
	make_train_lists()



