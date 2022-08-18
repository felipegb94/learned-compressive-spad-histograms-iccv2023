'''
	This script makes the datalists for all the test datasets used in the paper. Before running the script make sure to download and/or simulate the datasets
	The datasets include:
		- Our own simulated middlebury test dataset
		- Lindell et al., 2018 SIGGRAPH real dataset captured with a Linospad
		- Lindell et al., 2018 SIGGRAPH simulated middlebury test dataset. This is the same as our middlebury dataset but the noise levels may differ since they are different runs.
'''

#### Standard Library Imports
import os
from glob import glob

#### Library imports
import hydra
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from make_train_lists import generate_spad_data_fpaths, write_fpaths


def generate_matfile_data_fpaths(dirpaths):
	'''
		Go through each dirpath in the dirpaths list, and find all files with prefix spad_ and that are .mat files
	'''
	spad_data_fpaths = []
	for dirpath in dirpaths:
		pattern = os.path.join(dirpath, '*.mat')
		fpaths = glob(pattern)
		spad_data_fpaths.extend(fpaths)
	return spad_data_fpaths

@hydra.main(config_path="./conf", config_name="io_dirpaths")
def make_test_lists(cfg):
	## Load dirpaths
	datalists_dirpath = cfg.io_dirpaths.datalists_dirpath
	middlebury_spad_dataset_dirpath = cfg.io_dirpaths.middlebury_spad_dataset_dirpath
	middlebury_spad_dataset_dirpath = os.path.abspath(middlebury_spad_dataset_dirpath)
	lindell_linospad_dataset_dirpath = cfg.io_dirpaths.lindell_linospad_dataset_dirpath
	lindell_linospad_dataset_dirpath = os.path.abspath(lindell_linospad_dataset_dirpath)
	lindell_middlebury_spad_dataset_dirpath = cfg.io_dirpaths.lindell_middlebury_spad_dataset_dirpath
	lindell_middlebury_spad_dataset_dirpath = os.path.abspath(lindell_middlebury_spad_dataset_dirpath)

	## Input single dirpath to look into
	middlebury_spad_dataset_fpaths = generate_spad_data_fpaths([middlebury_spad_dataset_dirpath])
	lindell_linospad_dataset_fpaths = generate_matfile_data_fpaths([lindell_linospad_dataset_dirpath])
	lindell_middlebury_spad_dataset_fpaths = generate_matfile_data_fpaths([lindell_middlebury_spad_dataset_dirpath])

	## Set the dataset name
	middlebury_spad_dataset_name = 'middlebury_' + os.path.basename(middlebury_spad_dataset_dirpath)
	lindell_linospad_dataset_name = 'lindell_linospad_' + os.path.basename(lindell_linospad_dataset_dirpath)
	lindell_middlebury_spad_dataset_name = 'lindell_middlebury_' + os.path.basename(lindell_middlebury_spad_dataset_dirpath)

	print('Writing test files')
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+middlebury_spad_dataset_name+'.txt'), middlebury_spad_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(middlebury_spad_dataset_fpaths), middlebury_spad_dataset_name))
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+lindell_linospad_dataset_name+'.txt'), lindell_linospad_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(lindell_linospad_dataset_fpaths), lindell_linospad_dataset_name))
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+lindell_middlebury_spad_dataset_name+'.txt'), lindell_middlebury_spad_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(lindell_middlebury_spad_dataset_fpaths), lindell_middlebury_spad_dataset_name))

if __name__=='__main__':
	make_test_lists()

