'''
	This script makes the datalists for all the test datasets used in the paper. Before running the script make sure to download and/or simulate the datasets
	The datasets include:
		- Our own simulated middlebury test dataset
		- Our own simulated middlebury test dataset with low SBR levels
		- Our own simulated middlebury test dataset with high Signal levels
		- Our own simulated middlebury test dataset with different pulse widths
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
	## Input single dirpath to look into
	middlebury_spad_dataset_fpaths = generate_spad_data_fpaths([middlebury_spad_dataset_dirpath])
	## Set the dataset name
	middlebury_spad_dataset_name = 'middlebury_' + os.path.basename(middlebury_spad_dataset_dirpath)
	print('Writing test files')
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+middlebury_spad_dataset_name+'.txt'), middlebury_spad_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(middlebury_spad_dataset_fpaths), middlebury_spad_dataset_name))

	## Repeat for Middlebury low SBR dataset
	middlebury_lowsbr_spad_dataset_dirpath = cfg.io_dirpaths.middlebury_lowsbr_spad_dataset_dirpath
	middlebury_lowsbr_spad_dataset_dirpath = os.path.abspath(middlebury_lowsbr_spad_dataset_dirpath)
	## Input single dirpath to look into
	middlebury_lowsbr_spad_dataset_fpaths = generate_spad_data_fpaths([middlebury_lowsbr_spad_dataset_dirpath])
	## Set the dataset name
	middlebury_lowsbr_spad_dataset_name = 'middlebury_lowsbr_' + os.path.basename(middlebury_lowsbr_spad_dataset_dirpath)
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+middlebury_lowsbr_spad_dataset_name+'.txt'), middlebury_lowsbr_spad_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(middlebury_lowsbr_spad_dataset_fpaths), middlebury_lowsbr_spad_dataset_name))

	## Repeat for Middlebury low SBR dataset
	middlebury_highsignal_spad_dataset_dirpath = cfg.io_dirpaths.middlebury_highsignal_spad_dataset_dirpath
	middlebury_highsignal_spad_dataset_dirpath = os.path.abspath(middlebury_highsignal_spad_dataset_dirpath)
	## Input single dirpath to look into
	middlebury_highsignal_spad_dataset_fpaths = generate_spad_data_fpaths([middlebury_highsignal_spad_dataset_dirpath])
	## Set the dataset name
	middlebury_highsignal_spad_dataset_name = 'middlebury_highsignal_' + os.path.basename(middlebury_highsignal_spad_dataset_dirpath)
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+middlebury_highsignal_spad_dataset_name+'.txt'), middlebury_highsignal_spad_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(middlebury_highsignal_spad_dataset_fpaths), middlebury_highsignal_spad_dataset_name))

	## Repeat for Middlebury narrow pulse dataset
	middlebury_narrowpulse_spad_dataset_dirpath = cfg.io_dirpaths.middlebury_narrowpulse_spad_dataset_dirpath
	middlebury_narrowpulse_spad_dataset_dirpath = os.path.abspath(middlebury_narrowpulse_spad_dataset_dirpath)
	## Input single dirpath to look into
	middlebury_narrowpulse_spad_dataset_fpaths = generate_spad_data_fpaths([middlebury_narrowpulse_spad_dataset_dirpath])
	## Set the dataset name
	middlebury_narrowpulse_spad_dataset_name = 'middlebury_narrowpulse_' + os.path.basename(middlebury_narrowpulse_spad_dataset_dirpath)
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+middlebury_narrowpulse_spad_dataset_name+'.txt'), middlebury_narrowpulse_spad_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(middlebury_narrowpulse_spad_dataset_fpaths), middlebury_narrowpulse_spad_dataset_name))

	## Repeat for Middlebury wide pulse dataset
	middlebury_widepulse_spad_dataset_dirpath = cfg.io_dirpaths.middlebury_widepulse_spad_dataset_dirpath
	middlebury_widepulse_spad_dataset_dirpath = os.path.abspath(middlebury_widepulse_spad_dataset_dirpath)
	## Input single dirpath to look into
	middlebury_widepulse_spad_dataset_fpaths = generate_spad_data_fpaths([middlebury_widepulse_spad_dataset_dirpath])
	## Set the dataset name
	middlebury_widepulse_spad_dataset_name = 'middlebury_widepulse_' + os.path.basename(middlebury_widepulse_spad_dataset_dirpath)
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+middlebury_widepulse_spad_dataset_name+'.txt'), middlebury_widepulse_spad_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(middlebury_widepulse_spad_dataset_fpaths), middlebury_widepulse_spad_dataset_name))

	## Repeat for Middlebury Large Depths dataset
	middlebury_largedepth_spad_dataset_dirpath = cfg.io_dirpaths.middlebury_largedepth_spad_dataset_dirpath
	middlebury_largedepth_spad_dataset_dirpath = os.path.abspath(middlebury_largedepth_spad_dataset_dirpath)
	## Input single dirpath to look into
	middlebury_largedepth_spad_dataset_fpaths = generate_spad_data_fpaths([middlebury_largedepth_spad_dataset_dirpath])
	## Set the dataset name
	middlebury_largedepth_spad_dataset_name = 'middlebury_largedepth_' + os.path.basename(middlebury_largedepth_spad_dataset_dirpath)
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+middlebury_largedepth_spad_dataset_name+'.txt'), middlebury_largedepth_spad_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(middlebury_largedepth_spad_dataset_fpaths), middlebury_largedepth_spad_dataset_name))

	## Repeat for Middlebury Masked High Time Bins dataset
	middlebury_maskedhightimebins_spad_dataset_dirpath = cfg.io_dirpaths.middlebury_maskedhightimebins_spad_dataset_dirpath
	middlebury_maskedhightimebins_spad_dataset_dirpath = os.path.abspath(middlebury_maskedhightimebins_spad_dataset_dirpath)
	## Input single dirpath to look into
	middlebury_maskedhightimebins_spad_dataset_fpaths = generate_spad_data_fpaths([middlebury_maskedhightimebins_spad_dataset_dirpath])
	## Set the dataset name
	middlebury_maskedhightimebins_spad_dataset_name = 'middlebury_maskedhightimebins_' + os.path.basename(middlebury_maskedhightimebins_spad_dataset_dirpath)
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+middlebury_maskedhightimebins_spad_dataset_name+'.txt'), middlebury_maskedhightimebins_spad_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(middlebury_maskedhightimebins_spad_dataset_fpaths), middlebury_maskedhightimebins_spad_dataset_name))

	## Repeat for linospad dataset
	lindell_linospad_dataset_dirpath = cfg.io_dirpaths.lindell_linospad_dataset_dirpath
	lindell_linospad_dataset_dirpath = os.path.abspath(lindell_linospad_dataset_dirpath)
	lindell_linospad_dataset_fpaths = generate_matfile_data_fpaths([lindell_linospad_dataset_dirpath])
	lindell_linospad_dataset_name = 'lindell2018_linospad_' + os.path.basename(lindell_linospad_dataset_dirpath)
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+lindell_linospad_dataset_name+'.txt'), lindell_linospad_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(lindell_linospad_dataset_fpaths), lindell_linospad_dataset_name))
	
	## Repeat for linospad dataset that was divided into patches
	lindell_linospad_patch_dataset_dirpath = cfg.io_dirpaths.lindell_linospad_patch_dataset_dirpath
	lindell_linospad_patch_dataset_dirpath = os.path.abspath(lindell_linospad_patch_dataset_dirpath)
	lindell_linospad_patch_dataset_fpaths = generate_matfile_data_fpaths([lindell_linospad_patch_dataset_dirpath])
	lindell_linospad_patch_dataset_name = 'lindell2018_linospad_' + os.path.basename(lindell_linospad_patch_dataset_dirpath)
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+lindell_linospad_patch_dataset_name+'.txt'), lindell_linospad_patch_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(lindell_linospad_patch_dataset_fpaths), lindell_linospad_patch_dataset_name))

	## Repeat for linospad small dataset
	lindell_linospad_min_dataset_dirpath = cfg.io_dirpaths.lindell_linospad_min_dataset_dirpath
	lindell_linospad_min_dataset_dirpath = os.path.abspath(lindell_linospad_min_dataset_dirpath)
	lindell_linospad_min_dataset_fpaths = generate_matfile_data_fpaths([lindell_linospad_min_dataset_dirpath])
	lindell_linospad_min_dataset_name = 'lindell2018_linospad_' + os.path.basename(lindell_linospad_min_dataset_dirpath)
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+lindell_linospad_min_dataset_name+'.txt'), lindell_linospad_min_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(lindell_linospad_min_dataset_fpaths), lindell_linospad_min_dataset_name))

	## Repeat for Lindell et al., 2018 middlebury dataset 
	lindell_middlebury_spad_dataset_dirpath = cfg.io_dirpaths.lindell_middlebury_spad_dataset_dirpath
	lindell_middlebury_spad_dataset_dirpath = os.path.abspath(lindell_middlebury_spad_dataset_dirpath)
	lindell_middlebury_spad_dataset_fpaths = generate_matfile_data_fpaths([lindell_middlebury_spad_dataset_dirpath])
	lindell_middlebury_spad_dataset_name = 'lindell2018_middlebury_' + os.path.basename(lindell_middlebury_spad_dataset_dirpath)
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+lindell_middlebury_spad_dataset_name+'.txt'), lindell_middlebury_spad_dataset_fpaths)
	print('Wrote {} test file for {} dataset'.format(len(lindell_middlebury_spad_dataset_fpaths), lindell_middlebury_spad_dataset_name))

if __name__=='__main__':
	make_test_lists()

