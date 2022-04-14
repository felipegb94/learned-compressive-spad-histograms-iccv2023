#### Standard Library Imports
import os

#### Library imports
import hydra
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from make_train_lists import generate_spad_data_fpaths, write_fpaths


@hydra.main(config_path="./conf", config_name="io_dirpaths")
def make_test_lists(cfg):
	## Load dir paths
	datalists_dirpath = cfg.io_dirpaths.datalists
	spad_dataset_dirpath = cfg.io_dirpaths.middlebury_spad_dataset
	spad_dataset_dirpath = os.path.abspath(spad_dataset_dirpath)

	## Set the dataset name
	dataset_name = 'middlebury_' + os.path.basename(spad_dataset_dirpath)

	## Input single dirpath to look into
	test_fpaths = generate_spad_data_fpaths([spad_dataset_dirpath])

	print('Writing test files')
	write_fpaths(os.path.join(datalists_dirpath, 'test_'+dataset_name+'.txt'), test_fpaths)


	print('Wrote {} test'.format(len(test_fpaths)))

if __name__=='__main__':
	make_test_lists()

