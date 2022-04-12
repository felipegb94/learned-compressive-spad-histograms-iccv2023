#### Standard Library Imports
from asyncore import write
import os
from glob import glob

#### Library imports
import numpy as np

from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from research_utils import io_ops
from make_train_lists import *

if __name__=='__main__':
    ## Load dir paths
    io_dirpaths = io_ops.load_json('io_dirpaths.json')
    datalists_dirpath = io_dirpaths['datalists_dirpath']
    spad_dataset_dirpath = io_dirpaths['test_spad_dataset_dirpath']
    spad_dataset_dirpath = os.path.abspath(spad_dataset_dirpath)

    ## Set the dataset name
    dataset_name = 'middlebury_' + os.path.basename(spad_dataset_dirpath)

    ## Input single dirpath to look into
    test_fpaths = generate_spad_data_fpaths([spad_dataset_dirpath])

    print('Writing test files')
    write_fpaths(os.path.join(datalists_dirpath, 'test_'+dataset_name+'.txt'), test_fpaths)


    print('Wrote {} test'.format(len(test_fpaths)))