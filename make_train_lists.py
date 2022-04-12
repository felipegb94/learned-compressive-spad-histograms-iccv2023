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

if __name__=='__main__':
    ## Load dir paths
    io_dirpaths = io_ops.load_json('io_dirpaths.json')
    datalists_dirpath = io_dirpaths['datalists_dirpath']
    spad_dataset_dirpath = io_dirpaths['train_spad_dataset_dirpath']
    spad_dataset_dirpath = os.path.abspath(spad_dataset_dirpath)

    ## Set the dataset name
    dataset_name = 'nyuv2_' + os.path.basename(spad_dataset_dirpath)

    # Read the train split
    with open(os.path.join(datalists_dirpath, 'nyuv2_train.txt')) as f:
        train_files_dirs = f.read().split()
    # Concatenate the full directory path
    train_files_dirs = [os.path.join(spad_dataset_dirpath, curr_dir) for curr_dir in train_files_dirs]
    # Get all filepaths for SPAD files inside the list of directories
    train_fpaths = generate_spad_data_fpaths(train_files_dirs)
    print('Writing training files')
    write_fpaths(os.path.join(datalists_dirpath, 'train_'+dataset_name+'.txt'), train_fpaths)

    # Read the val split
    with open(os.path.join(datalists_dirpath, 'nyuv2_val.txt')) as f:
        val_files_dirs = f.read().split()
    # Concatenate the full directory path
    val_files_dirs = [os.path.join(spad_dataset_dirpath, curr_dir) for curr_dir in val_files_dirs]
    # Get all filepaths for SPAD files inside the list of directories
    val_fpaths = generate_spad_data_fpaths(val_files_dirs)
    print('Writing val files')
    write_fpaths(os.path.join(datalists_dirpath, 'val_'+dataset_name+'.txt'), val_fpaths)

    print('Wrote {} train, {} validation files'.format(len(train_fpaths), len(val_fpaths)))